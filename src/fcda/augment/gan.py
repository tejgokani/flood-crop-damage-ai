"""Conditional DCGAN augmentation for the image pipeline -- TRAINING split only.

Ma'am's image sequence places "GAN Augmentation on TRAINING data only" after the split, for
the same reason SMOTE sits there in the CSV sequence: a generator that has seen the test
tiles will happily reproduce them, and the resulting score is meaningless.

The literature motivation is explicit. Paper 3 names "the scarcity of temporally and
spatially aligned SAR and optical data" as the problem it had to work around, and Paper 5
lists "dataset expansion" as its first future-work item. This module is our answer to gap G1.

Design notes:

* **Conditional on severity.** An unconditional GAN would mostly generate Healthy tiles,
  since that is what the data mostly contains -- exactly the wrong thing for an imbalanced
  problem. Conditioning lets us ask specifically for Severe samples.
* **Generates the mask alongside the image.** The segmentation head needs a target, so the
  generator emits a 4-channel tensor: three SAR channels plus the flood mask. A synthetic
  image with no mask would only be usable by the classification head.
* **Kept deliberately small.** A DCGAN at 64x64 trains in minutes on an M3; a StyleGAN would
  eat the entire night for a component that is one step of the pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from .. import NUM_CLASSES

GAN_RESOLUTION = 64
LATENT_DIM = 100


class Generator(nn.Module):
    """z + class embedding -> 4 x 64 x 64 (3 SAR channels + flood mask)."""

    def __init__(self, latent_dim: int = LATENT_DIM, n_classes: int = NUM_CLASSES, ngf: int = 64):
        super().__init__()
        self.embed = nn.Embedding(n_classes, latent_dim)
        self.net = nn.Sequential(
            nn.ConvTranspose2d(latent_dim * 2, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf, 4, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        h = torch.cat([z, self.embed(y)], dim=1)[:, :, None, None]
        return self.net(h)


class Discriminator(nn.Module):
    """Projection-style conditional discriminator."""

    def __init__(self, n_classes: int = NUM_CLASSES, ndf: int = 64):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(4, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, True),
        )
        self.head = nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False)
        self.embed = nn.Embedding(n_classes, ndf * 8)
        # The projection term is scaled down: at full weight it dominates the real/fake
        # logit early in training and the discriminator wins outright, starving the
        # generator of gradient.
        self.proj_scale = 0.1

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        h = self.features(x)
        out = self.head(h).flatten(1).squeeze(1)
        # Projection term: encourages the critic to use the class, not ignore it.
        proj = (self.embed(y) * h.mean(dim=(2, 3))).sum(dim=1)
        return out + self.proj_scale * proj


@dataclass
class GanTrainResult:
    epochs: int
    d_loss: float
    g_loss: float
    seconds: float
    n_train_tiles: int


def _to_gan_tensor(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Pack image+mask into the generator's 4-channel [-1, 1] layout."""
    import cv2

    img = np.stack(
        [cv2.resize(c, (GAN_RESOLUTION, GAN_RESOLUTION), interpolation=cv2.INTER_AREA) for c in img]
    )
    m = cv2.resize(
        mask[0], (GAN_RESOLUTION, GAN_RESOLUTION), interpolation=cv2.INTER_NEAREST
    )[None]
    packed = np.concatenate([img, m], axis=0).astype(np.float32)
    lo, hi = packed.min(), packed.max()
    packed = (packed - lo) / (hi - lo + 1e-6)
    return packed * 2.0 - 1.0


def train_gan(
    train_samples: list[tuple[np.ndarray, np.ndarray, int]],
    epochs: int = 30,
    batch_size: int = 32,
    lr: float = 2e-4,
    device: str = "cpu",
    max_seconds: float = 1200.0,
    seed: int = 42,
    verbose: bool = True,
) -> tuple[Generator, GanTrainResult]:
    """Train the conditional DCGAN on training-split tiles only.

    ``train_samples`` must come from the training indices. There is deliberately no split
    argument here -- the caller does the slicing, so this function has no way to reach the
    evaluation data.
    """
    import time

    torch.manual_seed(seed)
    dev = torch.device(device)
    g, d = Generator().to(dev), Discriminator().to(dev)
    opt_g = torch.optim.Adam(g.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(d.parameters(), lr=lr, betas=(0.5, 0.999))
    crit = nn.BCEWithLogitsLoss()

    packed = np.stack([_to_gan_tensor(img, msk) for img, msk, _ in train_samples])
    labels = np.array([lab for _, _, lab in train_samples], dtype=np.int64)
    x_all = torch.from_numpy(packed).float()
    y_all = torch.from_numpy(labels)

    n = len(x_all)
    t0 = time.time()
    d_loss = g_loss = 0.0
    done = 0

    for ep in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]
            xb, yb = x_all[idx].to(dev), y_all[idx].to(dev)
            bs = len(xb)

            # --- discriminator
            z = torch.randn(bs, LATENT_DIM, device=dev)
            fake = g(z, yb).detach()
            loss_d = crit(d(xb, yb), torch.ones(bs, device=dev) * 0.9) + crit(
                d(fake, yb), torch.zeros(bs, device=dev)
            )
            opt_d.zero_grad(set_to_none=True)
            loss_d.backward()
            opt_d.step()

            # --- generator
            z = torch.randn(bs, LATENT_DIM, device=dev)
            gen = g(z, yb)
            loss_g = crit(d(gen, yb), torch.ones(bs, device=dev))
            opt_g.zero_grad(set_to_none=True)
            loss_g.backward()
            opt_g.step()

            d_loss, g_loss = float(loss_d.detach()), float(loss_g.detach())

        done = ep + 1
        if verbose and (done % 5 == 0 or done == epochs):
            print(f"  [gan] epoch {done}/{epochs}  d={d_loss:.3f}  g={g_loss:.3f}", flush=True)
        if time.time() - t0 > max_seconds:
            if verbose:
                print(f"  [gan] wall-clock cap hit after {done} epochs", flush=True)
            break

    return g, GanTrainResult(done, d_loss, g_loss, time.time() - t0, n)


@torch.no_grad()
def generate_samples(
    g: Generator,
    counts: dict[int, int],
    size: int,
    device: str = "cpu",
    seed: int = 0,
) -> list[tuple[np.ndarray, np.ndarray, int]]:
    """Synthesise ``counts[cls]`` tiles of each severity class, resized to ``size``.

    Returned in exactly the same ``(image, mask, label)`` form as real samples, so the
    training loop cannot tell them apart.
    """
    import cv2

    torch.manual_seed(seed)
    g.eval()
    dev = torch.device(device)
    out: list[tuple[np.ndarray, np.ndarray, int]] = []

    for cls, n in counts.items():
        if n <= 0:
            continue
        for i in range(0, n, 64):
            bs = min(64, n - i)
            z = torch.randn(bs, LATENT_DIM, device=dev)
            y = torch.full((bs,), cls, dtype=torch.long, device=dev)
            gen = g(z, y).cpu().numpy()
            for sample in gen:
                sample = (sample + 1.0) / 2.0
                img = np.stack(
                    [cv2.resize(c, (size, size), interpolation=cv2.INTER_LINEAR) for c in sample[:3]]
                )
                mask = cv2.resize(sample[3], (size, size), interpolation=cv2.INTER_LINEAR)
                out.append((img.astype(np.float32), (mask > 0.5).astype(np.float32)[None], cls))
    return out


def deficit_counts(labels: np.ndarray, cap: int | None = None) -> dict[int, int]:
    """How many synthetic tiles each class needs to reach the majority class count."""
    counts = {int(c): int((labels == c).sum()) for c in range(NUM_CLASSES)}
    target = max(counts.values()) if counts else 0
    if cap is not None:
        target = min(target, cap)
    return {c: max(0, target - n) for c, n in counts.items()}
