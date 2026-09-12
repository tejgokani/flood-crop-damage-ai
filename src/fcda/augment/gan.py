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

#: The generator emits a full pre/post *pair* plus the mask: 3 pre + 3 post + 1 mask.
#:
#: It used to emit 4 channels -- one post-flood image and its mask -- which no longer matches
#: what either model consumes. YOLO12 takes a 6-channel [post, post-pre] change stack and
#: CNN+LSTM takes an ordered [pre, post] sequence, so 3-channel single-date output could not be
#: fed to either and the augmentation step was quietly skipped. Generating the pair serves both:
#: the change model differences it, the temporal model sequences it.
GAN_CHANNELS = 7


class Generator(nn.Module):
    """z + class embedding -> 7 x 64 x 64 (3 pre + 3 post SAR channels + flood mask)."""

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
            nn.ConvTranspose2d(ngf, GAN_CHANNELS, 4, 2, 1, bias=False),
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
            nn.Conv2d(GAN_CHANNELS, ndf, 4, 2, 1, bias=False),
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


def _resize_stack(arr: np.ndarray, interp) -> np.ndarray:
    import cv2

    return np.stack(
        [cv2.resize(c, (GAN_RESOLUTION, GAN_RESOLUTION), interpolation=interp) for c in arr]
    )


def _to_gan_tensor(pre: np.ndarray, post: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Pack a pre/post pair and its mask into the generator's 7-channel [-1, 1] layout."""
    import cv2

    pre_r = _resize_stack(pre, cv2.INTER_AREA)
    post_r = _resize_stack(post, cv2.INTER_AREA)
    m = _resize_stack(mask, cv2.INTER_NEAREST)
    packed = np.concatenate([pre_r, post_r, m], axis=0).astype(np.float32)
    lo, hi = packed.min(), packed.max()
    packed = (packed - lo) / (hi - lo + 1e-6)
    return packed * 2.0 - 1.0


def train_gan(
    train_samples: list[tuple[np.ndarray, np.ndarray, np.ndarray, int]],
    epochs: int = 30,
    batch_size: int = 32,
    lr: float = 2e-4,
    device: str = "cpu",
    max_seconds: float = 1200.0,
    seed: int = 42,
    verbose: bool = True,
) -> tuple[Generator, GanTrainResult]:
    """Train the conditional DCGAN on training-split tiles only.

    ``train_samples`` is a list of ``(pre, post, mask, label)`` drawn from the training indices.
    There is deliberately no split argument here -- the caller does the slicing, so this
    function has no way to reach the evaluation data.
    """
    import time

    torch.manual_seed(seed)
    dev = torch.device(device)
    g, d = Generator().to(dev), Discriminator().to(dev)
    opt_g = torch.optim.Adam(g.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(d.parameters(), lr=lr, betas=(0.5, 0.999))
    crit = nn.BCEWithLogitsLoss()

    packed = np.stack([_to_gan_tensor(pre, post, msk) for pre, post, msk, _ in train_samples])
    labels = np.array([lab for *_, lab in train_samples], dtype=np.int64)
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
    mode: str = "change",
) -> list[tuple[np.ndarray, np.ndarray, int, float]]:
    """Synthesise ``counts[cls]`` tiles of each severity class, resized to ``size``.

    ``mode`` selects the layout the consuming model expects:

    * ``"change"``  -> ``[post(3), post - pre(3)]``, the 6-channel stack YOLO12 takes.
    * ``"temporal"``-> ``[pre, post]`` as ``[T=2, 3, H, W]``, what CNN+LSTM takes.
    * ``"single"``  -> ``post`` alone, for a plain 3-channel model.

    Returned as ``(image, mask, label, fraction)`` -- the same 4-tuple the real dataset yields,
    so the training loop genuinely cannot tell a synthetic tile from a real one.
    """
    import cv2

    torch.manual_seed(seed)
    g.eval()
    dev = torch.device(device)
    out: list[tuple[np.ndarray, np.ndarray, int, float]] = []

    def up(planes: np.ndarray, interp=cv2.INTER_LINEAR) -> np.ndarray:
        return np.stack([cv2.resize(c, (size, size), interpolation=interp) for c in planes])

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
                pre = up(sample[0:3]).astype(np.float32)
                post = up(sample[3:6]).astype(np.float32)
                mask = (up(sample[6:7]) > 0.5).astype(np.float32)

                if mode == "temporal":
                    img = np.stack([pre, post], axis=0)
                elif mode == "change":
                    img = np.concatenate([post, post - pre], axis=0)
                else:
                    img = post
                out.append((img, mask, cls, float(mask.mean())))
    return out


#: Synthetic tiles are capped at this multiple of the real training set.
#:
#: Balancing every class all the way up to the majority count sounds principled, but on a
#: corpus whose natural prior is 87% Healthy it more than doubles the training set: measured
#: at T3, epoch time went from 56s to 242s, which under a fixed wall-clock budget bought far
#: fewer epochs than the extra data was worth. It also lets GAN artefacts outnumber real
#: pixels for the rarest class, which is the opposite of what augmentation is for.
MAX_SYNTHETIC_FRACTION = 0.5


def deficit_counts(
    labels: np.ndarray,
    cap: int | None = None,
    max_fraction: float = MAX_SYNTHETIC_FRACTION,
) -> dict[int, int]:
    """How many synthetic tiles each class needs, subject to a total budget.

    Classes are first given their full deficit to the majority count, then scaled back
    proportionally if the total exceeds ``max_fraction`` of the real training set. Scaling
    proportionally rather than truncating keeps the relative emphasis on the rarest classes.
    """
    counts = {int(c): int((labels == c).sum()) for c in range(NUM_CLASSES)}
    target = max(counts.values()) if counts else 0
    if cap is not None:
        target = min(target, cap)
    deficits = {c: max(0, target - n) for c, n in counts.items()}

    total_real = int(sum(counts.values()))
    budget = int(total_real * max_fraction)
    total_deficit = sum(deficits.values())
    if total_deficit > budget > 0:
        scale = budget / total_deficit
        deficits = {c: int(round(v * scale)) for c, v in deficits.items()}
    return deficits
