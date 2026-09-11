"""Composable preprocessing and augmentation for the image pipeline."""

from __future__ import annotations

import numpy as np

from .clahe import CLAHE


class Compose:
    def __init__(self, steps: list) -> None:
        self.steps = [s for s in steps if s is not None]

    def __call__(self, img: np.ndarray) -> np.ndarray:
        for s in self.steps:
            img = s(img)
        return img

    def __repr__(self) -> str:
        return f"Compose({[repr(s) for s in self.steps]})"


class Normalize:
    """Per-channel standardisation to zero mean, unit variance."""

    def __init__(self, eps: float = 1e-6) -> None:
        self.eps = eps

    def __call__(self, img: np.ndarray) -> np.ndarray:
        m = img.mean(axis=(1, 2), keepdims=True)
        s = img.std(axis=(1, 2), keepdims=True)
        return (img - m) / (s + self.eps)

    def __repr__(self) -> str:
        return "Normalize()"


class RandomFlip:
    """Geometric augmentation. Safe for flood imagery: there is no canonical 'up'."""

    def __init__(self, p: float = 0.5, seed: int | None = None) -> None:
        self.p = p
        self.rng = np.random.default_rng(seed)

    def __call__(self, img: np.ndarray) -> np.ndarray:
        if self.rng.random() < self.p:
            img = img[:, :, ::-1]
        if self.rng.random() < self.p:
            img = img[:, ::-1, :]
        return np.ascontiguousarray(img)

    def __repr__(self) -> str:
        return f"RandomFlip(p={self.p})"


class SpeckleNoise:
    """Multiplicative gamma noise -- the physically correct augmentation for SAR.

    Additive Gaussian noise would be the wrong model here: SAR speckle is multiplicative,
    arising from coherent interference within a resolution cell.
    """

    def __init__(self, strength: float = 0.1, p: float = 0.3, seed: int | None = None) -> None:
        self.strength = strength
        self.p = p
        self.rng = np.random.default_rng(seed)

    def __call__(self, img: np.ndarray) -> np.ndarray:
        if self.rng.random() >= self.p:
            return img
        shape = 1.0 / max(self.strength, 1e-3)
        noise = self.rng.gamma(shape, 1.0 / shape, size=img.shape).astype(np.float32)
        return img * noise

    def __repr__(self) -> str:
        return f"SpeckleNoise(strength={self.strength}, p={self.p})"


def build_transform(
    train: bool,
    use_clahe: bool = True,
    augment_strength: float = 1.0,
    seed: int | None = None,
) -> Compose:
    """Standard preprocessing chain.

    Order matters and follows Ma'am's sequence: CLAHE is a *preprocessing* step applied to
    every split, while the random augmentations are training-only. Normalisation runs last
    so the network always sees standardised input regardless of what came before.
    """
    steps: list = []
    if use_clahe:
        steps.append(CLAHE())
    if train and augment_strength > 0:
        steps.append(RandomFlip(p=0.5 * augment_strength, seed=seed))
        steps.append(SpeckleNoise(strength=0.1 * augment_strength, p=0.3 * augment_strength, seed=seed))
    steps.append(Normalize())
    return Compose(steps)
