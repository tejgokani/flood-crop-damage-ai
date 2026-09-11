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


class Dihedral:
    """The full 8-element dihedral group: 4 rotations x optional transpose.

    Satellite imagery has no canonical 'up', so every one of the eight orientations is a
    physically valid view of the same scene. That makes this the cheapest legitimate way to
    multiply a 700-tile training set eightfold -- and with models that were peaking at epoch 1,
    more effective data is worth more than more gradient steps.

    Geometric, so it must be applied identically to the pre frame, the post frame and the mask.
    """

    paired = True

    def __init__(self, p: float = 1.0, seed: int | None = None) -> None:
        self.p = p
        self.rng = np.random.default_rng(seed)
        self._k = 0
        self._t = False

    def resample(self) -> None:
        """Draw a new orientation, then hold it for every array in the sample."""
        if self.rng.random() < self.p:
            self._k = int(self.rng.integers(0, 4))
            self._t = bool(self.rng.random() < 0.5)
        else:
            self._k, self._t = 0, False

    def __call__(self, img: np.ndarray) -> np.ndarray:
        out = np.rot90(img, k=self._k, axes=(1, 2))
        if self._t:
            out = np.swapaxes(out, 1, 2)
        return np.ascontiguousarray(out)

    def __repr__(self) -> str:
        return f"Dihedral(p={self.p})"


class RandomResizedCrop:
    """Zoom into a random sub-region, then resize back. Geometric, so paired."""

    paired = True

    def __init__(self, p: float = 0.5, min_scale: float = 0.7, seed: int | None = None) -> None:
        self.p = p
        self.min_scale = min_scale
        self.rng = np.random.default_rng(seed)
        self._box: tuple[int, int, int] | None = None

    def resample(self) -> None:
        self._box = None
        if self.rng.random() < self.p:
            scale = float(self.rng.uniform(self.min_scale, 1.0))
            self._box = (scale, float(self.rng.random()), float(self.rng.random()))

    def __call__(self, img: np.ndarray) -> np.ndarray:
        if self._box is None:
            return img
        import cv2

        scale, fy, fx = self._box
        _, h, w = img.shape
        ch, cw = max(8, int(h * scale)), max(8, int(w * scale))
        y0 = int((h - ch) * fy)
        x0 = int((w - cw) * fx)
        crop = img[:, y0 : y0 + ch, x0 : x0 + cw]
        out = np.stack([cv2.resize(c, (w, h), interpolation=cv2.INTER_LINEAR) for c in crop])
        return np.ascontiguousarray(out)

    def __repr__(self) -> str:
        return f"RandomResizedCrop(p={self.p}, min_scale={self.min_scale})"


class CoarseDropout:
    """Blank a few rectangles. Forces the model to use context rather than one cue.

    Photometric from the mask's point of view -- the target is deliberately left intact, so the
    model must infer the occluded region rather than being told it is empty.
    """

    def __init__(self, p: float = 0.3, n_holes: int = 4, size_frac: float = 0.12,
                 seed: int | None = None) -> None:
        self.p = p
        self.n_holes = n_holes
        self.size_frac = size_frac
        self.rng = np.random.default_rng(seed)

    def __call__(self, img: np.ndarray) -> np.ndarray:
        if self.rng.random() >= self.p:
            return img
        out = img.copy()
        _, h, w = out.shape
        for _ in range(int(self.rng.integers(1, self.n_holes + 1))):
            ph, pw = int(h * self.size_frac), int(w * self.size_frac)
            y = int(self.rng.integers(0, max(1, h - ph)))
            x = int(self.rng.integers(0, max(1, w - pw)))
            out[:, y : y + ph, x : x + pw] = float(out.mean())
        return out

    def __repr__(self) -> str:
        return f"CoarseDropout(p={self.p}, n_holes={self.n_holes})"


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


def apply_geometric_pair(transform, post, pre, target):
    """Run ``transform`` over a sample, keeping geometric steps synchronised.

    Steps marked ``paired`` draw their parameters once per sample (``resample()``) and are then
    applied to the post frame, the pre frame and the mask alike. Unpaired steps are photometric
    and touch only the imagery -- applying CLAHE or speckle to a binary mask would corrupt it.
    """
    steps = getattr(transform, "steps", [transform])
    for st in steps:
        if getattr(st, "paired", False):
            st.resample()

    def run(arr, is_mask=False):
        if arr is None:
            return None
        for st in steps:
            paired = getattr(st, "paired", False)
            if is_mask and not paired:
                continue
            arr = st(arr)
        if is_mask:
            # A resized crop interpolates, which leaves a binary mask with fractional values
            # along every edge. Threshold it back so the segmentation target stays a label.
            arr = (arr >= 0.5).astype(np.float32)
        return arr

    return run(post), run(pre), run(target, is_mask=True)


def build_transform(
    train: bool,
    use_clahe: bool = True,
    augment_strength: float = 1.0,
    seed: int | None = None,
) -> Compose:
    """Standard preprocessing chain.

    Order follows Ma'am's sequence: CLAHE is a *preprocessing* step applied to every split,
    while the random augmentations are training-only. Normalisation runs last so the network
    always sees standardised input regardless of what came before.

    The training chain is deliberately aggressive. Four of the five benchmarked models had a
    train-validation gap above 0.19 and one peaked at epoch 1, which is a data-quantity problem
    rather than an architecture problem; the dihedral group alone multiplies the effective
    training set by eight at no cost in label noise.
    """
    steps: list = []
    if use_clahe:
        steps.append(CLAHE())
    if train and augment_strength > 0:
        steps.append(Dihedral(p=min(1.0, 0.8 * augment_strength), seed=seed))
        steps.append(RandomResizedCrop(p=min(1.0, 0.4 * augment_strength), seed=seed))
        steps.append(SpeckleNoise(strength=0.1 * augment_strength,
                                  p=min(1.0, 0.4 * augment_strength), seed=seed))
        steps.append(CoarseDropout(p=min(1.0, 0.3 * augment_strength), seed=seed))
    steps.append(Normalize())
    return Compose(steps)
