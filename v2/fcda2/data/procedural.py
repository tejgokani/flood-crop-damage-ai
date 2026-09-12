"""Terrain-driven procedural SAR flood tiles, calibrated to the real corpus.

Why procedural rather than a GAN: v1's conditional DCGAN was measured, and it *hurt* --
-0.101 macro-F1 across three folds, and it reintroduced the overfitting that had just been
removed. Trained on 20 real Severe tiles it had nothing to learn Severe from, so it produced
samples the model memorised.

Procedural generation inverts the problem. We are not trying to learn the distribution; we
already know the physics that matters for this task:

1. **Water is specular.** It reflects radar away from the sensor, so flooded ground is dark.
   Measured on 150 real tiles: flooded VV averages 50.1 against 180.4 for dry ground -- a
   130-level gap in 8-bit terms. ``configs/real_stats.json`` holds the calibration, and the
   generator reproduces those means and spreads rather than inventing plausible-looking ones.
2. **Flood follows terrain.** Water fills low ground, so the flooded region is a level set of a
   height field, not a blob. That is what gives real flood masks their dendritic, river-hugging
   shape -- and it is what a distance-to-a-point generator (v1's approach) cannot produce.
3. **Speckle is multiplicative.** SAR noise comes from coherent interference within a
   resolution cell, so it scales the signal rather than adding to it.

The decisive property is control: **we generate at an exact target flood fraction.** The corpus
has 20 real Severe tiles and that is fixed, but we can synthesise a thousand at 40% inundation.
That is the one thing standing between macro-F1 0.46 and macro-F1 0.70, because macro-F1
averages the classes and Severe is the binding constraint.

Synthetic tiles are training-only. Nothing here is ever scored.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

CONFIG = Path(__file__).resolve().parents[2] / "configs" / "real_stats.json"

#: Fallback calibration if the measured file is missing (values from 150 real ETCI tiles).
FALLBACK_STATS = {
    "flood_vv": [50.1, 58.4], "dry_vv": [180.4, 65.4],
    "flood_vh": [87.9, 62.8], "dry_vh": [211.1, 58.0],
}


@lru_cache(maxsize=1)
def real_stats() -> dict:
    if CONFIG.exists():
        return json.loads(CONFIG.read_text())
    return FALLBACK_STATS


@dataclass(frozen=True)
class Sample:
    """One synthetic tile, in exactly the layout the real loader produces."""

    pre_vv: np.ndarray
    pre_vh: np.ndarray
    post_vv: np.ndarray
    post_vh: np.ndarray
    flood: np.ndarray        # net flood (permanent water excluded), uint8 0/255
    water_body: np.ndarray   # permanent water, uint8 0/255
    target_fraction: float

    def as_planes(self) -> dict[str, np.ndarray]:
        """The dict shape ``load_planes`` returns, so downstream code cannot tell the difference."""
        return {
            "pre_vv": self.pre_vv, "pre_vh": self.pre_vh,
            "pre_flood": self.water_body, "pre_water_body": self.water_body,
            "post_vv": self.post_vv, "post_vh": self.post_vh,
            "post_flood": np.maximum(self.flood, self.water_body),
            "post_water_body": self.water_body,
        }


def _fractal_noise(rng: np.random.Generator, size: int, octaves: int = 5) -> np.ndarray:
    """Sum of upsampled random grids -- cheap 1/f noise, no external dependency."""
    out = np.zeros((size, size), dtype=np.float32)
    amp = 1.0
    for o in range(octaves):
        n = max(2, size // (2 ** (octaves - o)))
        grid = rng.normal(0, 1, (n, n)).astype(np.float32)
        idx = np.linspace(0, n - 1, size)
        yi, xi = np.meshgrid(idx, idx, indexing="ij")
        y0, x0 = np.floor(yi).astype(int), np.floor(xi).astype(int)
        y1 = np.clip(y0 + 1, 0, n - 1)
        x1 = np.clip(x0 + 1, 0, n - 1)
        fy, fx = (yi - y0)[..., None][..., 0], (xi - x0)[..., None][..., 0]
        top = grid[y0, x0] * (1 - fx) + grid[y0, x1] * fx
        bot = grid[y1, x0] * (1 - fx) + grid[y1, x1] * fx
        out += amp * (top * (1 - fy) + bot * fy)
        amp *= 0.5
    out -= out.min()
    return out / (out.max() + 1e-6)


def _parcels(rng: np.random.Generator, size: int) -> np.ndarray:
    """Agricultural field patchwork: rectangles of differing backscatter."""
    canvas = np.zeros((size, size), dtype=np.float32)
    rows, cols = int(rng.integers(3, 8)), int(rng.integers(3, 8))
    ys = np.sort(rng.choice(np.arange(1, size), rows - 1, replace=False))
    xs = np.sort(rng.choice(np.arange(1, size), cols - 1, replace=False))
    ys = np.concatenate([[0], ys, [size]])
    xs = np.concatenate([[0], xs, [size]])
    for i in range(len(ys) - 1):
        for j in range(len(xs) - 1):
            # Centred on 1.0 so the field pattern varies the scene without shifting its mean
            # away from the measured dry backscatter.
            canvas[ys[i]:ys[i + 1], xs[j]:xs[j + 1]] = rng.uniform(0.80, 1.22)
    return canvas


def _river(rng: np.random.Generator, terrain: np.ndarray, size: int) -> np.ndarray:
    """A permanent watercourse carved along the lowest path. Present in pre *and* post."""
    mask = np.zeros((size, size), dtype=np.uint8)
    width = max(1, int(size * rng.uniform(0.008, 0.022)))
    y = int(np.argmin(terrain[:, 0]))
    for x in range(size):
        lo = max(0, y - 2)
        hi = min(size, y + 3)
        y = lo + int(np.argmin(terrain[lo:hi, x]))
        y = int(np.clip(y, width, size - width - 1))
        mask[y - width:y + width + 1, x] = 1
    return mask


def _flood_at_fraction(terrain: np.ndarray, river: np.ndarray, fraction: float) -> np.ndarray:
    """Flood the lowest ground until the *net* flooded area is ``fraction`` of the tile.

    A quantile of the height field is a water level, so the flooded region is a genuine level
    set -- irregular, connected along valleys, and hugging the river. Two details make the area
    land where it was asked to:

    * The quantile is taken over **non-river pixels only**, and rescaled by the river's share of
      the tile. Severity is computed on *net* flood with permanent water subtracted, so taking
      the quantile over the whole tile silently over-counts the river and lands the sample a
      class low -- measured at 16% of commissioned Mild tiles coming back Healthy.
    * The river bed is lowered first, so flood spreads outward from the watercourse rather than
      appearing as a disconnected pool.
    """
    if fraction <= 0:
        return np.zeros_like(river)
    h = terrain.copy()
    h[river > 0] -= 0.25

    land = river == 0
    land_share = float(land.mean())
    if land_share <= 0:
        return np.zeros_like(river)
    # Want `fraction` of the whole tile to be net flood, and net flood only exists on land.
    q = float(np.clip(fraction / land_share, 0.0, 1.0))
    level = np.quantile(h[land], q)
    return ((h <= level) | (river > 0)).astype(np.uint8)


def _to_backscatter(
    rng: np.random.Generator, parcels: np.ndarray, water: np.ndarray, pol: str
) -> np.ndarray:
    """Render an 8-bit SAR amplitude image with the measured wet/dry statistics."""
    s = real_stats()
    dry_mu, dry_sd = s[f"dry_{pol}"]
    wet_mu, wet_sd = s[f"flood_{pol}"]

    # Parcels modulate *land* only. Open water backscatter is set by the surface, not by the
    # crop underneath it, and multiplying water by the field pattern both misstates the physics
    # and drags the dry/wet contrast away from the measured 130 levels.
    is_water = water > 0
    img = np.where(is_water, wet_mu, dry_mu * parcels).astype(np.float32)
    sd = np.where(is_water, wet_sd, dry_sd).astype(np.float32)

    # Multiplicative gamma speckle: the correct noise model for SAR amplitude. Water is far
    # rougher in relative terms (near the noise floor), so it gets the heavier speckle.
    shape = np.where(is_water, 1.6, 6.0).astype(np.float32)
    speckle = rng.gamma(shape, 1.0 / shape).astype(np.float32)
    noise_scale = np.where(is_water, 0.62, 0.30).astype(np.float32)
    img = img * speckle + rng.normal(0, 1.0, img.shape).astype(np.float32) * sd * noise_scale
    return np.clip(img, 0, 255).astype(np.uint8)


def make_sample(seed: int, size: int = 256, target_fraction: float | None = None) -> Sample:
    """Generate one pre/post tile pair flooded to ``target_fraction`` of its area."""
    rng = np.random.default_rng(seed)
    terrain = _fractal_noise(rng, size)
    parcels = _parcels(rng, size)
    river = _river(rng, terrain, size)

    if target_fraction is None:
        target_fraction = float(rng.uniform(0.0, 0.6))

    flood_all = _flood_at_fraction(terrain, river, target_fraction)
    net_flood = (flood_all > 0) & (river == 0)         # permanent water is not damage

    pre_water = river
    post_water = np.maximum(flood_all, river)

    return Sample(
        pre_vv=_to_backscatter(rng, parcels, pre_water, "vv"),
        pre_vh=_to_backscatter(rng, parcels, pre_water, "vh"),
        post_vv=_to_backscatter(rng, parcels, post_water, "vv"),
        post_vh=_to_backscatter(rng, parcels, post_water, "vh"),
        flood=(net_flood * 255).astype(np.uint8),
        water_body=(river * 255).astype(np.uint8),
        target_fraction=float(target_fraction),
    )


#: Flood-fraction ranges that land in each severity class, given thresholds 0.02/0.10/0.33.
CLASS_FRACTION_RANGES: dict[int, tuple[float, float]] = {
    0: (0.000, 0.015),
    1: (0.025, 0.095),
    2: (0.105, 0.320),
    3: (0.345, 0.780),
}


def make_for_class(seed: int, label: int, size: int = 256) -> Sample:
    """Generate a tile that lands in ``label`` by construction, not by chance."""
    lo, hi = CLASS_FRACTION_RANGES[label]
    frac = float(np.random.default_rng(seed * 7919 + 13).uniform(lo, hi))
    return make_sample(seed, size=size, target_fraction=frac)


def make_balanced(counts: dict[int, int], size: int = 256, seed: int = 0) -> list[tuple[Sample, int]]:
    """Generate ``counts[label]`` tiles of each class. This is the point of the module."""
    out: list[tuple[Sample, int]] = []
    n = 0
    for label, want in sorted(counts.items()):
        for _ in range(want):
            out.append((make_for_class(seed * 1_000_003 + n, label, size), label))
            n += 1
    return out
