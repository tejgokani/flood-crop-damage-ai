"""Deterministic synthetic crop-flood imagery.

Used for two things: the offline smoke tier (so CI and a laptop with no network can still
run the whole pipeline end to end), and unit tests that need reproducible inputs.

The generator imitates the structure that matters for the task rather than trying to look
photographic: rectangular field parcels on a speckled SAR-like background, with flood water
pooling from a low corner and a permanent river that must *not* be counted as damage.
"""

from __future__ import annotations

import numpy as np

from .severity import severity_from_masks


def _speckle(rng: np.random.Generator, size: int) -> np.ndarray:
    """Multiplicative speckle, the characteristic noise of SAR amplitude imagery."""
    return rng.gamma(shape=4.0, scale=0.25, size=(size, size)).astype(np.float32)


def _field_parcels(rng: np.random.Generator, size: int) -> np.ndarray:
    """A patchwork of rectangular agricultural parcels with differing backscatter."""
    canvas = np.zeros((size, size), dtype=np.float32)
    n_rows, n_cols = rng.integers(3, 6), rng.integers(3, 6)
    ys = np.linspace(0, size, n_rows + 1).astype(int)
    xs = np.linspace(0, size, n_cols + 1).astype(int)
    for i in range(n_rows):
        for j in range(n_cols):
            canvas[ys[i] : ys[i + 1], xs[j] : xs[j + 1]] = rng.uniform(0.35, 0.85)
    return canvas


def _permanent_river(rng: np.random.Generator, size: int) -> np.ndarray:
    """A meandering permanent watercourse. Present in both pre and post images."""
    mask = np.zeros((size, size), dtype=np.uint8)
    y = rng.integers(size // 4, 3 * size // 4)
    width = max(2, size // 40)
    for x in range(size):
        y = int(np.clip(y + rng.integers(-1, 2), width, size - width - 1))
        mask[y - width : y + width, x] = 1
    return mask


def _flood_region(rng: np.random.Generator, size: int, target_fraction: float) -> np.ndarray:
    """Water spreading inward from a randomly chosen edge, covering ~target_fraction."""
    mask = np.zeros((size, size), dtype=np.uint8)
    if target_fraction <= 0:
        return mask
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    cy, cx = rng.uniform(0, size), rng.uniform(0, size)
    dist = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    dist += rng.normal(0, size * 0.04, dist.shape)  # irregular shoreline
    radius = np.quantile(dist, np.clip(target_fraction, 0.0, 1.0))
    mask[dist <= radius] = 1
    return mask


def make_sample(
    seed: int, size: int = 128, target_fraction: float | None = None
) -> dict[str, np.ndarray]:
    """Generate one synthetic pre/post flood tile with all four planes at both dates.

    The returned dict mirrors the real ETCI sample layout exactly, so downstream code cannot
    tell a synthetic tier from a real one.
    """
    rng = np.random.default_rng(seed)
    if target_fraction is None:
        # Deliberately imbalanced, like the real corpus: most tiles are barely flooded.
        target_fraction = float(rng.choice([0.0, 0.01, 0.05, 0.2, 0.45], p=[0.35, 0.2, 0.2, 0.15, 0.1]))

    parcels = _field_parcels(rng, size)
    river = _permanent_river(rng, size)
    flood = _flood_region(rng, size, target_fraction)
    flood = np.maximum(flood, river)  # water is water in the flood plane

    def to_sar(base: np.ndarray, water: np.ndarray, pol_gain: float) -> np.ndarray:
        """Water is specular: it returns almost nothing to the sensor, so it reads dark."""
        amp = base.copy()
        amp[water > 0] *= 0.12
        amp = amp * pol_gain * _speckle(rng, size)
        return np.clip(amp * 255.0, 0, 255).astype(np.uint8)

    pre_water = river
    return {
        "pre_vv": to_sar(parcels, pre_water, 1.0),
        "pre_vh": to_sar(parcels, pre_water, 0.6),
        "pre_flood": (pre_water * 255).astype(np.uint8),
        "pre_water_body": (river * 255).astype(np.uint8),
        "post_vv": to_sar(parcels, flood, 1.0),
        "post_vh": to_sar(parcels, flood, 0.6),
        "post_flood": (flood * 255).astype(np.uint8),
        "post_water_body": (river * 255).astype(np.uint8),
    }


def make_dataset(n: int, size: int = 128, seed: int = 0) -> list[dict]:
    """A reproducible synthetic tier, labelled with the same rule as the real data."""
    out = []
    for i in range(n):
        s = make_sample(seed * 100_000 + i, size=size)
        sev = severity_from_masks(s["post_flood"], s["post_water_body"])
        out.append({"tile_id": f"synth-{i:05d}", "planes": s, "severity": sev})
    return out
