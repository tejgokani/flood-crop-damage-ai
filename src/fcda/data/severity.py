"""Four-class crop damage severity derived from flood extent.

Problem Statement 6 asks for an output of ``Healthy -> Mild -> Moderate -> Severe``. The
ETCI-2021 corpus ships flood masks and permanent-water masks, but no agronomic damage
labels, so severity here is a **derived** quantity. The derivation is documented rather than
hidden, because it is the main scope boundary of this project.

Two decisions worth defending:

1. **Permanent water is subtracted before scoring.** A tile containing a river is not a
   damaged tile. Paper 4 in our literature review measures flood IoU at roughly half the IoU
   of permanent water precisely because the two are confused; we use the
   ``water_body_label`` plane to avoid inheriting that error.

2. **The Severe boundary sits at 33% inundation.** This is not an arbitrary round number:
   Indian disaster-relief practice (NDRF/SDRF input-subsidy norms) treats a crop loss of
   33% or more as the threshold at which a holding qualifies for compensation. Anchoring the
   top class there means a "Severe" prediction lines up with the decision a revenue officer
   actually has to make. All thresholds are configurable in ``configs/image_pipeline.yaml``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .. import SEVERITY_CLASSES

#: Upper bounds (exclusive) on the *net* inundated fraction for each class.
#: The final class has no upper bound.
DEFAULT_THRESHOLDS: tuple[float, float, float] = (0.02, 0.10, 0.33)

THRESHOLD_RATIONALE = {
    "healthy_max": "Below 2% net inundation is treated as speckle/edge effect, not damage.",
    "mild_max": "2-10% is localised waterlogging at field margins.",
    "moderate_max": (
        "10-33% is partial inundation below the Indian NDRF/SDRF 33% crop-loss "
        "compensation threshold."
    ),
    "severe": (
        "33% or more net inundation aligns with the NDRF/SDRF threshold at which crop loss "
        "qualifies for input-subsidy relief."
    ),
}


@dataclass(frozen=True)
class SeverityResult:
    """Per-tile severity with the evidence that produced it."""

    label: int
    name: str
    flood_fraction: float
    net_flood_fraction: float
    permanent_water_fraction: float


def net_flood_mask(flood: np.ndarray, water_body: np.ndarray | None) -> np.ndarray:
    """Flood pixels that are not permanent water.

    Both inputs may be 0/255 PNG planes or 0/1 masks; they are binarised on > 0.
    """
    f = flood > 0
    if water_body is None:
        return f
    return f & ~(water_body > 0)


def severity_from_masks(
    flood: np.ndarray,
    water_body: np.ndarray | None = None,
    thresholds: tuple[float, float, float] = DEFAULT_THRESHOLDS,
) -> SeverityResult:
    """Classify one tile into Healthy / Mild / Moderate / Severe."""
    if flood.size == 0:
        raise ValueError("empty flood mask")
    total = float(flood.size)
    net = net_flood_mask(flood, water_body)
    net_frac = float(net.sum()) / total
    raw_frac = float((flood > 0).sum()) / total
    perm_frac = float((water_body > 0).sum()) / total if water_body is not None else 0.0

    label = int(np.searchsorted(np.asarray(thresholds, dtype=float), net_frac, side="right"))
    label = min(label, len(SEVERITY_CLASSES) - 1)
    return SeverityResult(
        label=label,
        name=SEVERITY_CLASSES[label],
        flood_fraction=raw_frac,
        net_flood_fraction=net_frac,
        permanent_water_fraction=perm_frac,
    )


def severity_from_probability_map(
    prob: np.ndarray,
    water_body: np.ndarray | None = None,
    decision_threshold: float = 0.5,
    thresholds: tuple[float, float, float] = DEFAULT_THRESHOLDS,
) -> SeverityResult:
    """Same rule applied to a model's predicted flood probability map (used at inference)."""
    return severity_from_masks((prob >= decision_threshold).astype(np.uint8) * 255, water_body, thresholds)


def class_distribution(labels: list[int] | np.ndarray) -> dict[str, int]:
    """Count per class -- used to report the imbalance that motivates our augmentation."""
    arr = np.asarray(labels)
    return {name: int((arr == i).sum()) for i, name in enumerate(SEVERITY_CLASSES)}
