"""The severity rule is the project's main derived quantity, so it gets its own tests."""

from __future__ import annotations

import numpy as np

from fcda import SEVERITY_CLASSES
from fcda.data.severity import DEFAULT_THRESHOLDS, net_flood_mask, severity_from_masks


def _mask(fraction: float, size: int = 100) -> np.ndarray:
    m = np.zeros((size, size), dtype=np.uint8)
    n = int(round(fraction * size * size))
    m.flat[:n] = 255
    return m


def test_thresholds_map_to_the_four_classes():
    assert severity_from_masks(_mask(0.00)).name == "Healthy"
    assert severity_from_masks(_mask(0.05)).name == "Mild"
    assert severity_from_masks(_mask(0.20)).name == "Moderate"
    assert severity_from_masks(_mask(0.50)).name == "Severe"


def test_severe_boundary_sits_at_the_ndrf_threshold():
    """33% is the Indian relief-eligibility threshold the Severe class is anchored on."""
    assert DEFAULT_THRESHOLDS[-1] == 0.33
    assert severity_from_masks(_mask(0.34)).name == "Severe"
    assert severity_from_masks(_mask(0.32)).name == "Moderate"


def test_permanent_water_is_not_counted_as_damage():
    """A tile containing a river is not a damaged tile."""
    river = _mask(0.30)
    result = severity_from_masks(flood=river, water_body=river)
    assert result.name == "Healthy"
    assert result.net_flood_fraction == 0.0
    assert result.permanent_water_fraction > 0.29


def test_flood_beyond_the_river_still_counts():
    water = _mask(0.10)
    flood = _mask(0.50)  # river plus new inundation
    result = severity_from_masks(flood=flood, water_body=water)
    assert result.name == "Severe"
    assert 0.39 < result.net_flood_fraction < 0.41


def test_net_mask_excludes_only_overlapping_pixels():
    flood = _mask(0.4)
    water = _mask(0.1)
    net = net_flood_mask(flood, water)
    assert net.sum() == (flood > 0).sum() - (water > 0).sum()


def test_class_names_are_ordered_by_increasing_damage():
    assert SEVERITY_CLASSES == ("Healthy", "Mild", "Moderate", "Severe")
