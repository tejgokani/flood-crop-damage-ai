"""Calibration must sharpen confidence without touching the decision or the held-out fold."""

from __future__ import annotations

import numpy as np

from fcda.eval.calibration import apply_temperature, expected_calibration_error, fit_temperature


def _underconfident(n=400, seed=0):
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, 4, n)
    logits = rng.normal(0, 0.3, (n, 4))
    logits[np.arange(n), labels] += 0.9
    return logits, labels


def test_temperature_scaling_never_changes_the_prediction():
    """Dividing logits by a scalar is argmax-invariant, so accuracy cannot move."""
    logits, labels = _underconfident()
    result = fit_temperature(logits, labels)
    before = logits.argmax(1)
    after = apply_temperature(logits, result.temperature).argmax(1)
    assert (before == after).all()


def test_calibration_improves_expected_calibration_error():
    logits, labels = _underconfident()
    result = fit_temperature(logits, labels)
    assert result.ece_after < result.ece_before


def test_ece_is_zero_for_a_perfectly_calibrated_predictor():
    probs = np.full((100, 4), 0.25)
    labels = np.tile([0, 1, 2, 3], 25)
    assert expected_calibration_error(probs, labels) < 0.02


def test_confidence_rises_for_an_underconfident_model():
    """This is the honest version of 'increase the confidence percentage'."""
    logits, labels = _underconfident()
    result = fit_temperature(logits, labels)
    assert result.mean_confidence_after > result.mean_confidence_before
