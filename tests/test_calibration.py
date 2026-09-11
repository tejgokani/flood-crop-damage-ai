"""Calibration must sharpen confidence without touching the decision or the held-out fold."""

from __future__ import annotations

import numpy as np

from fcda.eval.calibration import apply_temperature, expected_calibration_error, fit_temperature


def _underconfident(n=600, seed=0):
    """A model that is right about 70% of the time but reports far less confidence than that."""
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, 4, n)
    logits = rng.normal(0, 0.5, (n, 4))
    # Correct class favoured most of the time, but by a small margin -> under-confident.
    correct = rng.random(n) < 0.7
    logits[np.arange(n), labels] += np.where(correct, 1.2, -0.4)
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
    assert result.accepted, result.note
    assert result.ece_after < result.ece_before


def test_ece_is_zero_for_a_perfectly_calibrated_predictor():
    probs = np.full((100, 4), 0.25)
    labels = np.tile([0, 1, 2, 3], 25)
    assert expected_calibration_error(probs, labels) < 0.02


def test_confidence_rises_for_an_underconfident_model():
    """This is the honest version of 'increase the confidence percentage'."""
    logits, labels = _underconfident()
    result = fit_temperature(logits, labels)
    assert result.accepted, result.note
    assert result.mean_confidence_after > result.mean_confidence_before


def test_a_marginal_fit_is_rejected_rather_than_applied():
    """Guards the real failure seen in cross validation.

    One fold drove the temperature to its bound and reported 96.3% mean confidence off an ECE
    move of 0.283 -> 0.260 -- an 8% change, well inside the noise of ~100 samples. A fit that
    barely moves calibration must be discarded, not applied, or the pipeline manufactures
    confidence it has not earned.

    Here the model is already close to calibrated, so no temperature can help much.
    """
    rng = np.random.default_rng(11)
    n = 400
    labels = rng.integers(0, 4, n)
    logits = np.zeros((n, 4))
    # Margin tuned so accuracy roughly matches reported confidence already.
    correct = rng.random(n) < 0.55
    logits[np.arange(n), labels] += np.where(correct, 1.05, -0.35)
    logits += rng.normal(0, 0.15, logits.shape)

    result = fit_temperature(logits, labels)
    improvement = 1.0 - (result.ece_after / max(result.ece_before, 1e-9))
    if result.accepted:
        # If it was accepted it must have earned it.
        assert improvement >= 0.20 - 1e-9, f"accepted a {improvement:.0%} improvement"
    else:
        assert result.temperature == 1.0
        assert result.mean_confidence_after == result.mean_confidence_before


def test_rejection_leaves_confidence_untouched():
    """A rejected fit must not move the reported confidence at all."""
    rng = np.random.default_rng(5)
    labels = rng.integers(0, 4, 20)  # below min_samples -> always rejected
    logits = rng.normal(0, 1.0, (20, 4))
    result = fit_temperature(logits, labels)
    assert not result.accepted
    assert result.temperature == 1.0
    assert result.mean_confidence_after == result.mean_confidence_before
    assert result.ece_after == result.ece_before
