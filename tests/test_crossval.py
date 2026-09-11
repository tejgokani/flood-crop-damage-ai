"""Cross-validation must score every sample exactly once, out of fold."""

from __future__ import annotations

import numpy as np

from fcda.splits import kfold_indices


def _labels(n=300, seed=0):
    rng = np.random.default_rng(seed)
    return np.concatenate([
        np.zeros(int(n * 0.7)), np.ones(int(n * 0.18)),
        np.full(int(n * 0.09), 2), np.full(max(5, int(n * 0.03)), 3),
    ]).astype(int)[:n] if n else np.array([], dtype=int)


def test_every_sample_is_held_out_exactly_once():
    """This is what makes the out-of-fold evaluation cover the whole dataset."""
    labels = _labels()
    seen = np.zeros(len(labels), dtype=int)
    for _, test_idx in kfold_indices(labels, n_splits=5, seed=1):
        seen[test_idx] += 1
    assert (seen == 1).all(), f"{(seen != 1).sum()} samples not held out exactly once"


def test_train_and_test_folds_never_intersect():
    labels = _labels()
    for train_idx, test_idx in kfold_indices(labels, n_splits=5, seed=2):
        assert not (set(train_idx) & set(test_idx))


def test_rare_class_is_evaluated_far_more_than_under_a_single_split():
    """The reason for cross-validating: a 15% holdout scores ~3 Severe tiles, CV scores all of them."""
    labels = _labels()
    n_rare = int((labels == 3).sum())
    evaluated = 0
    for _, test_idx in kfold_indices(labels, n_splits=5, seed=3):
        evaluated += int((labels[test_idx] == 3).sum())
    assert evaluated == n_rare
    assert evaluated > n_rare * 0.15 * 2
