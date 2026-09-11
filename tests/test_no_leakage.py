"""Machine-checkable version of docs/sequence_compliance.md.

These tests exist because "we applied SMOTE to the training data only" is a claim, and a
claim in a README is worth less than an assertion that fails the build.
"""

from __future__ import annotations

import numpy as np
import pytest

from fcda.augment.smote import assert_untouched, balance_training_split
from fcda.preprocess.leakage import (
    average_hash,
    check_duplicate_leakage,
    check_suspicious_features,
    check_target_leakage,
    hamming,
)
from fcda.splits import grouped_stratified_split, kfold_indices, stratified_split


@pytest.fixture
def imbalanced():
    rng = np.random.default_rng(0)
    y = np.concatenate([np.zeros(200), np.ones(60), np.full(40, 2), np.full(12, 3)]).astype(int)
    x = rng.normal(size=(len(y), 6)) + y[:, None] * 0.4
    return x, y


def test_splits_are_disjoint(imbalanced):
    _, y = imbalanced
    s = stratified_split(y, seed=1)
    assert not (set(s.train) & set(s.val))
    assert not (set(s.train) & set(s.test))
    assert not (set(s.val) & set(s.test))
    assert len(s.train) + len(s.val) + len(s.test) == len(y)


def test_every_class_survives_the_split(imbalanced):
    """The rarest class must appear in all three splits, or its F1 is meaningless."""
    _, y = imbalanced
    s = stratified_split(y, seed=1)
    for idx in (s.train, s.val, s.test):
        assert set(np.unique(y[idx])) == set(np.unique(y))


def test_grouped_split_never_shares_a_group(imbalanced):
    _, y = imbalanced
    groups = np.array([f"d{i % 40}" for i in range(len(y))])
    s = grouped_stratified_split(y, groups, seed=3)
    for a, b in ((s.train, s.val), (s.train, s.test), (s.val, s.test)):
        assert not (set(groups[a]) & set(groups[b]))


def test_smote_touches_only_the_training_split(imbalanced):
    """The central claim of Ma'am's CSV sequence, asserted."""
    x, y = imbalanced
    s = stratified_split(y, seed=2)
    n_val, n_test = len(s.val), len(s.test)

    res = balance_training_split(x[s.train], y[s.train], seed=2)

    # Training grew and is now balanced.
    assert len(res.y) > len(s.train)
    assert len(set(res.after.values())) == 1, f"not balanced: {res.after}"
    # Evaluation splits are byte-for-byte the same size.
    assert_untouched(n_val, len(s.val), "val")
    assert_untouched(n_test, len(s.test), "test")


def test_smote_inside_cv_folds_does_not_grow_the_held_out_fold(imbalanced):
    x, y = imbalanced
    for tr, te in kfold_indices(y, n_splits=5, seed=4):
        before = len(te)
        balance_training_split(x[tr], y[tr], seed=4)
        assert len(te) == before


def test_target_leakage_is_detected():
    import pandas as pd

    rng = np.random.default_rng(5)
    label = rng.integers(0, 4, 300)
    df = pd.DataFrame(
        {
            "label": label,
            "honest_feature": rng.normal(size=300),
            "leaky_copy": label.astype(float) + rng.normal(0, 0.001, 300),
        }
    )
    rep = check_target_leakage(df, "label", known_leaky=["leaky_copy"])
    assert not rep.clean
    assert any("leaky_copy" in str(f.items) for f in rep.critical)


def test_suspicious_features_flags_constant_and_id_columns():
    import pandas as pd

    df = pd.DataFrame(
        {
            "label": np.random.default_rng(6).integers(0, 4, 100),
            "constant": np.ones(100),
            "row_id": np.arange(100),
        }
    )
    rep = check_suspicious_features(df, "label")
    kinds = " ".join(str(f.items) for f in rep.findings)
    assert "constant" in kinds
    assert "row_id" in kinds


def test_average_hash_catches_near_duplicates():
    rng = np.random.default_rng(7)
    img = rng.random((64, 64)).astype(np.float32)
    nudged = img + rng.normal(0, 0.002, img.shape).astype(np.float32)
    different = rng.random((64, 64)).astype(np.float32)

    assert hamming(average_hash(img), average_hash(nudged)) <= 3
    assert hamming(average_hash(img), average_hash(different)) > 3


def test_duplicate_leakage_across_splits_is_critical():
    rng = np.random.default_rng(8)
    shared = rng.random((64, 64)).astype(np.float32)
    h = average_hash(shared)
    rep = check_duplicate_leakage(
        {
            "train": {"a": h, "b": average_hash(rng.random((64, 64)).astype(np.float32))},
            "test": {"c": h},
        }
    )
    assert not rep.clean
    assert rep.critical[0].kind == "duplicate_leakage"
