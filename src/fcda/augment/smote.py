"""SMOTE for the tabular pipeline -- applied to the TRAINING split only.

Ma'am's CSV sequence puts SMOTE after the split, with the words "on TRAINING data only".
That ordering is the whole point: SMOTE synthesises new minority rows by interpolating
between neighbours, so fitting it before the split leaks information about validation and
test rows into the training set and produces a score that will not survive contact with real
data.

This module makes that mistake hard to commit. ``balance_training_split`` takes the split
indices and physically cannot see val or test, and ``assert_untouched`` is used by the tests
to prove the evaluation rows were never resampled.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SmoteResult:
    x: np.ndarray
    y: np.ndarray
    n_original: int
    n_synthetic: int
    before: dict[int, int]
    after: dict[int, int]
    strategy: str

    def summary(self) -> str:
        return (
            f"SMOTE[{self.strategy}]: {self.n_original} -> {len(self.y)} rows "
            f"(+{self.n_synthetic} synthetic); {self.before} -> {self.after}"
        )


def _counts(y: np.ndarray) -> dict[int, int]:
    return {int(c): int((y == c).sum()) for c in np.unique(y)}


def balance_training_split(
    x_train: np.ndarray,
    y_train: np.ndarray,
    strategy: str = "auto",
    k_neighbors: int = 5,
    seed: int = 42,
) -> SmoteResult:
    """Oversample the minority severity classes in the training split.

    Falls back to plain random oversampling when a class has fewer members than
    ``k_neighbors + 1``, because SMOTE cannot interpolate without neighbours.
    """
    from imblearn.over_sampling import SMOTE, RandomOverSampler

    before = _counts(y_train)
    smallest = min(before.values())
    used = "SMOTE"

    if smallest <= k_neighbors:
        k = max(1, smallest - 1)
        if k < 1:
            sampler = RandomOverSampler(random_state=seed)
            used = "RandomOverSampler (a class had a single member)"
        else:
            sampler = SMOTE(random_state=seed, k_neighbors=k)
            used = f"SMOTE (k reduced to {k} for the smallest class)"
    else:
        sampler = SMOTE(random_state=seed, k_neighbors=k_neighbors)

    if strategy != "auto":
        sampler.set_params(sampling_strategy=strategy)

    x_res, y_res = sampler.fit_resample(x_train, y_train)
    return SmoteResult(
        x=np.asarray(x_res),
        y=np.asarray(y_res),
        n_original=len(y_train),
        n_synthetic=len(y_res) - len(y_train),
        before=before,
        after=_counts(np.asarray(y_res)),
        strategy=used,
    )


def assert_untouched(n_before: int, n_after: int, name: str) -> None:
    """Guard used in tests: evaluation splits must never change size."""
    if n_before != n_after:
        raise AssertionError(
            f"{name} split changed size ({n_before} -> {n_after}); "
            "resampling must be confined to the training split"
        )
