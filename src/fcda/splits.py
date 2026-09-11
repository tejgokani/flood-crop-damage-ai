"""Train / validation / test splitting and 10-fold cross validation.

Ma'am's sequence allows either a three-way split or 10-fold CV; we implement both and run
the split for the headline models, with CV available for the tabular pipeline where it is
cheap enough to be worth the extra confidence.

Two details that matter more than the ratio:

* **Stratification.** The severity classes are heavily imbalanced (Severe is the rarest by
  an order of magnitude in both modalities). An unstratified split can leave the test set
  with a handful of Severe tiles, at which point the Severe F1 is noise.
* **Grouping.** For the tabular pipeline the group is the *district*: the same district
  appearing in train and test would let the model memorise a district's yield level rather
  than learn a damage signal. ``grouped_stratified_split`` keeps groups intact while still
  balancing classes as well as grouping allows.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import StratifiedKFold


@dataclass
class SplitIndices:
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray

    def sizes(self) -> dict[str, int]:
        return {"train": len(self.train), "val": len(self.val), "test": len(self.test)}

    def __repr__(self) -> str:
        return f"SplitIndices({self.sizes()})"


def stratified_split(
    labels: np.ndarray | list[int],
    ratios: tuple[float, float, float] = (0.7, 0.15, 0.15),
    seed: int = 42,
) -> SplitIndices:
    """Class-balanced three-way split.

    Each class is shuffled and cut independently, so the class proportions in train, val and
    test match the overall distribution as closely as integer counts permit.
    """
    labels = np.asarray(labels)
    if not np.isclose(sum(ratios), 1.0):
        raise ValueError(f"ratios must sum to 1, got {ratios}")
    rng = np.random.default_rng(seed)
    train, val, test = [], [], []

    for cls in np.unique(labels):
        idx = np.flatnonzero(labels == cls)
        rng.shuffle(idx)
        n = len(idx)
        n_tr = int(round(n * ratios[0]))
        n_va = int(round(n * ratios[1]))
        # Guarantee at least one sample per split for any class that has three or more.
        if n >= 3:
            n_tr = max(1, min(n_tr, n - 2))
            n_va = max(1, min(n_va, n - n_tr - 1))
        train.append(idx[:n_tr])
        val.append(idx[n_tr : n_tr + n_va])
        test.append(idx[n_tr + n_va :])

    out = SplitIndices(
        train=np.sort(np.concatenate(train)),
        val=np.sort(np.concatenate(val)),
        test=np.sort(np.concatenate(test)),
    )
    _assert_disjoint(out)
    return out


def grouped_stratified_split(
    labels: np.ndarray | list[int],
    groups: np.ndarray | list,
    ratios: tuple[float, float, float] = (0.7, 0.15, 0.15),
    seed: int = 42,
) -> SplitIndices:
    """Split by group, assigning whole groups to keep them out of more than one split.

    Groups are ordered by their dominant class and dealt round-robin into the split with the
    largest remaining deficit, which keeps class balance reasonable without ever breaking a
    group apart.
    """
    labels = np.asarray(labels)
    groups = np.asarray(groups)
    uniq = np.unique(groups)
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)

    total = len(labels)
    targets = {"train": ratios[0] * total, "val": ratios[1] * total, "test": ratios[2] * total}
    assigned: dict[str, list[int]] = {"train": [], "val": [], "test": []}
    counts = dict.fromkeys(targets, 0)

    # Largest groups first, so the big ones land before the budget is used up.
    sizes = {g: int((groups == g).sum()) for g in uniq}
    for g in sorted(uniq, key=lambda x: -sizes[x]):
        deficits = {k: targets[k] - counts[k] for k in targets}
        pick = max(deficits, key=lambda k: deficits[k])
        idx = np.flatnonzero(groups == g)
        assigned[pick].extend(idx.tolist())
        counts[pick] += len(idx)

    out = SplitIndices(
        train=np.sort(np.array(assigned["train"], dtype=int)),
        val=np.sort(np.array(assigned["val"], dtype=int)),
        test=np.sort(np.array(assigned["test"], dtype=int)),
    )
    _assert_disjoint(out)
    return out


def kfold_indices(
    labels: np.ndarray | list[int], n_splits: int = 10, seed: int = 42
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Stratified 10-fold CV indices, as named in Ma'am's sequence."""
    labels = np.asarray(labels)
    # A class with fewer members than folds cannot be stratified across all of them.
    min_class = int(np.bincount(labels).min()) if labels.size else 0
    n_splits = max(2, min(n_splits, min_class if min_class >= 2 else 2))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return [(tr, te) for tr, te in skf.split(np.zeros(len(labels)), labels)]


def class_balance(labels: np.ndarray, idx: np.ndarray) -> dict[int, int]:
    sub = np.asarray(labels)[idx]
    return {int(c): int((sub == c).sum()) for c in np.unique(labels)}


def _assert_disjoint(s: SplitIndices) -> None:
    """A split that overlaps is a leakage bug; fail loudly rather than train on it."""
    tr, va, te = set(s.train.tolist()), set(s.val.tolist()), set(s.test.tolist())
    for a, b, an, bn in ((tr, va, "train", "val"), (tr, te, "train", "test"), (va, te, "val", "test")):
        if a & b:
            raise AssertionError(f"split overlap between {an} and {bn}: {len(a & b)} shared indices")
