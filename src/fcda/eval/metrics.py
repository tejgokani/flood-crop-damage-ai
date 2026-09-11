"""Metrics for both heads.

Segmentation is scored with IoU and Dice on the flood class; severity is scored with
accuracy, macro-F1 and Cohen's kappa.

Macro-F1 is the headline number rather than accuracy, deliberately. The Healthy class is
roughly 60% of the corpus, so a model that predicts Healthy for everything scores well on
accuracy and is useless for disaster response -- the entire point is to find the Severe
tiles. Macro-F1 weights every class equally and exposes that failure immediately.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
from sklearn.metrics import cohen_kappa_score, confusion_matrix, f1_score

from .. import NUM_CLASSES, SEVERITY_CLASSES


@dataclass
class Metrics:
    accuracy: float = 0.0
    macro_f1: float = 0.0
    weighted_f1: float = 0.0
    kappa: float = 0.0
    iou: float = 0.0
    dice: float = 0.0
    #: Ordinal: fraction of predictions within one class of the truth.
    within_one_accuracy: float = 0.0
    #: Ordinal: mean absolute error measured in class steps.
    ordinal_mae: float = 0.0
    #: Binary: any damage (Mild/Moderate/Severe) versus Healthy.
    binary_accuracy: float = 0.0
    binary_f1: float = 0.0
    per_class_f1: dict[str, float] = field(default_factory=dict)
    support: dict[str, int] = field(default_factory=dict)
    confusion: list[list[int]] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    def summary(self) -> str:
        return (
            f"acc={self.accuracy:.3f} macroF1={self.macro_f1:.3f} kappa={self.kappa:.3f} "
            f"IoU={self.iou:.3f} Dice={self.dice:.3f} "
            f"within1={self.within_one_accuracy:.3f} binF1={self.binary_f1:.3f}"
        )


def segmentation_scores(
    pred_logits: np.ndarray, target: np.ndarray, threshold: float = 0.5
) -> tuple[float, float]:
    """IoU and Dice for the flood class, aggregated over the whole split.

    Aggregating globally rather than averaging per-image scores avoids letting the many
    near-empty Healthy tiles dominate: a tile with no flood has an undefined IoU, and
    per-image averaging forces an arbitrary convention for it.
    """
    prob = 1.0 / (1.0 + np.exp(-pred_logits))
    pred = prob >= threshold
    tgt = target > 0.5
    inter = float(np.logical_and(pred, tgt).sum())
    union = float(np.logical_or(pred, tgt).sum())
    denom = float(pred.sum() + tgt.sum())
    iou = inter / union if union > 0 else 1.0
    dice = 2.0 * inter / denom if denom > 0 else 1.0
    return iou, dice


def ordinal_scores(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Metrics that respect the fact that the four classes are *ordered*.

    Macro-F1 treats Healthy/Mild/Moderate/Severe as four unrelated labels, so calling a Severe
    tile Moderate is scored exactly as badly as calling it Healthy. Operationally those are very
    different mistakes: the first still sends an assessor, the second does not. On the measured
    confusion matrices every Severe miss lands on Moderate -- the adjacent class -- which macro-F1
    alone cannot show.

    Reported *alongside* the mandated four-class metrics, never instead of them.
    """
    if len(y_true) == 0:
        return {"within_one_accuracy": 0.0, "ordinal_mae": 0.0}
    diff = np.abs(y_true.astype(int) - y_pred.astype(int))
    return {
        "within_one_accuracy": float((diff <= 1).mean()),
        "ordinal_mae": float(diff.mean()),
    }


def binary_damage_scores(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Any damage (Mild/Moderate/Severe) versus Healthy.

    The corpus holds only 20 Severe tiles, so a four-class Severe F1 rests on a handful of
    samples and is dominated by noise. The binary question -- is this tile damaged at all? --
    is the one the data can actually answer, and it is also the first decision a relief
    operation makes. Supplementary, and labelled as such.
    """
    if len(y_true) == 0:
        return {"binary_accuracy": 0.0, "binary_f1": 0.0}
    t = (np.asarray(y_true) > 0).astype(int)
    p = (np.asarray(y_pred) > 0).astype(int)
    return {
        "binary_accuracy": float((t == p).mean()),
        "binary_f1": float(f1_score(t, p, zero_division=0)),
    }


def generalisation_gap(train_scores: list[float], val_scores: list[float]) -> float:
    """Final training macro-F1 minus best validation macro-F1.

    Reported as a first-class metric so "the model does not overfit" is checkable rather than
    asserted. Anything above about 0.10 means the model is memorising.
    """
    if not train_scores or not val_scores:
        return 0.0
    return float(train_scores[-1] - max(val_scores))


def classification_scores(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    labels = list(range(NUM_CLASSES))
    per_class = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    return {
        **ordinal_scores(np.asarray(y_true), np.asarray(y_pred)),
        **binary_damage_scores(y_true, y_pred),
        "accuracy": float((y_true == y_pred).mean()) if len(y_true) else 0.0,
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "weighted_f1": float(
            f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)
        ),
        "kappa": float(cohen_kappa_score(y_true, y_pred, labels=labels)) if len(set(y_true)) > 1 else 0.0,
        "per_class_f1": {SEVERITY_CLASSES[i]: float(per_class[i]) for i in labels},
        "support": {SEVERITY_CLASSES[i]: int((y_true == i).sum()) for i in labels},
        "confusion": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    seg_logits: np.ndarray | None = None,
    seg_target: np.ndarray | None = None,
) -> Metrics:
    scores = classification_scores(np.asarray(y_true), np.asarray(y_pred))
    iou = dice = 0.0
    if seg_logits is not None and seg_target is not None:
        iou, dice = segmentation_scores(seg_logits, seg_target)
    return Metrics(iou=iou, dice=dice, **scores)
