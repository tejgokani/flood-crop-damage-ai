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
    per_class_f1: dict[str, float] = field(default_factory=dict)
    support: dict[str, int] = field(default_factory=dict)
    confusion: list[list[int]] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    def summary(self) -> str:
        return (
            f"acc={self.accuracy:.3f} macroF1={self.macro_f1:.3f} kappa={self.kappa:.3f} "
            f"IoU={self.iou:.3f} Dice={self.dice:.3f}"
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


def classification_scores(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    labels = list(range(NUM_CLASSES))
    per_class = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    return {
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
