"""Correction policies -- "Apply Correction Technique and Retrain" in Ma'am's sequence.

``diagnostics.py`` says *what* is wrong; this module says *what to do about it* and produces
a concrete, logged change to the training configuration. The correction is then applied and
the model retrained, with before/after metrics recorded so the decision is auditable rather
than a matter of taste.

The policies are the standard ones, and they are deliberately paired with their diagnosis:

* **Over-fitting** -- the model has enough capacity and is memorising. Raise dropout, raise
  weight decay, strengthen augmentation, and stop earlier.
* **Under-fitting** -- the model is not learning the signal. Lower regularisation, raise the
  learning rate, and give it more epochs.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .diagnostics import Diagnosis, FitStatus


@dataclass
class TrainConfig:
    """Everything the correction policy is allowed to change."""

    lr: float = 3e-4
    #: Regularisation starts high rather than being raised only after overfitting is detected.
    #: Four of the five benchmarked models had a train-validation gap above 0.19 under the old
    #: defaults (dropout 0.1, weight decay 1e-4, no smoothing), and YOLO12 peaked at epoch 1.
    weight_decay: float = 1e-3
    dropout: float = 0.3
    augment_strength: float = 1.0
    epochs: int = 14
    patience: int = 4
    label_smoothing: float = 0.05

    def copy(self) -> TrainConfig:
        return TrainConfig(**self.__dict__)

    def to_dict(self) -> dict:
        return dict(self.__dict__)


@dataclass
class Correction:
    applied: bool
    technique: str
    changes: dict = field(default_factory=dict)
    config: TrainConfig = field(default_factory=TrainConfig)

    def describe(self) -> str:
        if not self.applied:
            return f"No correction: {self.technique}"
        bits = ", ".join(f"{k}: {v[0]} -> {v[1]}" for k, v in self.changes.items())
        return f"{self.technique} ({bits})"

    def to_dict(self) -> dict:
        return {
            "applied": self.applied,
            "technique": self.technique,
            "changes": {k: list(v) for k, v in self.changes.items()},
            "config": self.config.to_dict(),
        }


def _diff(before: TrainConfig, after: TrainConfig) -> dict:
    out = {}
    for k, v in before.to_dict().items():
        nv = getattr(after, k)
        if nv != v:
            out[k] = (round(v, 6) if isinstance(v, float) else v,
                      round(nv, 6) if isinstance(nv, float) else nv)
    return out


def plan_correction(diagnosis: Diagnosis, config: TrainConfig) -> Correction:
    """Turn a diagnosis into a concrete configuration change."""
    new = config.copy()

    if diagnosis.status is FitStatus.OVERFIT:
        # Scale the response to the size of the gap: a marginal gap gets a gentle nudge.
        severity = min(max((diagnosis.gap - 0.15) / 0.25, 0.0), 1.0)
        new.dropout = min(0.5, config.dropout + 0.15 + 0.15 * severity)
        new.weight_decay = min(1e-2, config.weight_decay * (5.0 + 5.0 * severity))
        new.augment_strength = min(2.0, config.augment_strength + 0.5 + 0.5 * severity)
        new.label_smoothing = max(config.label_smoothing, 0.05 + 0.05 * severity)
        new.patience = max(2, config.patience - 2)
        return Correction(
            True,
            "Regularisation increase (dropout, weight decay, augmentation, label smoothing) "
            "with earlier stopping",
            _diff(config, new),
            new,
        )

    if diagnosis.status is FitStatus.UNDERFIT:
        if diagnosis.val_slope > 0:
            # Still climbing: it needs time and a slightly bolder step, not more capacity.
            new.epochs = int(config.epochs * 1.5)
            new.lr = min(3e-3, config.lr * 2.0)
            new.patience = config.patience + 3
            technique = "Extended schedule with a higher learning rate (still improving)"
        else:
            # Plateaued low: regularisation is holding it back.
            new.dropout = max(0.0, config.dropout - 0.05)
            new.weight_decay = max(1e-6, config.weight_decay * 0.2)
            new.lr = min(3e-3, config.lr * 1.5)
            new.epochs = int(config.epochs * 1.3)
            technique = "Reduced regularisation with a higher learning rate (plateaued)"
        return Correction(True, technique, _diff(config, new), new)

    return Correction(False, "fit is within tolerance; no retraining needed", {}, config)
