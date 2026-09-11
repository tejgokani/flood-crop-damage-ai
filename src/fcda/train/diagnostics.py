"""Over-fitting / under-fitting detection -- an explicit step in Ma'am's sequence.

The sequence does not say "tune the model"; it says *detect* the condition and then *apply a
correction and retrain*. So this has to be a measurement, not a judgement call.

Two signals are used together:

* **Generalisation gap** -- train metric minus validation metric. A large positive gap is
  the classic over-fitting signature.
* **Learning-curve slope** -- the trend of validation performance over the last few epochs.
  A model can have a small gap and still be under-fitting, if both curves are low and still
  climbing when training stops. The slope is what separates "converged and poor" (under-fit,
  needs capacity) from "still improving" (needs more epochs).

Paper 3 named "enhancing regularization" as future work; making the decision measurable and
automatic is our answer to that.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np


class FitStatus(StrEnum):
    OVERFIT = "OVERFIT"
    UNDERFIT = "UNDERFIT"
    OK = "OK"


@dataclass
class Diagnosis:
    status: FitStatus
    gap: float
    val_slope: float
    best_val: float
    reason: str

    def to_dict(self) -> dict:
        return {
            "status": self.status.value,
            "gap": round(self.gap, 4),
            "val_slope": round(self.val_slope, 5),
            "best_val": round(self.best_val, 4),
            "reason": self.reason,
        }


#: A train-minus-val gap above this is treated as over-fitting.
GAP_THRESHOLD = 0.15
#: Validation macro-F1 below this, with a small gap, is treated as under-fitting.
UNDERFIT_CEILING = 0.45
#: Slope (per epoch) above which the model is judged still to be learning.
IMPROVING_SLOPE = 0.004


def _slope(values: list[float], window: int = 5) -> float:
    """Least-squares slope of the last ``window`` points."""
    tail = values[-window:]
    if len(tail) < 2:
        return 0.0
    x = np.arange(len(tail), dtype=float)
    return float(np.polyfit(x, np.asarray(tail, dtype=float), 1)[0])


def diagnose(
    train_scores: list[float],
    val_scores: list[float],
    gap_threshold: float = GAP_THRESHOLD,
    underfit_ceiling: float = UNDERFIT_CEILING,
) -> Diagnosis:
    """Classify the run from its learning curves (macro-F1 per epoch)."""
    if not train_scores or not val_scores:
        return Diagnosis(FitStatus.OK, 0.0, 0.0, 0.0, "no epochs recorded")

    best_val = max(val_scores)
    # Compare the final training score against the best validation score, not the final one:
    # a noisy last epoch should not change the diagnosis.
    gap = train_scores[-1] - best_val
    slope = _slope(val_scores)

    if gap > gap_threshold and best_val < train_scores[-1]:
        return Diagnosis(
            FitStatus.OVERFIT,
            gap,
            slope,
            best_val,
            f"train macro-F1 exceeds validation by {gap:.3f} (threshold {gap_threshold})",
        )

    if best_val < underfit_ceiling and gap <= gap_threshold:
        if slope > IMPROVING_SLOPE:
            return Diagnosis(
                FitStatus.UNDERFIT,
                gap,
                slope,
                best_val,
                f"validation still improving at {slope:+.4f}/epoch and only reached {best_val:.3f}",
            )
        return Diagnosis(
            FitStatus.UNDERFIT,
            gap,
            slope,
            best_val,
            f"validation plateaued at {best_val:.3f}, below the {underfit_ceiling} ceiling",
        )

    return Diagnosis(
        FitStatus.OK,
        gap,
        slope,
        best_val,
        f"gap {gap:.3f} within tolerance and validation reached {best_val:.3f}",
    )
