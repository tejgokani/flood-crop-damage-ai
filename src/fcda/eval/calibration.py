"""Confidence calibration by temperature scaling.

The demo showed predictions at 35-71% confidence with nothing behind those numbers. A softmax
over an uncalibrated network is not a probability: it is a score that happens to sum to one.
Temperature scaling divides the logits by a single learned scalar T, fitted by minimising
negative log-likelihood on held-out data. It cannot change which class is predicted -- the
argmax is invariant -- so it improves the *honesty* of the confidence without inflating
accuracy. T < 1 sharpens an under-confident model; T > 1 softens an over-confident one.

This is literature-grounded rather than cosmetic: Paper 4 of our review names "calibration
techniques such as temperature scaling" as its first future-work item.

**The fitting set matters more than the method.** T is fitted on an inner validation slice of
the training folds and never on the held-out fold. Fitting it on the data it is then evaluated
on would be the same overfitting we are trying to remove, just moved into a hyperparameter.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class CalibrationResult:
    temperature: float
    ece_before: float
    ece_after: float
    nll_before: float
    nll_after: float
    #: False when the fit was rejected and the temperature reset to 1.0 (no scaling).
    accepted: bool = True
    mean_confidence_before: float = 0.0
    mean_confidence_after: float = 0.0
    n_fit: int = 0
    note: str = ""

    def to_dict(self) -> dict:
        return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in self.__dict__.items()}

    def summary(self) -> str:
        tag = "" if self.accepted else "  [REJECTED -> T=1]"
        return (
            f"T={self.temperature:.3f}  ECE {self.ece_before:.3f} -> {self.ece_after:.3f}  "
            f"mean confidence {self.mean_confidence_before:.1%} -> "
            f"{self.mean_confidence_after:.1%}{tag}"
        )


def expected_calibration_error(
    probs: np.ndarray, labels: np.ndarray, n_bins: int = 10
) -> float:
    """Gap between confidence and accuracy, averaged over confidence bins.

    A perfectly calibrated model that says 70% is right 70% of the time, so ECE is 0.
    """
    if len(labels) == 0:
        return 0.0
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == labels).astype(float)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:], strict=False):
        in_bin = (conf > lo) & (conf <= hi)
        if in_bin.sum() == 0:
            continue
        ece += (in_bin.mean()) * abs(correct[in_bin].mean() - conf[in_bin].mean())
    return float(ece)


def _nll(logits: torch.Tensor, labels: torch.Tensor, temperature: float = 1.0) -> float:
    return float(F.cross_entropy(logits / temperature, labels))


#: Sanity range for the temperature. Wide on purpose -- the real gate is the ECE test below.
TEMPERATURE_BOUNDS = (0.1, 10.0)

#: A fit is accepted only if it cuts expected calibration error by at least this fraction.
#:
#: "Any improvement" is too weak a test on a small inner-validation slice. One fold came back at
#: T=0.05 -- multiplying every logit by twenty -- reporting 96.3% mean confidence off the back of
#: an ECE change of 0.283 -> 0.260, an 8% move that is well inside the noise of ~100 samples.
#: Requiring a 20% relative reduction separates a real correction from a fitting artefact, and
#: keeps the pipeline from manufacturing confidence it has not earned.
MIN_ECE_IMPROVEMENT = 0.20


def fit_temperature(
    logits: np.ndarray,
    labels: np.ndarray,
    max_iter: int = 200,
    bounds: tuple[float, float] = TEMPERATURE_BOUNDS,
    min_samples: int = 50,
) -> CalibrationResult:
    """Fit a single scalar temperature by minimising NLL on the supplied (held-in) data.

    The fit is **accepted only if it actually reduces expected calibration error** on the data
    it was fitted to. Otherwise the temperature is reset to 1.0 and the result is marked
    rejected. Reporting a sharpened confidence that does not improve calibration would be
    inflating a number rather than correcting it.
    """
    lg = torch.tensor(np.asarray(logits), dtype=torch.float32)
    lb = torch.tensor(np.asarray(labels), dtype=torch.long)

    probs_before = torch.softmax(lg, dim=1).numpy()
    ece_before = expected_calibration_error(probs_before, np.asarray(labels))
    nll_before = _nll(lg, lb)

    # Optimise log T so the temperature stays positive without a constraint.
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=max_iter)

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(lg / log_t.exp().clamp(*bounds), lb)
        loss.backward()
        return loss

    try:
        opt.step(closure)
        temperature = float(log_t.detach().exp().clamp(*bounds))
    except Exception:  # noqa: BLE001 - a failed fit falls back to no scaling
        temperature = 1.0

    probs_after = torch.softmax(lg / temperature, dim=1).numpy()
    ece_after = expected_calibration_error(probs_after, np.asarray(labels))

    accepted, note = True, ""
    if len(labels) < min_samples:
        accepted, note = False, f"only {len(labels)} samples to fit on (min {min_samples})"
    elif ece_before <= 1e-6:
        accepted, note = False, "already calibrated; nothing to correct"
    elif ece_after > ece_before * (1.0 - MIN_ECE_IMPROVEMENT):
        accepted, note = (
            False,
            f"calibration error fell only {100 * (1 - ece_after / ece_before):.0f}% "
            f"(need {100 * MIN_ECE_IMPROVEMENT:.0f}%)",
        )

    if not accepted:
        temperature = 1.0
        probs_after = probs_before
        ece_after = ece_before

    return CalibrationResult(
        temperature=temperature,
        accepted=accepted,
        ece_before=ece_before,
        ece_after=ece_after,
        nll_before=nll_before,
        nll_after=_nll(lg, lb, temperature),
        mean_confidence_before=float(probs_before.max(axis=1).mean()),
        mean_confidence_after=float(probs_after.max(axis=1).mean()),
        n_fit=len(labels),
        note=note,
    )


def apply_temperature(logits: np.ndarray, temperature: float) -> np.ndarray:
    """Calibrated probabilities. The argmax is unchanged, only the confidence moves."""
    lg = torch.tensor(np.asarray(logits), dtype=torch.float32)
    return torch.softmax(lg / max(temperature, 1e-3), dim=1).numpy()
