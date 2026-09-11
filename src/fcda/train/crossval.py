"""K-fold cross-validation with out-of-fold predictions.

This replaces the single 70/15/15 split as the headline evaluation, for two reasons.

**Reliability.** A 15% test split of 900 tiles is 135 samples, and of those roughly 3 are
Severe. Any number computed on 3 samples is noise. Cross-validation evaluates *every* tile
exactly once, on a model that never saw it -- so the evaluation set becomes 900 tiles and all
20 Severe tiles in the pool get scored instead of 3. Same data, 6.7x the evidence.

**Honesty.** Per-fold scores give a standard deviation, and a modest number with error bars is
far more defensible than a modest number without them.

`Sequence.docx` explicitly offers 10-fold cross validation as an alternative to the three-way
split, so this follows the prescribed sequence rather than departing from it. We use 5 folds
because 10 would not fit the wall-clock budget for two deep models on a laptop.

Each fold splits its training portion again into an inner train/validation pair. The inner
validation drives early stopping and fits the calibration temperature; the outer fold is
touched exactly once, for scoring. Fitting anything on the outer fold would reintroduce the
overfitting this module exists to remove.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch

from ..eval.calibration import CalibrationResult, apply_temperature, fit_temperature
from ..eval.metrics import Metrics, compute_metrics
from ..splits import kfold_indices
from .correction import TrainConfig
from .loop import DualLoss, evaluate, train_model


@dataclass
class FoldResult:
    fold: int
    n_train: int
    n_inner_val: int
    n_test: int
    best_val_f1: float
    gap: float
    epochs_run: int
    seconds: float
    metrics: dict = field(default_factory=dict)
    calibration: dict = field(default_factory=dict)
    ok: bool = True
    error: str = ""


@dataclass
class CrossValResult:
    model: str
    n_folds: int
    n_samples: int
    folds: list[FoldResult] = field(default_factory=list)
    oof_metrics: dict = field(default_factory=dict)
    oof_metrics_calibrated: dict = field(default_factory=dict)
    mean_gap: float = 0.0
    seconds: float = 0.0

    def to_dict(self) -> dict:
        return {
            "model": self.model,
            "n_folds": self.n_folds,
            "n_samples": self.n_samples,
            "mean_gap": round(self.mean_gap, 4),
            "seconds": round(self.seconds, 1),
            "folds": [asdict(f) for f in self.folds],
            "oof_metrics": self.oof_metrics,
            "oof_metrics_calibrated": self.oof_metrics_calibrated,
        }

    def fold_spread(self, key: str = "macro_f1") -> tuple[float, float]:
        vals = [f.metrics.get(key, 0.0) for f in self.folds if f.ok]
        return (float(np.mean(vals)), float(np.std(vals))) if vals else (0.0, 0.0)


class _Subset(torch.utils.data.Dataset):
    def __init__(self, base, indices):
        self.base = base
        self.indices = list(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        return self.base[self.indices[i]]


def run_cross_validation(
    model_name: str,
    build,
    train_base,
    eval_base,
    labels: np.ndarray,
    n_folds: int = 5,
    inner_val_frac: float = 0.15,
    batch_size: int = 8,
    max_minutes_per_fold: float = 14.0,
    device: str = "cpu",
    config: TrainConfig | None = None,
    checkpoint_dir: Path | None = None,
    reports_dir: Path | None = None,
    seed: int = 42,
    verbose: bool = True,
) -> CrossValResult:
    """Train ``n_folds`` models and assemble out-of-fold predictions over every sample."""
    t0 = time.time()
    config = config or TrainConfig()
    labels = np.asarray(labels)
    result = CrossValResult(model=model_name, n_folds=n_folds, n_samples=len(labels))

    folds = kfold_indices(labels, n_splits=n_folds, seed=seed)
    oof_logits = np.zeros((len(labels), 4), dtype=np.float32)
    oof_cal = np.zeros((len(labels), 4), dtype=np.float32)
    scored = np.zeros(len(labels), dtype=bool)
    rng = np.random.default_rng(seed)

    for k, (train_idx, test_idx) in enumerate(folds, 1):
        f_t0 = time.time()
        if verbose:
            print(f"\n  --- fold {k}/{len(folds)}  train={len(train_idx)} test={len(test_idx)}", flush=True)

        # Inner split: early stopping and calibration are fitted here, never on test_idx.
        perm = rng.permutation(len(train_idx))
        n_inner = max(8, int(len(train_idx) * inner_val_frac))
        inner_val = train_idx[perm[:n_inner]]
        inner_train = train_idx[perm[n_inner:]]

        try:
            model = build()
            res = train_model(
                model,
                _Subset(train_base, inner_train),
                _Subset(eval_base, inner_val),
                config,
                device=device,
                batch_size=batch_size,
                max_minutes=max_minutes_per_fold,
                class_weights=None,  # the sampler owns imbalance correction; see loop._sampler_kwargs
                checkpoint=(checkpoint_dir / f"{model_name}_fold{k}.pt") if checkpoint_dir else None,
                verbose=verbose,
                tag=f"{model_name} f{k}",
            )
            if not res.ok:
                raise RuntimeError(res.error)

            dev = torch.device(device)
            # Unweighted at evaluation time: training uses a frequency-aware sampler instead of
            # loss weights, and scoring should reflect the real class distribution.
            loss_fn = DualLoss(class_weights=None)

            # Calibrate on the inner validation slice.
            _, _, val_logits, val_labels = evaluate(
                model,
                torch.utils.data.DataLoader(_Subset(eval_base, inner_val), batch_size=batch_size),
                dev, loss_fn, collect_seg=False, return_logits=True,
            )
            cal = fit_temperature(val_logits, val_labels)

            # Score the outer fold exactly once.
            m, _, test_logits, test_labels = evaluate(
                model,
                torch.utils.data.DataLoader(_Subset(eval_base, test_idx), batch_size=batch_size),
                dev, loss_fn, collect_seg=True, return_logits=True,
            )
            oof_logits[test_idx] = test_logits
            oof_cal[test_idx] = apply_temperature(test_logits, cal.temperature)
            scored[test_idx] = True

            result.folds.append(FoldResult(
                fold=k, n_train=len(inner_train), n_inner_val=len(inner_val),
                n_test=len(test_idx), best_val_f1=res.best_val_f1, gap=res.gap,
                epochs_run=len(res.epochs), seconds=time.time() - f_t0,
                metrics=m.to_dict(), calibration=cal.to_dict(),
            ))
            if verbose:
                print(f"    fold {k}: {m.summary()}  gap={res.gap:+.3f}  {cal.summary()}", flush=True)

        except Exception as exc:  # noqa: BLE001 - a failed fold must not lose the others
            result.folds.append(FoldResult(
                fold=k, n_train=len(train_idx), n_inner_val=0, n_test=len(test_idx),
                best_val_f1=0.0, gap=0.0, epochs_run=0, seconds=time.time() - f_t0,
                ok=False, error=f"{type(exc).__name__}: {exc}",
            ))
            if verbose:
                print(f"    fold {k} FAILED: {exc}", flush=True)

        # Persist after every fold: an interrupted run still reports what completed.
        if reports_dir is not None:
            result.seconds = time.time() - t0
            _finalise(result, labels, oof_logits, oof_cal, scored)
            (reports_dir / f"cv_{model_name}.json").write_text(json.dumps(result.to_dict(), indent=2))

    result.seconds = time.time() - t0
    _finalise(result, labels, oof_logits, oof_cal, scored)
    return result


def _finalise(result: CrossValResult, labels, oof_logits, oof_cal, scored) -> None:
    """Score the assembled out-of-fold predictions over every sample that has one.

    Classification metrics are computed on the pooled out-of-fold predictions, which is the
    whole point: every tile scored once, by a model that never saw it.

    Segmentation is different. Holding full-resolution probability maps for every tile is not
    worth the memory, so IoU and Dice are carried across as the *mean over folds* rather than
    recomputed on a pooled array. Leaving them at the default would report 0.000 next to
    per-fold values of 0.25-0.29, which reads as total failure rather than as a missing number.
    """
    good = [f for f in result.folds if f.ok]
    result.mean_gap = float(np.mean([f.gap for f in good])) if good else 0.0
    if not scored.any():
        return

    y_true = labels[scored]
    pooled = compute_metrics(y_true, oof_logits[scored].argmax(1)).to_dict()
    pooled_cal = compute_metrics(y_true, oof_cal[scored].argmax(1)).to_dict()

    # Segmentation scores are fold-averaged, not pooled -- see the docstring.
    for key in ("iou", "dice"):
        vals = [f.metrics.get(key, 0.0) for f in good if f.metrics]
        mean_val = float(np.mean(vals)) if vals else 0.0
        pooled[key] = mean_val
        pooled_cal[key] = mean_val

    pooled["n_scored"] = int(scored.sum())
    pooled["segmentation_is_fold_mean"] = True
    result.oof_metrics = pooled
    result.oof_metrics_calibrated = pooled_cal


def summarise(results: list[CrossValResult]) -> str:
    rows = ["| Model | OOF macro-F1 | fold σ | Accuracy | within-1 | binary F1 | mean gap |",
            "|---|---:|---:|---:|---:|---:|---:|"]
    for r in results:
        m = r.oof_metrics
        mean, std = r.fold_spread()
        rows.append(
            f"| {r.model} | **{m.get('macro_f1', 0):.3f}** | ±{std:.3f} "
            f"| {m.get('accuracy', 0):.3f} | {m.get('within_one_accuracy', 0):.3f} "
            f"| {m.get('binary_f1', 0):.3f} | {r.mean_gap:+.3f} |"
        )
    return "\n".join(rows)


__all__ = ["CrossValResult", "FoldResult", "run_cross_validation", "summarise",
           "CalibrationResult", "Metrics"]
