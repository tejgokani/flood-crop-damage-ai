"""v2 cross-validated runner.

Same discipline as v1 -- every real tile scored once, out of fold, with early stopping and
calibration fitted on an inner split -- plus the one thing v2 adds: synthetic tiles joining the
*training* portion of each fold and nothing else.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch

from fcda.eval.calibration import apply_temperature, fit_temperature
from fcda.eval.metrics import compute_metrics
from fcda.preprocess.transforms import build_transform
from fcda.splits import kfold_indices
from fcda.train.correction import TrainConfig
from fcda.train.loop import DualLoss, evaluate, train_model

from ..data.dataset import (
    FloodDataset,
    TileRef,
    assert_no_synthetic_in_eval,
    build_synthetic,
    plan_synthetic,
)

MODE_FOR_MODEL = {
    "yolo12_unet": "change",
    "cnn_lstm": "temporal",
    "effnet_attention": "single",
}


@dataclass
class FoldOutcome:
    fold: int
    n_real_train: int
    n_synthetic: int
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
class RunResult:
    model: str
    use_synthetic: bool
    n_folds: int
    folds: list[FoldOutcome] = field(default_factory=list)
    oof: dict = field(default_factory=dict)
    mean_gap: float = 0.0
    seconds: float = 0.0

    def to_dict(self) -> dict:
        return {
            "model": self.model,
            "use_synthetic": self.use_synthetic,
            "n_folds": self.n_folds,
            "mean_gap": round(self.mean_gap, 4),
            "seconds": round(self.seconds, 1),
            "folds": [asdict(f) for f in self.folds],
            "oof": self.oof,
        }


class _Subset(torch.utils.data.Dataset):
    def __init__(self, base, indices):
        self.base, self.indices = base, list(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        return self.base[self.indices[i]]


def run(
    model_name: str,
    build,
    real: list[TileRef],
    data_root: Path,
    size: int = 192,
    n_folds: int = 3,
    use_synthetic: bool = True,
    extra_healthy: list[TileRef] | None = None,
    per_class_target: int = 600,
    inner_val_frac: float = 0.15,
    batch_size: int = 8,
    max_minutes: float = 25.0,
    epochs: int = 30,
    samples_per_epoch: int | None = None,
    device: str = "cpu",
    reports_dir: Path | None = None,
    seed: int = 42,
    verbose: bool = True,
) -> RunResult:
    t0 = time.time()
    mode = MODE_FOR_MODEL[model_name]
    labels = np.array([r.label for r in real])
    result = RunResult(model=model_name, use_synthetic=use_synthetic, n_folds=n_folds)
    rng = np.random.default_rng(seed)

    train_tf = build_transform(train=True, use_clahe=True, seed=seed)
    eval_tf = build_transform(train=False, use_clahe=True, seed=seed)
    config = TrainConfig(epochs=epochs)

    oof_pred = np.full(len(real), -1, dtype=int)

    for k, (train_idx, test_idx) in enumerate(kfold_indices(labels, n_splits=n_folds, seed=seed), 1):
        f0 = time.time()
        perm = rng.permutation(len(train_idx))
        n_inner = max(16, int(len(train_idx) * inner_val_frac))
        inner_val = train_idx[perm[:n_inner]]
        inner_train = train_idx[perm[n_inner:]]

        real_train = [real[i] for i in inner_train]
        # Extra real Healthy tiles join training only -- they are outside the benchmark, so they
        # can never leak into an evaluation fold.
        extra = list(extra_healthy or [])
        synthetic: list[TileRef] = []
        if use_synthetic:
            plan = plan_synthetic(real_train + extra, per_class_target=per_class_target)
            synthetic = build_synthetic(plan, seed=seed * 100 + k)
            if verbose:
                print(f"\n  --- fold {k}: {len(real_train)} benchmark + {len(extra)} extra-healthy "
                      f"+ {len(synthetic)} synthetic (plan {plan})", flush=True)
        elif verbose:
            print(f"\n  --- fold {k}: {len(real_train)} benchmark + {len(extra)} extra-healthy, "
                  f"no synthetic", flush=True)

        # The guard that makes the separation checkable rather than trusted.
        eval_refs = [real[i] for i in np.concatenate([inner_val, test_idx])]
        assert_no_synthetic_in_eval(eval_refs, f"fold{k}-eval")

        try:
            train_ds = FloodDataset(real_train + extra + synthetic, data_root, size, mode, train_tf)
            val_ds = FloodDataset([real[i] for i in inner_val], data_root, size, mode, eval_tf)
            test_ds = FloodDataset([real[i] for i in test_idx], data_root, size, mode, eval_tf)

            model = build()
            # Fixed samples-per-epoch so the with/without-synthetic arms get identical compute
            # per epoch; otherwise the larger arm simply gets fewer epochs.
            res = train_model(
                model, train_ds, val_ds, config, device=device, batch_size=batch_size,
                max_minutes=max_minutes, class_weights=None, verbose=verbose,
                samples_per_epoch=samples_per_epoch,
                tag=f"{model_name} f{k}{'+syn' if use_synthetic else ''}",
            )
            if not res.ok:
                raise RuntimeError(res.error)

            dev = torch.device(device)
            loss_fn = DualLoss(class_weights=None)
            _, _, vlog, vlab = evaluate(
                model, torch.utils.data.DataLoader(val_ds, batch_size=batch_size),
                dev, loss_fn, collect_seg=False, return_logits=True)
            cal = fit_temperature(vlog, vlab)

            m, _, tlog, _ = evaluate(
                model, torch.utils.data.DataLoader(test_ds, batch_size=batch_size),
                dev, loss_fn, collect_seg=True, return_logits=True)
            oof_pred[test_idx] = apply_temperature(tlog, cal.temperature).argmax(1)

            result.folds.append(FoldOutcome(
                fold=k, n_real_train=len(real_train) + len(extra), n_synthetic=len(synthetic),
                n_test=len(test_idx), best_val_f1=res.best_val_f1, gap=res.gap,
                epochs_run=len(res.epochs), seconds=time.time() - f0,
                metrics=m.to_dict(), calibration=cal.to_dict(),
            ))
            if verbose:
                print(f"    fold {k}: {m.summary()}  gap={res.gap:+.3f}  {cal.summary()}", flush=True)

        except Exception as exc:  # noqa: BLE001 - a failed fold must not lose the others
            result.folds.append(FoldOutcome(
                fold=k, n_real_train=len(real_train), n_synthetic=len(synthetic),
                n_test=len(test_idx), best_val_f1=0.0, gap=0.0, epochs_run=0,
                seconds=time.time() - f0, ok=False, error=f"{type(exc).__name__}: {exc}"))
            if verbose:
                print(f"    fold {k} FAILED: {exc}", flush=True)

        if reports_dir is not None:
            _finalise(result, labels, oof_pred)
            result.seconds = time.time() - t0
            tag = f"{model_name}{'_syn' if use_synthetic else '_nosyn'}"
            (reports_dir / f"run_{tag}.json").write_text(json.dumps(result.to_dict(), indent=2))

    _finalise(result, labels, oof_pred)
    result.seconds = time.time() - t0
    return result


def _finalise(result: RunResult, labels: np.ndarray, oof_pred: np.ndarray) -> None:
    good = [f for f in result.folds if f.ok]
    result.mean_gap = float(np.mean([f.gap for f in good])) if good else 0.0
    scored = oof_pred >= 0
    if not scored.any():
        return
    m = compute_metrics(labels[scored], oof_pred[scored])
    d = m.to_dict()
    # Segmentation is fold-averaged: pooling full-resolution maps is not worth the memory.
    for key in ("iou", "dice"):
        vals = [f.metrics.get(key, 0.0) for f in good if f.metrics]
        d[key] = float(np.mean(vals)) if vals else 0.0
    d["n_scored"] = int(scored.sum())
    result.oof = d
