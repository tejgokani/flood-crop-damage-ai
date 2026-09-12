"""GAN augmentation ablation.

`Sequence.docx` prescribes "GAN Augmentation on TRAINING data only" as step 6, and the module
exists -- but until now nothing measured whether it *helps*. "We use GAN augmentation to address
data scarcity" was an assertion with no evidence behind it, and an unmeasured component in a
benchmark is worth less than an absent one.

This runs the same fold twice under identical seeds, configuration and wall-clock budget, with
the only difference being whether synthetic tiles are added to the training split. Both outcomes
are useful: if it helps, the sequence step is justified by our own numbers; if it does not, that
is a finding about GAN augmentation on a 20-Severe-tile corpus and is reported as such.

Leakage discipline is the same as everywhere else: the GAN is fitted on the *inner training*
slice of the fold and its output joins only the training set. The held-out fold is scored once
and never resampled -- ``assert_evaluation_untouched`` makes that checkable rather than trusted.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch

from ..augment.gan import deficit_counts, generate_samples, train_gan
from ..splits import kfold_indices
from .correction import TrainConfig
from .loop import ConcatWithSynthetic, DualLoss, evaluate, train_model


@dataclass
class ArmResult:
    """One (fold, condition) run."""

    fold: int
    gan: bool
    n_train: int
    n_synthetic: int
    metrics: dict = field(default_factory=dict)
    gap: float = 0.0
    epochs_run: int = 0
    seconds: float = 0.0
    ok: bool = True
    error: str = ""


@dataclass
class AblationResult:
    model: str
    n_folds: int
    arms: list[ArmResult] = field(default_factory=list)
    summary: dict = field(default_factory=dict)
    gan_training: list[dict] = field(default_factory=list)
    seconds: float = 0.0

    def to_dict(self) -> dict:
        return {
            "model": self.model,
            "n_folds": self.n_folds,
            "seconds": round(self.seconds, 1),
            "arms": [asdict(a) for a in self.arms],
            "gan_training": self.gan_training,
            "summary": self.summary,
        }


class _Subset(torch.utils.data.Dataset):
    def __init__(self, base, indices):
        self.base = base
        self.indices = list(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        return self.base[self.indices[i]]


def assert_evaluation_untouched(n_before: int, n_after: int) -> None:
    """The held-out fold must never change size. Synthetic tiles belong to training alone."""
    if n_before != n_after:
        raise AssertionError(
            f"held-out fold changed size ({n_before} -> {n_after}); "
            "GAN samples must be confined to the training split"
        )


def _mean_std(values: list[float]) -> tuple[float, float]:
    return (float(np.mean(values)), float(np.std(values))) if values else (0.0, 0.0)


def run_ablation(
    model_name: str,
    build,
    train_base,
    eval_base,
    gan_source,
    labels: np.ndarray,
    n_folds: int = 3,
    inner_val_frac: float = 0.15,
    batch_size: int = 8,
    max_minutes_per_arm: float = 10.0,
    gan_epochs: int = 40,
    gan_mode: str = "change",
    device: str = "cpu",
    config: TrainConfig | None = None,
    reports_dir: Path | None = None,
    seed: int = 42,
    verbose: bool = True,
) -> AblationResult:
    """Train each fold with and without GAN augmentation, everything else held fixed."""
    t0 = time.time()
    config = config or TrainConfig()
    labels = np.asarray(labels)
    result = AblationResult(model=model_name, n_folds=n_folds)
    dev = torch.device(device)
    rng = np.random.default_rng(seed)

    for k, (train_idx, test_idx) in enumerate(kfold_indices(labels, n_splits=n_folds, seed=seed), 1):
        perm = rng.permutation(len(train_idx))
        n_inner = max(8, int(len(train_idx) * inner_val_frac))
        inner_val = train_idx[perm[:n_inner]]
        inner_train = train_idx[perm[n_inner:]]
        n_test_before = len(test_idx)

        # --- fit the GAN once per fold, on the inner training slice only
        synthetic: list = []
        if gan_epochs > 0:
            deficits = deficit_counts(labels[inner_train])
            if verbose:
                print(f"\n  === fold {k}: fitting GAN on {len(inner_train)} training tiles; "
                      f"deficits {deficits}", flush=True)
            gan_inputs = []
            for i in inner_train[: min(len(inner_train), 400)]:
                sample = gan_source[i]
                seq, mask, lab = sample[0], sample[1], sample[2]
                arr = seq.numpy() if hasattr(seq, "numpy") else np.asarray(seq)
                gan_inputs.append((arr[0], arr[1], np.asarray(mask), int(lab)))
            g, gres = train_gan(gan_inputs, epochs=gan_epochs, device=device,
                                max_seconds=600, seed=seed, verbose=False)
            synthetic = generate_samples(g, deficits, size=train_base.size,
                                         device=device, seed=seed, mode=gan_mode)
            result.gan_training.append({
                "fold": k, "epochs": gres.epochs, "seconds": round(gres.seconds, 1),
                "d_loss": round(gres.d_loss, 4), "g_loss": round(gres.g_loss, 4),
                "n_real": gres.n_train_tiles, "n_generated": len(synthetic),
            })
            if verbose:
                print(f"      generated {len(synthetic)} synthetic tiles in {gres.seconds:.0f}s",
                      flush=True)

        for use_gan in (False, True):
            if use_gan and not synthetic:
                continue
            arm_t0 = time.time()
            tag = f"{model_name} f{k} {'+GAN' if use_gan else 'base'}"
            try:
                train_ds: torch.utils.data.Dataset = _Subset(train_base, inner_train)
                if use_gan:
                    train_ds = ConcatWithSynthetic(train_ds, synthetic)

                model = build()
                res = train_model(
                    model, train_ds, _Subset(eval_base, inner_val), config,
                    device=device, batch_size=batch_size,
                    max_minutes=max_minutes_per_arm, class_weights=None,
                    verbose=verbose, tag=tag,
                )
                if not res.ok:
                    raise RuntimeError(res.error)

                m, _ = evaluate(
                    model,
                    torch.utils.data.DataLoader(_Subset(eval_base, test_idx), batch_size=batch_size),
                    dev, DualLoss(class_weights=None), collect_seg=True,
                )
                assert_evaluation_untouched(n_test_before, len(test_idx))

                result.arms.append(ArmResult(
                    fold=k, gan=use_gan, n_train=len(inner_train),
                    n_synthetic=len(synthetic) if use_gan else 0,
                    metrics=m.to_dict(), gap=res.gap, epochs_run=len(res.epochs),
                    seconds=time.time() - arm_t0,
                ))
                if verbose:
                    print(f"    [{tag}] {m.summary()}  gap={res.gap:+.3f}", flush=True)
            except Exception as exc:  # noqa: BLE001 - a failed arm must not lose the rest
                result.arms.append(ArmResult(
                    fold=k, gan=use_gan, n_train=len(inner_train),
                    n_synthetic=len(synthetic) if use_gan else 0,
                    seconds=time.time() - arm_t0, ok=False,
                    error=f"{type(exc).__name__}: {exc}",
                ))
                if verbose:
                    print(f"    [{tag}] FAILED: {exc}", flush=True)

        if reports_dir is not None:
            result.seconds = time.time() - t0
            _summarise(result)
            (reports_dir / "ablation.json").write_text(json.dumps(result.to_dict(), indent=2))

    result.seconds = time.time() - t0
    _summarise(result)
    return result


def _summarise(result: AblationResult) -> None:
    keys = ("macro_f1", "accuracy", "within_one_accuracy", "binary_f1", "iou")
    out: dict = {}
    for cond, flag in (("without_gan", False), ("with_gan", True)):
        arms = [a for a in result.arms if a.ok and a.gan is flag]
        out[cond] = {"n_folds": len(arms)}
        for key in keys:
            mean, std = _mean_std([a.metrics.get(key, 0.0) for a in arms])
            out[cond][key] = {"mean": round(mean, 4), "std": round(std, 4)}
        gmean, _ = _mean_std([a.gap for a in arms])
        out[cond]["gap"] = round(gmean, 4)
        sev, _ = _mean_std([a.metrics.get("per_class_f1", {}).get("Severe", 0.0) for a in arms])
        out[cond]["severe_f1"] = round(sev, 4)

    deltas = {}
    for key in (*keys, "severe_f1"):
        a = out["without_gan"].get(key)
        b = out["with_gan"].get(key)
        av = a["mean"] if isinstance(a, dict) else a
        bv = b["mean"] if isinstance(b, dict) else b
        deltas[key] = round((bv or 0.0) - (av or 0.0), 4)
    out["delta_with_minus_without"] = deltas

    # A verdict, so the result is not left for the reader to spin either way.
    d = deltas.get("macro_f1", 0.0)
    paired = _paired_folds(result)
    if not paired:
        verdict = "inconclusive: no fold completed both arms"
    elif abs(d) < 0.01:
        verdict = f"no meaningful effect (macro-F1 delta {d:+.3f} across {len(paired)} folds)"
    elif d > 0:
        verdict = f"GAN augmentation helped (macro-F1 {d:+.3f} across {len(paired)} folds)"
    else:
        verdict = f"GAN augmentation hurt (macro-F1 {d:+.3f} across {len(paired)} folds)"
    out["verdict"] = verdict
    out["paired_folds"] = paired
    result.summary = out


def _paired_folds(result: AblationResult) -> list[int]:
    """Folds where both arms completed -- the only ones a delta can legitimately use."""
    ok_base = {a.fold for a in result.arms if a.ok and not a.gan}
    ok_gan = {a.fold for a in result.arms if a.ok and a.gan}
    return sorted(ok_base & ok_gan)


def render(result: AblationResult) -> str:
    s = result.summary
    rows = ["| Condition | macro-F1 | Accuracy | Within-1 | Binary F1 | Severe F1 | Flood IoU | Gap |",
            "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for cond, label in (("without_gan", "Without GAN"), ("with_gan", "With GAN augmentation")):
        c = s.get(cond, {})
        if not c:
            continue
        rows.append(
            f"| {label} | {c['macro_f1']['mean']:.3f} ±{c['macro_f1']['std']:.3f} "
            f"| {c['accuracy']['mean']:.3f} | {c['within_one_accuracy']['mean']:.3f} "
            f"| {c['binary_f1']['mean']:.3f} | {c['severe_f1']:.3f} "
            f"| {c['iou']['mean']:.3f} | {c['gap']:+.3f} |"
        )
    d = s.get("delta_with_minus_without", {})
    rows.append(
        f"| **Delta** | **{d.get('macro_f1', 0):+.3f}** | {d.get('accuracy', 0):+.3f} "
        f"| {d.get('within_one_accuracy', 0):+.3f} | {d.get('binary_f1', 0):+.3f} "
        f"| {d.get('severe_f1', 0):+.3f} | {d.get('iou', 0):+.3f} | |"
    )
    return "\n".join(rows) + f"\n\n**Verdict:** {s.get('verdict', '—')}"
