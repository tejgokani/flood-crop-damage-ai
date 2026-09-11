"""Pipeline orchestration -- both of Ma'am's sequences, executed literally.

Image sequence (``run_image_pipeline``):
    Load -> Preprocess -> Leakage checks -> CLAHE -> Split / 10-fold
    -> GAN augmentation (TRAIN only) -> Initial training
    -> Over/under-fit detection -> Apply correction and retrain -> Final test evaluation

CSV sequence (``run_tabular_pipeline``):
    Load -> Preprocess -> Leakage checks -> Split / 10-fold
    -> SMOTE (TRAIN only) -> Initial training
    -> Over/under-fit detection -> Apply correction and retrain -> Final test evaluation

``run_progressive`` walks the dataset tier ladder, running the image sequence at each rung
and stopping when ``capacity.can_advance`` refuses the next one.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from . import NUM_CLASSES, SEVERITY_CLASSES
from .augment.gan import deficit_counts, generate_samples, train_gan
from .augment.smote import balance_training_split
from .data import capacity as cap
from .data.etci import FloodTileDataset, build_index, load_planes, synthetic_dataset
from .data.tiers import Tier, get_tier, next_tier
from .eval.metrics import Metrics, compute_metrics
from .preprocess.leakage import (
    average_hash,
    check_duplicate_leakage,
    check_suspicious_features,
    check_target_leakage,
    merge_reports,
)
from .preprocess.transforms import build_transform
from .splits import grouped_stratified_split, kfold_indices, stratified_split
from .train.correction import TrainConfig, plan_correction
from .train.diagnostics import FitStatus, diagnose
from .train.loop import (
    ConcatWithSynthetic,
    compute_class_weights,
    evaluate,
    pick_device,
    train_model,
)


class Subset(torch.utils.data.Dataset):
    """Index-selected view of a dataset (the split applied)."""

    def __init__(self, base, indices):
        self.base = base
        self.indices = list(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        return self.base[self.indices[i]]


@dataclass
class ModelOutcome:
    name: str
    initial: dict = field(default_factory=dict)
    diagnosis: dict = field(default_factory=dict)
    correction: dict = field(default_factory=dict)
    retrained: dict | None = None
    #: Which run supplied the model used for the final test: "initial" or "corrected".
    selected_run: str = "initial"
    final_test: dict | None = None
    seconds: float = 0.0
    ok: bool = True
    error: str = ""


@dataclass
class PipelineResult:
    tier: str
    n_tiles: int
    image_size: int
    device: str
    class_distribution: dict = field(default_factory=dict)
    leakage: dict = field(default_factory=dict)
    gan: dict = field(default_factory=dict)
    models: list[ModelOutcome] = field(default_factory=list)
    seconds: float = 0.0

    def to_dict(self) -> dict:
        return {
            "tier": self.tier,
            "n_tiles": self.n_tiles,
            "image_size": self.image_size,
            "device": self.device,
            "class_distribution": self.class_distribution,
            "leakage": self.leakage,
            "gan": self.gan,
            "seconds": round(self.seconds, 1),
            "models": [m.__dict__ for m in self.models],
        }


def _banner(text: str) -> None:
    print(f"\n{'=' * 74}\n  {text}\n{'=' * 74}", flush=True)


def _step(n: int, text: str) -> None:
    print(f"\n[{n}] {text}", flush=True)


def run_image_pipeline(
    tier: Tier,
    data_root: Path,
    model_names: list[str],
    reports_dir: Path,
    offline: bool = False,
    max_minutes_per_model: float = 25.0,
    gan_epochs: int = 30,
    device: str = "auto",
    seed: int = 42,
    verbose: bool = True,
) -> PipelineResult:
    """The image sequence, in Ma'am's order, at one dataset tier."""
    from .models.registry import DISPLAY_NAMES, build_model, is_temporal, wants_change_input

    t0 = time.time()
    dev = pick_device(device)
    result = PipelineResult(tier=tier.name, n_tiles=tier.n_tiles, image_size=tier.image_size, device=dev)
    _banner(f"IMAGE PIPELINE  tier={tier.name}  n={tier.n_tiles}  size={tier.image_size}px  device={dev}")

    # ---------------------------------------------------------------- 1. Load
    _step(1, "Load image dataset")
    if offline:
        records = []
        base_spatial = synthetic_dataset(tier.n_tiles, tier.image_size, seed=seed)
        base_temporal = synthetic_dataset(tier.n_tiles, tier.image_size, seed=seed, temporal=True)
        base_change = synthetic_dataset(tier.n_tiles, tier.image_size, seed=seed, change=True)
        labels = np.array([base_spatial.synthetic[i]["severity"].label for i in range(tier.n_tiles)])
        print(f"    synthetic tier: {tier.n_tiles} tiles (offline)")
    else:
        records = build_index(data_root, tier.n_tiles, seed=seed, progress=verbose)
        labels = np.array([r.label for r in records])
        print(f"    loaded {len(records)} real ETCI tiles")

    dist = {SEVERITY_CLASSES[i]: int((labels == i).sum()) for i in range(NUM_CLASSES)}
    result.class_distribution = dist
    print(f"    class distribution: {dist}")

    # ------------------------------------------------------- 2. Preprocessing
    _step(2, "Data preprocessing (VV/VH stack + derived ratio, resize, normalise)")
    train_tf = build_transform(train=True, use_clahe=True, seed=seed)
    eval_tf = build_transform(train=False, use_clahe=True, seed=seed)
    print(f"    train chain: {train_tf}")
    print(f"    eval  chain: {eval_tf}")

    # ----------------------------------------------------- 3. Leakage checks
    _step(3, "Data leakage checks (duplicate / near-duplicate tiles across splits)")
    split = stratified_split(labels, seed=seed)
    print(f"    split sizes: {split.sizes()}")

    if not offline and records:
        hashes = {}
        for split_name, idx in (("train", split.train), ("val", split.val), ("test", split.test)):
            hs = {}
            for i in idx[: min(len(idx), 400)]:
                planes = load_planes(data_root, records[i].tile_id)
                hs[records[i].tile_id] = average_hash(planes["post_vv"])
            hashes[split_name] = hs
        leak = check_duplicate_leakage(hashes, max_distance=2)
    else:
        leak = check_duplicate_leakage({})
    result.leakage = leak.to_dict()
    print("    " + (leak.render().replace("\n", "\n    ") if leak.findings else "No leakage findings."))

    # -------------------------------------------------------------- 4. CLAHE
    _step(4, "CLAHE contrast enhancement (applied to SAR channels in the transform chain)")
    print("    CLAHE(clip_limit=2.0, tile_grid=8) on VV and VH; ratio channel left untouched")

    # --------------------------------------------- 5. Split + 10-fold folds
    _step(5, "Train / validation / test split and 10-fold cross-validation indices")
    folds = kfold_indices(labels[split.train], n_splits=10, seed=seed)
    print(f"    stratified 70/15/15 split; {len(folds)} CV folds prepared over the training split")

    # ------------------------------------- 6. GAN augmentation (TRAIN ONLY)
    _step(6, "GAN augmentation on TRAINING data only")
    synthetic_samples: list = []
    if offline:
        base_for_gan = base_spatial
    else:
        base_for_gan = FloodTileDataset(records, data_root, size=tier.image_size, transform=eval_tf)

    train_labels = labels[split.train]
    deficits = deficit_counts(train_labels, cap=int(np.bincount(train_labels).max()))
    print(f"    training-split class counts: {dict(enumerate(np.bincount(train_labels, minlength=4)))}")
    print(f"    synthetic tiles required to balance: {deficits}")

    if sum(deficits.values()) > 0 and gan_epochs > 0:
        gan_inputs = []
        for i in split.train[: min(len(split.train), 400)]:
            sample = base_for_gan[i]
            img, mask, lab = sample[0], sample[1], sample[2]
            gan_inputs.append((img.numpy(), mask.numpy(), int(lab)))
        g, gan_res = train_gan(
            gan_inputs, epochs=gan_epochs, device=dev, max_seconds=900, seed=seed, verbose=verbose
        )
        synthetic_samples = generate_samples(g, deficits, size=tier.image_size, device=dev, seed=seed)
        result.gan = {
            "epochs": gan_res.epochs,
            "seconds": round(gan_res.seconds, 1),
            "d_loss": round(gan_res.d_loss, 4),
            "g_loss": round(gan_res.g_loss, 4),
            "n_real_used": gan_res.n_train_tiles,
            "n_generated": len(synthetic_samples),
            "per_class": {SEVERITY_CLASSES[k]: v for k, v in deficits.items()},
        }
        print(f"    generated {len(synthetic_samples)} synthetic training tiles")
    else:
        print("    skipped (already balanced or gan_epochs=0)")

    class_weights = compute_class_weights(train_labels)

    # ----------------------------- 7-9. Train, diagnose, correct, retrain
    for name in model_names:
        temporal = is_temporal(name)
        change = wants_change_input(name)
        if offline:
            base_tr = base_ev = base_temporal if temporal else (base_change if change else base_spatial)
        else:
            base_tr = FloodTileDataset(
                records, data_root, size=tier.image_size, transform=train_tf,
                temporal=temporal, change=change,
            )
            base_ev = FloodTileDataset(
                records, data_root, size=tier.image_size, transform=eval_tf,
                temporal=temporal, change=change,
            )

        train_ds: torch.utils.data.Dataset = Subset(base_tr, split.train)
        # GAN samples are spatial; the temporal model keeps the real pairs only.
        # GAN tiles are single-date 3-channel images, so they can only join a model that
        # consumes single-date 3-channel input -- not the temporal pair or the change stack.
        if synthetic_samples and not temporal and not change:
            train_ds = ConcatWithSynthetic(train_ds, synthetic_samples)
        val_ds = Subset(base_ev, split.val)
        test_ds = Subset(base_ev, split.test)

        outcome = _train_diagnose_correct(
            name=name,
            display=DISPLAY_NAMES.get(name, name),
            build=lambda n=name: build_model(n, pretrained=not (is_temporal(n) or wants_change_input(n))),
            train_ds=train_ds,
            val_ds=val_ds,
            test_ds=test_ds,
            tier=tier,
            dev=dev,
            class_weights=class_weights,
            max_minutes=max_minutes_per_model,
            reports_dir=reports_dir,
            verbose=verbose,
        )
        result.models.append(outcome)

    result.seconds = time.time() - t0
    return result


def _train_diagnose_correct(
    name,
    display,
    build,
    train_ds,
    val_ds,
    test_ds,
    tier,
    dev,
    class_weights,
    max_minutes,
    reports_dir,
    verbose,
) -> ModelOutcome:
    """Steps 7-10: initial training, diagnosis, correction + retrain, final test."""
    from .train.loop import DualLoss

    t0 = time.time()
    outcome = ModelOutcome(name=name)
    _banner(f"{display}   [tier {tier.name}]")

    config = TrainConfig(epochs=tier.epochs)
    ckpt = reports_dir.parent / "checkpoints" / f"{name}_{tier.name}.pt"

    _step(7, f"Initial model training -- {display}")
    model = build()
    first = train_model(
        model, train_ds, val_ds, config, device=dev, batch_size=tier.batch_size,
        max_minutes=max_minutes, class_weights=class_weights, checkpoint=ckpt,
        verbose=verbose, tag=name,
    )
    outcome.initial = first.to_dict()
    if not first.ok:
        outcome.ok, outcome.error = False, first.error
        outcome.seconds = time.time() - t0
        return outcome

    _step(8, "Overfitting / underfitting detection")
    diag = diagnose(first.train_curve, first.val_curve)
    outcome.diagnosis = diag.to_dict()
    print(f"    diagnosis: {diag.status.value} -- {diag.reason}")

    _step(9, "Apply correction technique and retrain")
    corr = plan_correction(diag, config)
    outcome.correction = corr.to_dict()
    print(f"    {corr.describe()}")

    outcome.selected_run = "initial"
    if corr.applied:
        retrain_budget = max(max_minutes * 0.8, 3.0)
        model2 = build()
        second = train_model(
            model2, train_ds, val_ds, corr.config, device=dev, batch_size=tier.batch_size,
            max_minutes=retrain_budget, class_weights=class_weights,
            checkpoint=ckpt.with_name(f"{name}_{tier.name}_corrected.pt"),
            verbose=verbose, tag=f"{name}*",
        )
        outcome.retrained = second.to_dict()
        if second.ok and second.best_val_f1 > first.best_val_f1:
            print(f"    correction improved val macro-F1 {first.best_val_f1:.3f} -> {second.best_val_f1:.3f}")
            model = model2
            outcome.selected_run = "corrected"
        else:
            got = second.best_val_f1 if second.ok else float("nan")
            print(f"    correction did not improve validation ({first.best_val_f1:.3f} -> {got:.3f}); keeping the original")

    _step(10, "Final test evaluation")
    loader = torch.utils.data.DataLoader(test_ds, batch_size=tier.batch_size, shuffle=False)
    test_metrics, _ = evaluate(model, loader, dev, DualLoss(class_weights=class_weights.to(torch.device(dev))))
    outcome.final_test = test_metrics.to_dict()
    print(f"    TEST  {test_metrics.summary()}")
    print(f"    per-class F1: { {k: round(v,3) for k,v in test_metrics.per_class_f1.items()} }")

    outcome.seconds = time.time() - t0
    return outcome


def run_cv_pipeline(
    tier: Tier,
    data_root: Path,
    model_names: list[str],
    reports_dir: Path,
    offline: bool = False,
    n_folds: int = 5,
    max_minutes_per_fold: float = 14.0,
    gan_epochs: int = 0,
    device: str = "auto",
    seed: int = 42,
    verbose: bool = True,
) -> dict:
    """The image sequence with 10-fold-style cross validation as the headline evaluation.

    Same prescribed steps as ``run_image_pipeline`` -- load, preprocess, leakage checks, CLAHE,
    split, augment, train, diagnose, correct, evaluate -- but step 5 takes the cross-validation
    branch that ``Sequence.docx`` offers instead of the single three-way split, and step 10
    scores out-of-fold predictions covering every tile rather than a 135-tile holdout.
    """
    from .models.registry import DISPLAY_NAMES, build_model, is_temporal, wants_change_input
    from .train.crossval import run_cross_validation, summarise

    t0 = time.time()
    dev = pick_device(device)
    _banner(f"CROSS-VALIDATED PIPELINE  tier={tier.name}  n={tier.n_tiles}  "
            f"size={tier.image_size}px  folds={n_folds}  device={dev}")

    _step(1, "Load image dataset")
    if offline:
        records = []
        labels = np.array([], dtype=int)
    else:
        records = build_index(data_root, tier.n_tiles, seed=seed, progress=verbose)
        labels = np.array([r.label for r in records])
        print(f"    loaded {len(records)} real ETCI tiles")
    dist = {SEVERITY_CLASSES[i]: int((labels == i).sum()) for i in range(NUM_CLASSES)}
    print(f"    class distribution: {dist}")

    _step(2, "Data preprocessing")
    train_tf = build_transform(train=True, use_clahe=True, seed=seed)
    eval_tf = build_transform(train=False, use_clahe=True, seed=seed)
    print(f"    train chain: {train_tf}")

    _step(3, "Data leakage checks")
    if not offline and records:
        hashes = {}
        for fold_i, (_, te) in enumerate(kfold_indices(labels, n_splits=n_folds, seed=seed), 1):
            hs = {}
            for i in te[: min(len(te), 200)]:
                hs[records[i].tile_id] = average_hash(load_planes(data_root, records[i].tile_id)["post_vv"])
            hashes[f"fold{fold_i}"] = hs
        leak = check_duplicate_leakage(hashes, max_distance=2)
    else:
        leak = check_duplicate_leakage({})
    print("    " + (leak.render().replace("\n", "\n    ") if leak.findings else "No leakage findings."))

    _step(4, "CLAHE contrast enhancement")
    print("    CLAHE(clip_limit=2.0, tile_grid=8) on VV and VH")

    _step(5, f"{n_folds}-fold cross validation (the Sequence.docx alternative to a single split)")
    print(f"    every one of {len(labels)} tiles is scored exactly once, by a model that never saw it")
    print(f"    Severe tiles evaluated: {dist['Severe']} (a single 15% holdout would score ~{max(1, dist['Severe'] // n_folds)})")

    _step(6, "Augmentation -- training folds only")
    print(f"    geometric + photometric chain applied to inner-training data: {train_tf}")
    print("    GAN augmentation is evaluated separately as an ablation (see reports/ablation.json)")

    outputs: list = []
    for name in model_names:
        temporal = is_temporal(name)
        change = wants_change_input(name)
        train_base = FloodTileDataset(records, data_root, size=tier.image_size,
                                      transform=train_tf, temporal=temporal, change=change)
        eval_base = FloodTileDataset(records, data_root, size=tier.image_size,
                                     transform=eval_tf, temporal=temporal, change=change)
        _banner(f"{DISPLAY_NAMES.get(name, name)}   [{n_folds}-fold CV]")
        _step(7, f"Training {n_folds} folds -- steps 7-9 run inside each fold")
        cv = run_cross_validation(
            model_name=name,
            build=lambda n=name: build_model(n, pretrained=False),
            train_base=train_base, eval_base=eval_base, labels=labels,
            n_folds=n_folds, batch_size=tier.batch_size,
            max_minutes_per_fold=max_minutes_per_fold, device=dev,
            config=TrainConfig(epochs=tier.epochs),
            checkpoint_dir=reports_dir.parent / "checkpoints",
            reports_dir=reports_dir, seed=seed, verbose=verbose,
        )
        outputs.append(cv)

        _step(8, "Overfitting / underfitting detection across folds")
        gaps = [f.gap for f in cv.folds if f.ok]
        verdict = "HEALTHY" if cv.mean_gap <= 0.10 else "STILL OVERFITTING"
        print(f"    per-fold gaps: {[round(g, 3) for g in gaps]}")
        print(f"    mean train-val gap {cv.mean_gap:+.3f} -> {verdict} (threshold 0.10)")

        _step(10, "Out-of-fold evaluation over every tile")
        mean, std = cv.fold_spread()
        print(f"    OOF  {Metrics(**{k: v for k, v in cv.oof_metrics.items() if k in Metrics.__dataclass_fields__}).summary()}")
        print(f"    fold-to-fold macro-F1 {mean:.3f} +/- {std:.3f}")

    payload = {
        "tier": tier.name, "n_tiles": int(len(labels)), "image_size": tier.image_size,
        "device": dev, "n_folds": n_folds, "class_distribution": dist,
        "leakage": leak.to_dict(), "seconds": round(time.time() - t0, 1),
        "models": [c.to_dict() for c in outputs],
    }
    (reports_dir / "cv_results.json").write_text(json.dumps(payload, indent=2))
    print("\n" + summarise(outputs))
    return payload


def run_tabular_pipeline(
    csv_path: Path, reports_dir: Path, seed: int = 42, verbose: bool = True
) -> dict:
    """The CSV sequence, in Ma'am's order, with SMOTE on the training split only."""
    from sklearn.ensemble import RandomForestClassifier

    from .data.india_agri import FEATURE_COLS, LEAKY_COLS, build_tabular_dataset

    _banner("TABULAR PIPELINE -- Indian district crop statistics (ICRISAT)")
    out: dict = {}

    _step(1, "Load CSV dataset")
    df = build_tabular_dataset(csv_path)
    print(f"    {len(df)} rows | {df.district.nunique()} districts | {df.crop.nunique()} crops")

    _step(2, "Data preprocessing (wide -> long melt, yield anomaly, feature engineering)")
    counts = df.label.value_counts().sort_index()
    dist = {SEVERITY_CLASSES[i]: int(counts.get(i, 0)) for i in range(NUM_CLASSES)}
    print(f"    class distribution: {dist} (imbalance {counts.max() / counts.min():.1f}:1)")
    out["class_distribution"] = dist

    _step(3, "Data leakage checks (target / duplicate / suspicious features)")
    rep = merge_reports(
        check_target_leakage(df, "label", known_leaky=LEAKY_COLS),
        check_suspicious_features(df[FEATURE_COLS + ["label"]], "label"),
    )
    print("    " + rep.render().replace("\n", "\n    "))
    dropped = [c for c in LEAKY_COLS if c in df.columns]
    print(f"    action taken: dropping {dropped} before modelling")
    out["leakage"] = rep.to_dict()
    out["dropped_columns"] = dropped

    _step(4, "Train / validation / test split (grouped by district) and 10-fold CV")
    x = df[FEATURE_COLS].to_numpy(dtype=float)
    y = df.label.to_numpy()
    split = grouped_stratified_split(y, df.district.to_numpy(), seed=seed)
    print(f"    split sizes: {split.sizes()} | district overlap train/test: "
          f"{len(set(df.district.to_numpy()[split.train]) & set(df.district.to_numpy()[split.test]))}")
    out["split_sizes"] = split.sizes()

    _step(5, "SMOTE on TRAINING data only")
    sm = balance_training_split(x[split.train], y[split.train], seed=seed)
    print(f"    {sm.summary()}")
    print(f"    validation and test left untouched: {len(split.val)} / {len(split.test)} rows")
    out["smote"] = {
        "before": sm.before, "after": sm.after,
        "n_synthetic": sm.n_synthetic, "strategy": sm.strategy,
    }

    _step(6, "Initial model training (RandomForest)")
    clf = RandomForestClassifier(n_estimators=300, max_depth=None, n_jobs=-1, random_state=seed)
    clf.fit(sm.x, sm.y)
    tr_f1 = compute_metrics(sm.y, clf.predict(sm.x)).macro_f1
    val_m = compute_metrics(y[split.val], clf.predict(x[split.val]))
    print(f"    train macroF1 {tr_f1:.3f} | val {val_m.summary()}")

    _step(7, "Overfitting / underfitting detection")
    diag = diagnose([tr_f1] * 3, [val_m.macro_f1] * 3)
    print(f"    diagnosis: {diag.status.value} -- {diag.reason}")
    out["diagnosis"] = diag.to_dict()

    _step(8, "Apply correction technique and retrain")
    if diag.status is FitStatus.OVERFIT:
        technique = "depth limit + minimum leaf size + feature subsampling"
        clf2 = RandomForestClassifier(
            n_estimators=400, max_depth=12, min_samples_leaf=4, max_features="sqrt",
            n_jobs=-1, random_state=seed, class_weight="balanced_subsample",
        )
    elif diag.status is FitStatus.UNDERFIT:
        technique = "increased capacity (more and deeper trees)"
        clf2 = RandomForestClassifier(
            n_estimators=600, max_depth=None, min_samples_leaf=1, n_jobs=-1, random_state=seed
        )
    else:
        technique = "none required"
        clf2 = None

    print(f"    correction: {technique}")
    best = clf
    if clf2 is not None:
        clf2.fit(sm.x, sm.y)
        val2 = compute_metrics(y[split.val], clf2.predict(x[split.val]))
        print(f"    after correction: val {val2.summary()}")
        if val2.macro_f1 > val_m.macro_f1:
            best, val_m = clf2, val2
            print("    correction improved validation; keeping the corrected model")
        else:
            print("    correction did not improve validation; keeping the original")
    out["correction"] = {"technique": technique}
    out["val_metrics"] = val_m.to_dict()

    _step(9, "10-fold cross-validation on the training split")
    fold_scores = []
    for tr_idx, te_idx in kfold_indices(y[split.train], n_splits=10, seed=seed):
        xr, yr = x[split.train][tr_idx], y[split.train][tr_idx]
        sm_f = balance_training_split(xr, yr, seed=seed)
        m = RandomForestClassifier(n_estimators=150, n_jobs=-1, random_state=seed).fit(sm_f.x, sm_f.y)
        fold_scores.append(
            compute_metrics(y[split.train][te_idx], m.predict(x[split.train][te_idx])).macro_f1
        )
    print(f"    10-fold macro-F1: {np.mean(fold_scores):.3f} +/- {np.std(fold_scores):.3f}")
    out["cv"] = {"mean_macro_f1": float(np.mean(fold_scores)), "std": float(np.std(fold_scores)),
                 "folds": [float(s) for s in fold_scores]}

    _step(10, "Final test evaluation")
    test_m = compute_metrics(y[split.test], best.predict(x[split.test]))
    print(f"    TEST  {test_m.summary()}")
    print(f"    per-class F1: { {k: round(v,3) for k,v in test_m.per_class_f1.items()} }")
    out["test_metrics"] = test_m.to_dict()
    out["feature_importance"] = dict(
        sorted(zip(FEATURE_COLS, [float(v) for v in best.feature_importances_], strict=False),
               key=lambda kv: -kv[1])
    )
    return out


def run_progressive(
    data_root: Path,
    reports_dir: Path,
    model_names: list[str],
    start_tier: str = "T1_tiny",
    offline: bool = False,
    max_minutes_per_model: float = 25.0,
    gan_epochs: int = 30,
    budget_minutes: float = 45.0,
    device: str = "auto",
    verbose: bool = True,
    resume: bool = False,
) -> dict:
    """Walk the tier ladder, stopping when the machine can no longer take the next rung.

    With ``resume``, tiers already present in ``reports/results.json`` are carried forward
    instead of being recomputed, so a ladder can be continued after an interruption without
    throwing away the rungs that already finished.
    """
    log = cap.ScalingLog(machine=cap.machine_profile())
    reports_dir.mkdir(parents=True, exist_ok=True)
    results: list[PipelineResult] = []
    prior: list[dict] = []
    if resume and (reports_dir / "results.json").exists():
        prior = json.loads((reports_dir / "results.json").read_text()).get("tiers", [])
        prior = [t for t in prior if t["tier"] != start_tier]
        if prior and verbose:
            print(f"[resume] carrying forward tiers: {[t['tier'] for t in prior]}")
    tier = get_tier(start_tier)

    while True:
        watermark = cap.MemoryWatermark()
        res = run_image_pipeline(
            tier, data_root, model_names, reports_dir, offline=offline,
            max_minutes_per_model=max_minutes_per_model, gan_epochs=gan_epochs,
            device=device, verbose=verbose,
        )
        results.append(res)
        watermark.sample()

        # Persist after every tier: a run interrupted at 4am still leaves a complete table.
        (reports_dir / "results.json").write_text(
            json.dumps({"tiers": prior + [r.to_dict() for r in results]}, indent=2)
        )

        epoch_times = [
            e["seconds"] for m in res.models for e in m.initial.get("epochs", []) if m.initial
        ]
        measurement = cap.TierMeasurement(
            tier=tier.name,
            n_tiles=tier.n_tiles,
            image_size=tier.image_size,
            seconds_per_epoch=float(np.mean(epoch_times)) if epoch_times else 0.0,
            epochs_run=len(epoch_times),
            peak_rss_gb=watermark.peak,
            free_disk_gb_after=cap.free_disk_gb(),
            total_seconds=res.seconds,
            ok=all(m.ok for m in res.models),
            note="" if all(m.ok for m in res.models) else "one or more models failed",
        )
        log.measurements.append(measurement)
        log.final_tier = tier.name

        allowed, reason = cap.can_advance(measurement, tier, budget_minutes=budget_minutes)
        print(f"\n[capacity] {reason}")
        log.stopped_because = reason
        log.to_json(reports_dir / "scaling_log.json")

        if not allowed:
            break
        nxt = next_tier(tier.name)
        if nxt is None:
            break
        tier = nxt

    return {"tiers": prior + [r.to_dict() for r in results], "scaling": json.loads(
        (reports_dir / "scaling_log.json").read_text())}
