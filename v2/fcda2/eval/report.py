"""v2 reporting: did we hit macro-F1 >= 0.70, and is the answer trustworthy?"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from fcda import SEVERITY_CLASSES
from fcda.models.registry import DISPLAY_NAMES

TARGET = 0.70
#: What a model that always predicts the majority class scores, for context on every table.
ALWAYS_HEALTHY_ACC = 0.703
ALWAYS_HEALTHY_MACRO_F1 = 0.206


def load_runs(reports: Path) -> list[dict]:
    out = []
    for p in sorted(reports.glob("run_*.json")):
        try:
            out.append(json.loads(p.read_text()))
        except json.JSONDecodeError:
            continue
    return out


def headline(runs: list[dict]) -> str:
    rows = ["| Model | Synthetic | macro-F1 | fold σ | Accuracy | κ | Within-1 | Severe F1 | IoU | Gap | Target |",
            "|---|:--:|---:|---:|---:|---:|---:|---:|---:|---:|:--:|"]
    for r in sorted(runs, key=lambda r: -r.get("oof", {}).get("macro_f1", 0)):
        o = r.get("oof", {})
        folds = [f for f in r.get("folds", []) if f.get("ok")]
        std = float(np.std([f["metrics"].get("macro_f1", 0) for f in folds])) if folds else 0.0
        f1 = o.get("macro_f1", 0)
        rows.append(
            f"| {DISPLAY_NAMES.get(r['model'], r['model'])} "
            f"| {'yes' if r.get('use_synthetic') else 'no'} "
            f"| **{f1:.3f}** | ±{std:.3f} | {o.get('accuracy', 0):.3f} | {o.get('kappa', 0):.3f} "
            f"| {o.get('within_one_accuracy', 0):.3f} "
            f"| {o.get('per_class_f1', {}).get('Severe', 0):.3f} | {o.get('iou', 0):.3f} "
            f"| {r.get('mean_gap', 0):+.3f} | {'✅' if f1 >= TARGET else '—'} |"
        )
    rows.append(
        f"| _always-Healthy baseline_ | – | _{ALWAYS_HEALTHY_MACRO_F1:.3f}_ | – "
        f"| _{ALWAYS_HEALTHY_ACC:.3f}_ | _0.000_ | – | _0.000_ | – | – | – |"
    )
    return "\n".join(rows)


def per_class(runs: list[dict]) -> str:
    rows = ["| Model | Synthetic | " + " | ".join(SEVERITY_CLASSES) + " |",
            "|---|:--:|" + "---:|" * len(SEVERITY_CLASSES)]
    for r in sorted(runs, key=lambda r: -r.get("oof", {}).get("macro_f1", 0)):
        f1 = r.get("oof", {}).get("per_class_f1", {})
        rows.append(
            f"| {DISPLAY_NAMES.get(r['model'], r['model'])} "
            f"| {'yes' if r.get('use_synthetic') else 'no'} | "
            + " | ".join(f"{f1.get(c, 0):.3f}" for c in SEVERITY_CLASSES) + " |"
        )
    return "\n".join(rows)


def synthetic_effect(runs: list[dict]) -> str:
    """Paired comparison wherever a model was run both with and without synthetic data."""
    by_model: dict[str, dict[bool, dict]] = {}
    for r in runs:
        by_model.setdefault(r["model"], {})[bool(r.get("use_synthetic"))] = r
    pairs = {m: v for m, v in by_model.items() if len(v) == 2}
    if not pairs:
        return "_No paired with/without-synthetic runs yet._"

    rows = ["| Model | Real only | + synthetic | Δ macro-F1 | Δ Severe F1 | Δ gap |",
            "|---|---:|---:|---:|---:|---:|"]
    for m, v in pairs.items():
        a, b = v[False].get("oof", {}), v[True].get("oof", {})
        da = b.get("macro_f1", 0) - a.get("macro_f1", 0)
        ds = (b.get("per_class_f1", {}).get("Severe", 0)
              - a.get("per_class_f1", {}).get("Severe", 0))
        dg = v[True].get("mean_gap", 0) - v[False].get("mean_gap", 0)
        rows.append(
            f"| {DISPLAY_NAMES.get(m, m)} | {a.get('macro_f1', 0):.3f} | {b.get('macro_f1', 0):.3f} "
            f"| **{da:+.3f}** | {ds:+.3f} | {dg:+.3f} |")
    return "\n".join(rows)


def build(reports: Path) -> str:
    runs = load_runs(reports)
    if not runs:
        return "_No v2 runs yet — `python run.py pilot` or `python run.py full`._"

    best = max(r.get("oof", {}).get("macro_f1", 0) for r in runs)
    status = (f"**Target met** — best macro-F1 {best:.3f} ≥ {TARGET}."
              if best >= TARGET else
              f"**Target not met** — best macro-F1 {best:.3f}, target {TARGET}. "
              f"Reported as-is rather than adjusted.")

    n = runs[0].get("oof", {}).get("n_scored", 0)
    parts = [
        status,
        f"\nEvaluated on **{n} real tiles**, each scored once out of fold. Synthetic tiles are "
        "training-only and are never scored.\n",
        "### Results\n", headline(runs),
        "\n### Per-class F1 (out-of-fold, real tiles)\n", per_class(runs),
        "\n### Does synthetic data help?\n", synthetic_effect(runs),
        "\n> For context, v1's conditional DCGAN measured **−0.101 macro-F1**. v2 uses procedural "
        "generation calibrated to the real corpus (flooded VV 50.1 ± 58.4 against dry 180.4 ± 65.4, "
        "measured on 150 real tiles) with terrain-driven flood shapes and exact class targeting.\n",
    ]
    return "\n".join(parts)
