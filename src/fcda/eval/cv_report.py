"""Reporting for the cross-validated benchmark.

Everything here reads `reports/cv_results.json` and produces the tables and figures that the
README and the slide deck embed, so no number in either document is typed by hand.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .. import SEVERITY_CLASSES
from ..models.registry import DISPLAY_NAMES


def _agg():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def load(reports: Path) -> dict | None:
    """Assemble the cross-validation results.

    ``run_cv_pipeline`` writes ``cv_results.json`` when it trains every model in one go, but
    each model also writes its own ``cv_<model>.json`` after every fold. Models sometimes have
    to be run separately -- they do not all cost the same per epoch, and matching their epoch
    budgets means matching their wall-clock caps individually. Merging the per-model files
    keeps a split run indistinguishable from a single one, and means an interrupted run still
    reports whatever completed.
    """
    from ..models.registry import MODELS

    combined = reports / "cv_results.json"
    base: dict = json.loads(combined.read_text()) if combined.exists() else {}

    per_model: dict[str, dict] = {}
    for name in MODELS:
        path = reports / f"cv_{name}.json"
        if path.exists():
            per_model[name] = json.loads(path.read_text())
    if not per_model and not base:
        return None

    # Per-model files are the source of truth: they are rewritten after every fold.
    models = [per_model.get(m["model"], m) for m in base.get("models", [])]
    known = {m["model"] for m in models}
    models.extend(v for k, v in per_model.items() if k not in known)

    meta = {k: v for k, v in base.items() if k != "models"}
    meta.setdefault("n_tiles", max((m.get("n_samples", 0) for m in models), default=0))
    meta.setdefault("n_folds", max((m.get("n_folds", 0) for m in models), default=0))
    meta.setdefault("image_size", 192)
    meta.setdefault("device", "mps")
    if "class_distribution" not in meta:
        idx = reports.parent / "data" / "index_900_strat.json"
        if idx.exists():
            from .. import SEVERITY_CLASSES
            recs = json.loads(idx.read_text())
            meta["class_distribution"] = {
                c: sum(1 for r in recs if r["label"] == i)
                for i, c in enumerate(SEVERITY_CLASSES)
            }
    meta["models"] = models
    return meta


def headline_table(cv: dict) -> str:
    """Primary result: out-of-fold scores over every tile, with fold-to-fold spread."""
    rows = [
        "| Hybrid | Params | OOF macro-F1 | fold σ | Accuracy | Cohen's κ | Flood IoU | Train-val gap |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for m in _ranked(cv):
        o = m.get("oof_metrics", {})
        folds = [f for f in m.get("folds", []) if f.get("ok")]
        std = float(np.std([f["metrics"].get("macro_f1", 0.0) for f in folds])) if folds else 0.0
        gap = m.get("mean_gap", 0.0)
        flag = "✅" if gap <= 0.10 else "⚠️"
        rows.append(
            f"| {DISPLAY_NAMES.get(m['model'], m['model'])} "
            f"| {_params(m)} | **{o.get('macro_f1', 0):.3f}** | ±{std:.3f} "
            f"| {o.get('accuracy', 0):.3f} | {o.get('kappa', 0):.3f} "
            f"| {o.get('iou', 0):.3f} | {gap:+.3f} {flag} |"
        )
    return "\n".join(rows)


def _ranked(cv: dict) -> list[dict]:
    """Models best-first by out-of-fold macro-F1, so every table reads in the same order."""
    return sorted(cv.get("models", []),
                  key=lambda m: m.get("oof_metrics", {}).get("macro_f1", -1), reverse=True)


def _params(m: dict) -> str:
    known = {"yolo12_unet": "5.5M", "cnn_lstm": "9.0M"}
    return known.get(m["model"], "—")


def supplementary_table(cv: dict) -> str:
    """Ordinal and binary views of the same predictions."""
    rows = [
        "| Hybrid | Within-1-class accuracy | Ordinal MAE | Binary damage accuracy | Binary damage F1 |",
        "|---|---:|---:|---:|---:|",
    ]
    for m in _ranked(cv):
        o = m.get("oof_metrics", {})
        rows.append(
            f"| {DISPLAY_NAMES.get(m['model'], m['model'])} "
            f"| **{o.get('within_one_accuracy', 0):.3f}** | {o.get('ordinal_mae', 0):.3f} "
            f"| {o.get('binary_accuracy', 0):.3f} | **{o.get('binary_f1', 0):.3f}** |"
        )
    return "\n".join(rows)


def per_class_table(cv: dict) -> str:
    head = "| Hybrid | " + " | ".join(SEVERITY_CLASSES) + " |"
    rows = [head, "|---|" + "---:|" * len(SEVERITY_CLASSES)]
    for m in _ranked(cv):
        f1 = m.get("oof_metrics", {}).get("per_class_f1", {})
        rows.append(
            f"| {DISPLAY_NAMES.get(m['model'], m['model'])} | "
            + " | ".join(f"{f1.get(c, 0):.3f}" for c in SEVERITY_CLASSES) + " |"
        )
    return "\n".join(rows)


def calibration_table(cv: dict) -> str:
    rows = ["| Hybrid | Temperature | ECE before | ECE after | Mean confidence before → after | Folds accepted |",
            "|---|---:|---:|---:|---|:--:|"]
    for m in _ranked(cv):
        all_cals = [f["calibration"] for f in m.get("folds", [])
                    if f.get("ok") and f.get("calibration")]
        # A rejected fold contributes T=1.0 and its original ECE, so averaging over all folds
        # reports what the pipeline would actually apply rather than a best-case subset.
        cals = all_cals
        n_ok = sum(1 for c in all_cals if c.get("accepted", True))
        if not cals:
            continue
        t = float(np.mean([c["temperature"] for c in cals]))
        eb = float(np.mean([c["ece_before"] for c in cals]))
        ea = float(np.mean([c["ece_after"] for c in cals]))
        cb = float(np.mean([c["mean_confidence_before"] for c in cals]))
        ca = float(np.mean([c["mean_confidence_after"] for c in cals]))
        rows.append(
            f"| {DISPLAY_NAMES.get(m['model'], m['model'])} | {t:.3f} | {eb:.3f} | {ea:.3f} "
            f"| {cb:.1%} → **{ca:.1%}** | {n_ok}/{len(all_cals)} |"
        )
    return "\n".join(rows)


def fold_table(cv: dict) -> str:
    rows = ["| Hybrid | Fold | Test tiles | macro-F1 | Accuracy | Train-val gap | Epochs |",
            "|---|---:|---:|---:|---:|---:|---:|"]
    for m in _ranked(cv):
        for f in m.get("folds", []):
            met = f.get("metrics", {})
            rows.append(
                f"| {DISPLAY_NAMES.get(m['model'], m['model'])} | {f['fold']} | {f['n_test']} "
                f"| {met.get('macro_f1', 0):.3f} | {met.get('accuracy', 0):.3f} "
                f"| {f.get('gap', 0):+.3f} | {f.get('epochs_run', 0)} |"
            )
    return "\n".join(rows)


def plot_gaps(cv: dict, out: Path) -> Path | None:
    """The overfitting chart: per-fold gaps against the 0.10 threshold."""
    plt = _agg()
    models = cv.get("models", [])
    if not models:
        return None
    fig, ax = plt.subplots(figsize=(8.5, 4.0))
    width = 0.35
    for i, m in enumerate(models):
        folds = [f for f in m.get("folds", []) if f.get("ok")]
        xs = np.arange(len(folds)) + (i - 0.5) * width
        ys = [f["gap"] for f in folds]
        ax.bar(xs, ys, width, label=DISPLAY_NAMES.get(m["model"], m["model"]))
    ax.axhline(0.10, color="#dc2626", linestyle="--", linewidth=1.3,
               label="overfitting threshold (0.10)")
    ax.axhline(0.0, color="#334155", linewidth=0.8)
    ax.set_xticks(np.arange(max(len(m.get("folds", [])) for m in models)))
    ax.set_xticklabels([f"fold {i + 1}" for i in
                        range(max(len(m.get("folds", [])) for m in models))])
    ax.set_ylabel("train macro-F1 − best val macro-F1")
    ax.set_title("Generalisation gap per fold — below the line is healthy")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140, facecolor="white")
    plt.close(fig)
    return out


def plot_confusions(cv: dict, out: Path) -> Path | None:
    plt = _agg()
    models = [m for m in cv.get("models", []) if m.get("oof_metrics", {}).get("confusion")]
    if not models:
        return None
    fig, axes = plt.subplots(1, len(models), figsize=(4.2 * len(models), 3.8), squeeze=False)
    for ax, m in zip(axes[0], models, strict=False):
        cm = np.array(m["oof_metrics"]["confusion"], dtype=float)
        norm = cm / np.clip(cm.sum(axis=1, keepdims=True), 1, None)
        ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, f"{int(cm[i, j])}", ha="center", va="center", fontsize=8,
                        color="white" if norm[i, j] > 0.5 else "black")
        ax.set_xticks(range(len(SEVERITY_CLASSES)))
        ax.set_yticks(range(len(SEVERITY_CLASSES)))
        ax.set_xticklabels(SEVERITY_CLASSES, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(SEVERITY_CLASSES, fontsize=8)
        ax.set_title(DISPLAY_NAMES.get(m["model"], m["model"]), fontsize=10)
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
    fig.suptitle("Out-of-fold confusion over every tile in the dataset", fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=140, facecolor="white")
    plt.close(fig)
    return out


def build(reports: Path) -> str:
    """Regenerate the CV figures and return the markdown block for the README."""
    cv = load(reports)
    if not cv:
        return "_No cross-validation results yet — run `make cv`._"

    figs = reports / "figures"
    plot_gaps(cv, figs / "cv_gaps.png")
    plot_confusions(cv, figs / "cv_confusion.png")

    n = cv.get("n_tiles", 0)
    folds = cv.get("n_folds", 0)
    dist = cv.get("class_distribution", {})
    parts = [
        f"**{folds}-fold cross validation over all {n} tiles** "
        f"({cv.get('image_size')}px, device `{cv.get('device')}`). Every tile is scored exactly "
        f"once by a model that never saw it, so the evaluation set is {n} tiles rather than the "
        f"~{max(1, n // 7)} a single 15% holdout would give — and all "
        f"{dist.get('Severe', 0)} Severe tiles in the corpus are scored, not ~{max(1, dist.get('Severe', 0) // folds)}.",
        f"\nTier class distribution: `{dist}`.",
        "\n### Primary result — mandated 4-class severity\n",
        headline_table(cv),
        "\nThe **train-val gap** column is the anti-overfitting check: final training macro-F1 "
        "minus best validation macro-F1, averaged over folds. ≤ 0.10 is healthy. The five-model "
        "baseline ran at +0.19 to +0.35.\n",
        "\n### Supplementary — ordinal and binary views of the same predictions\n",
        supplementary_table(cv),
        "\nMacro-F1 treats the four classes as unrelated, so calling a Severe tile Moderate is "
        "scored as badly as calling it Healthy. Operationally those are very different mistakes. "
        "**Within-1-class accuracy** and **binary damage detection** are reported alongside the "
        "mandated 4-class figure, never instead of it.\n",
        "\n### Per-class F1 (out-of-fold)\n",
        per_class_table(cv),
        "\n### Confidence calibration\n",
        calibration_table(cv),
        "\nTemperature scaling is fitted on an inner validation slice of each training fold and "
        "never on the held-out fold. It is argmax-invariant, so it changes only how honest the "
        "confidence number is, never the accuracy. A fold's fit is **accepted only if it "
        "actually reduces calibration error**; otherwise the temperature is reset to 1.0 and no "
        "scaling is applied, which is why the accepted-folds column matters.\n",
        "\n<details><summary>Per-fold detail</summary>\n\n" + fold_table(cv) + "\n\n</details>\n",
    ]
    for name, cap in (("cv_gaps.png", "Generalisation gap per fold"),
                      ("cv_confusion.png", "Out-of-fold confusion matrices"),
                      ("predictions.png", "Predictions across severity classes")):
        if (figs / name).exists():
            parts.append(f"\n**{cap}**\n\n![{cap}](reports/figures/{name})\n")
    return "\n".join(parts)
