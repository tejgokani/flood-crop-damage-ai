"""Turn reports/results.json into the figures and tables used in the README and slides."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .. import SEVERITY_CLASSES
from ..models.registry import DISPLAY_NAMES


def _use_agg():
    import matplotlib
    matplotlib.use("Agg")


def comparison_table(results: dict) -> str:
    """Markdown table of every model at the deepest tier that completed."""
    tiers = results.get("tiers", [])
    if not tiers:
        return "_No results yet._"
    tier = tiers[-1]
    rows = [
        "| Hybrid architecture | Params | Test macro-F1 | Accuracy | Cohen's κ | Flood IoU | Dice | Diagnosis | Correction helped |",
        "|---|---:|---:|---:|---:|---:|---:|---|:--:|",
    ]
    ranked = sorted(
        tier["models"],
        key=lambda m: (m.get("final_test") or {}).get("macro_f1", -1),
        reverse=True,
    )
    for m in ranked:
        t = m.get("final_test") or {}
        init = m.get("initial") or {}
        re_ = m.get("retrained") or {}
        helped = "—"
        if re_:
            helped = "yes" if re_.get("best_val_f1", 0) > init.get("best_val_f1", 0) else "no"
        rows.append(
            f"| {DISPLAY_NAMES.get(m['name'], m['name'])} "
            f"| {init.get('n_params', 0) / 1e6:.1f}M "
            f"| **{t.get('macro_f1', 0):.3f}** | {t.get('accuracy', 0):.3f} "
            f"| {t.get('kappa', 0):.3f} | {t.get('iou', 0):.3f} | {t.get('dice', 0):.3f} "
            f"| {(m.get('diagnosis') or {}).get('status', '—')} | {helped} |"
        )
    return "\n".join(rows)


def per_class_table(results: dict) -> str:
    tiers = results.get("tiers", [])
    if not tiers:
        return ""
    tier = tiers[-1]
    head = "| Hybrid | " + " | ".join(SEVERITY_CLASSES) + " |"
    sep = "|---|" + "---:|" * len(SEVERITY_CLASSES)
    rows = [head, sep]
    for m in tier["models"]:
        t = (m.get("final_test") or {}).get("per_class_f1", {})
        rows.append(
            f"| {DISPLAY_NAMES.get(m['name'], m['name'])} | "
            + " | ".join(f"{t.get(c, 0):.3f}" for c in SEVERITY_CLASSES)
            + " |"
        )
    return "\n".join(rows)


def plot_learning_curves(results: dict, out: Path) -> Path | None:
    _use_agg()
    import matplotlib.pyplot as plt

    tiers = results.get("tiers", [])
    if not tiers:
        return None
    tier = tiers[-1]
    models = [m for m in tier["models"] if m.get("initial", {}).get("epochs")]
    if not models:
        return None

    n = len(models)
    fig, axes = plt.subplots(1, n, figsize=(3.6 * n, 3.4), squeeze=False)
    for ax, m in zip(axes[0], models, strict=False):
        eps = m["initial"]["epochs"]
        x = [e["epoch"] for e in eps]
        ax.plot(x, [e["train_f1"] for e in eps], marker="o", ms=3, label="train")
        ax.plot(x, [e["val_f1"] for e in eps], marker="s", ms=3, label="val")
        if m.get("retrained", {}) and m["retrained"].get("epochs"):
            r = m["retrained"]["epochs"]
            ax.plot([e["epoch"] for e in r], [e["val_f1"] for e in r],
                    linestyle="--", label="val (corrected)")
        ax.set_title(DISPLAY_NAMES.get(m["name"], m["name"]), fontsize=9)
        ax.set_xlabel("epoch")
        ax.set_ylabel("macro-F1")
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.suptitle(f"Learning curves — tier {tier['tier']}", fontsize=11)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def plot_confusions(results: dict, out: Path) -> Path | None:
    _use_agg()
    import matplotlib.pyplot as plt

    tiers = results.get("tiers", [])
    if not tiers:
        return None
    tier = tiers[-1]
    models = [m for m in tier["models"] if (m.get("final_test") or {}).get("confusion")]
    if not models:
        return None

    n = len(models)
    fig, axes = plt.subplots(1, n, figsize=(3.1 * n, 3.2), squeeze=False)
    for ax, m in zip(axes[0], models, strict=False):
        cm = np.array(m["final_test"]["confusion"], dtype=float)
        norm = cm / np.clip(cm.sum(axis=1, keepdims=True), 1, None)
        ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, f"{int(cm[i, j])}", ha="center", va="center",
                        fontsize=7, color="white" if norm[i, j] > 0.5 else "black")
        ax.set_xticks(range(len(SEVERITY_CLASSES)))
        ax.set_yticks(range(len(SEVERITY_CLASSES)))
        ax.set_xticklabels(SEVERITY_CLASSES, rotation=45, ha="right", fontsize=7)
        ax.set_yticklabels(SEVERITY_CLASSES, fontsize=7)
        ax.set_title(DISPLAY_NAMES.get(m["name"], m["name"]), fontsize=9)
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
    fig.suptitle(f"Confusion matrices — tier {tier['tier']}", fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def plot_scaling(scaling: dict, out: Path) -> Path | None:
    _use_agg()
    import matplotlib.pyplot as plt

    ms = scaling.get("measurements", [])
    if not ms:
        return None
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))
    names = [m["tier"] for m in ms]
    axes[0].bar(names, [m["n_tiles"] for m in ms], color="#2b6cb0")
    axes[0].set_title("Tiles per tier")
    axes[1].bar(names, [m["peak_rss_gb"] for m in ms], color="#2f855a")
    axes[1].set_title("Peak RSS (GB)")
    axes[2].bar(names, [m["seconds_per_epoch"] for m in ms], color="#b7791f")
    axes[2].set_title("Seconds per epoch")
    for ax in axes:
        ax.tick_params(axis="x", rotation=30, labelsize=8)
        ax.grid(alpha=0.3, axis="y")
    fig.suptitle("Progressive dataset scaling — what the machine actually sustained", fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def results_section(results: dict, scaling: dict | None, tabular: dict | None) -> str:
    """The block injected between the RESULTS markers in the README."""
    tiers = results.get("tiers", [])
    if not tiers:
        return "_No results yet._"
    tier = tiers[-1]
    lines: list[str] = []

    reached = [t["tier"] for t in tiers]
    lines.append(
        f"Deepest tier completed: **{tier['tier']}** "
        f"({tier['n_tiles']} tiles at {tier['image_size']}px, device `{tier['device']}`). "
        f"Tiers walked: {' → '.join(reached)}."
    )
    if scaling and scaling.get("stopped_because"):
        lines.append(f"\nScaling stopped because: _{scaling['stopped_because']}_")

    lines.append("\n### Image pipeline — five hybrids, one protocol\n")
    lines.append(comparison_table(results))
    lines.append("\n**Per-class F1 (test)**\n")
    lines.append(per_class_table(results))
    lines.append(
        f"\nTier class distribution: `{tier.get('class_distribution', {})}` "
        "(tiers cap Healthy at 45%; the natural prior in the full pool is 86.9% Healthy / "
        "0.98% Severe)."
    )

    gan = tier.get("gan") or {}
    if gan:
        lines.append(
            f"\n**GAN augmentation** (training split only): {gan.get('n_generated', 0)} synthetic "
            f"tiles from {gan.get('n_real_used', 0)} real ones over {gan.get('epochs', 0)} epochs "
            f"in {gan.get('seconds', 0):.0f}s."
        )

    leak = tier.get("leakage") or {}
    if leak:
        lines.append(
            f"\n**Leakage audit (image)**: {'clean' if leak.get('clean') else 'findings present'} — "
            f"{leak.get('n_findings', 0)} finding(s)."
        )

    if tabular:
        tm = tabular.get("test_metrics", {})
        cv = tabular.get("cv", {})
        sm = tabular.get("smote", {})
        lines.append("\n### Tabular pipeline — Indian district crop statistics\n")
        lines.append(
            f"| Metric | Value |\n|---|---:|\n"
            f"| Test macro-F1 | **{tm.get('macro_f1', 0):.3f}** |\n"
            f"| Test accuracy | {tm.get('accuracy', 0):.3f} |\n"
            f"| Cohen's κ | {tm.get('kappa', 0):.3f} |\n"
            f"| 10-fold CV macro-F1 | {cv.get('mean_macro_f1', 0):.3f} ± {cv.get('std', 0):.3f} |\n"
            f"| SMOTE | {sm.get('before', {})} → balanced (+{sm.get('n_synthetic', 0)} rows) |\n"
            f"| Target leakage caught | `{', '.join(tabular.get('dropped_columns', []))}` |"
        )

    figs = []
    for name, cap in (("learning_curves.png", "Learning curves"),
                      ("confusion_matrices.png", "Confusion matrices"),
                      ("scaling.png", "Progressive scaling")):
        if (Path("reports/figures") / name).exists():
            figs.append(f"**{cap}**\n\n![{cap}](reports/figures/{name})")
    if figs:
        lines.append("\n" + "\n\n".join(figs))

    return "\n".join(lines)


def inject_into_readme(reports: Path, readme: Path = Path("README.md")) -> None:
    """Replace the RESULTS block in the README with freshly generated content."""
    if not readme.exists():
        return
    results = json.loads((reports / "results.json").read_text())
    scaling = None
    if (reports / "scaling_log.json").exists():
        scaling = json.loads((reports / "scaling_log.json").read_text())
    tabular = None
    if (reports / "tabular.json").exists():
        tabular = json.loads((reports / "tabular.json").read_text())

    body = results_section(results, scaling, tabular)
    text = readme.read_text()
    start, end = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"
    if start in text and end in text:
        head = text.split(start)[0]
        tail = text.split(end)[1]
        readme.write_text(f"{head}{start}\n{body}\n{end}{tail}")
        print(f"README results section updated ({len(body)} chars)")


def build_all(reports: Path) -> dict:
    """Regenerate every figure and table that the README and slides consume."""
    results_path = reports / "results.json"
    if not results_path.exists():
        print(f"No {results_path}; run `fcda train` first.")
        return {}
    results = json.loads(results_path.read_text())
    figs = reports / "figures"
    made = []
    for fn, name in (
        (plot_learning_curves, "learning_curves.png"),
        (plot_confusions, "confusion_matrices.png"),
    ):
        p = fn(results, figs / name)
        if p:
            made.append(str(p))
    scaling_path = reports / "scaling_log.json"
    if scaling_path.exists():
        p = plot_scaling(json.loads(scaling_path.read_text()), figs / "scaling.png")
        if p:
            made.append(str(p))

    inject_into_readme(reports)
    table = comparison_table(results)
    (reports / "comparison.md").write_text(
        "# Model comparison\n\n" + table + "\n\n## Per-class F1\n\n" + per_class_table(results) + "\n"
    )
    print(table)
    print(f"\nFigures: {made}")
    return {"figures": made, "table": table}
