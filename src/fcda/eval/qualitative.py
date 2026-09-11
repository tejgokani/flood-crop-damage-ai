"""Qualitative prediction panels.

A metrics table says a model works; a picture shows it. This renders, for a handful of test
tiles spanning all four severity classes, the pre-flood image, the post-flood image, the
reference mask and the model's prediction side by side.

It is also the fastest way to catch a failure a metric can hide -- a model that has learned
to predict "water everywhere" can post a respectable Dice score while being obviously wrong
to the eye.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from .. import SEVERITY_CLASSES


def _agg(name: str):
    import matplotlib
    matplotlib.use("Agg")
    return __import__("matplotlib.pyplot", fromlist=["pyplot"])


def render_predictions(
    checkpoint: Path,
    model_name: str,
    data_root: Path,
    out: Path,
    n_per_class: int = 1,
    size: int = 256,
) -> Path | None:
    """Render a prediction panel for the given checkpoint."""
    plt = _agg("agg")
    from ..data.etci import _stack_sar, load_planes
    from ..data.severity import severity_from_masks
    from ..models.registry import DISPLAY_NAMES, build_model, is_temporal, wants_change_input
    from ..preprocess.transforms import build_transform

    idx_files = sorted(data_root.glob("index_*.json"))
    if not idx_files or not checkpoint.exists():
        return None
    records = json.loads(idx_files[-1].read_text())

    # Pick the most representative tile of each class: the one nearest its class centroid
    # in flood fraction, so the panel is not all extreme cases.
    chosen: list[dict] = []
    for c in range(len(SEVERITY_CLASSES)):
        members = [r for r in records if r["label"] == c]
        if not members:
            continue
        target = float(np.median([r["net_flood_fraction"] for r in members]))
        members.sort(key=lambda r: abs(r["net_flood_fraction"] - target))
        chosen.extend(members[:n_per_class])
    if not chosen:
        return None

    model = build_model(model_name, pretrained=False)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    model.eval()
    tf = build_transform(train=False, use_clahe=True)
    temporal = is_temporal(model_name)
    change = wants_change_input(model_name)

    rows = len(chosen)
    fig, axes = plt.subplots(rows, 4, figsize=(13, 3.05 * rows), squeeze=False)

    for r, rec in enumerate(chosen):
        p = load_planes(data_root, rec["tile_id"])
        post = tf(_stack_sar(p["post_vv"], p["post_vh"], size))
        pre = tf(_stack_sar(p["pre_vv"], p["pre_vh"], size))
        x = torch.from_numpy(post).float()[None]
        if temporal:
            x = torch.stack([torch.from_numpy(pre).float(), x[0]], dim=0)[None]
        elif change:
            x = torch.from_numpy(np.concatenate([post, post - pre], axis=0)).float()[None]
        with torch.no_grad():
            seg, cls = model(x)
        prob = torch.sigmoid(seg)[0, 0].numpy()
        probs = torch.softmax(cls, 1)[0].numpy()
        pred_cls = SEVERITY_CLASSES[int(probs.argmax())]
        truth = severity_from_masks(p["post_flood"], p["post_water_body"])

        import cv2

        ref = cv2.resize(
            ((p["post_flood"] > 0) & ~(p["post_water_body"] > 0)).astype(np.uint8),
            (size, size), interpolation=cv2.INTER_NEAREST,
        )

        pred_frac = float((prob >= 0.5).mean())
        panels = [
            (p["pre_vv"], "Pre-flood VV (2017-03-14)", "gray", None),
            (p["post_vv"], "Post-flood VV (2017-06-06)", "gray", None),
            (ref, f"Reference: {truth.name} ({truth.net_flood_fraction:.0%})", "Blues", (0, 1)),
            (
                prob,
                f"Predicted: {pred_cls} ({probs.max():.0%} conf) — mask {pred_frac:.0%}",
                "Blues",
                (0, 1),
            ),
        ]
        for c, (img, title, cmap, lim) in enumerate(panels):
            ax = axes[r][c]
            # The probability panel is pinned to a true 0-1 scale. Letting matplotlib
            # autoscale would render a timid 0.1-0.3 probability map as saturated blue and
            # make an uncertain model look confident.
            if lim is None:
                ax.imshow(img, cmap=cmap)
            else:
                ax.imshow(img, cmap=cmap, vmin=lim[0], vmax=lim[1])
            ax.set_title(title, fontsize=9)
            ax.axis("off")
        axes[r][0].text(
            -0.08, 0.5, truth.name, transform=axes[r][0].transAxes, rotation=90,
            va="center", ha="center", fontsize=11, fontweight="bold",
        )

    fig.suptitle(
        f"{DISPLAY_NAMES.get(model_name, model_name)} — test predictions across severity classes\n"
        "Right column is the raw flood probability on a fixed 0–1 scale (not autoscaled)",
        fontsize=12,
    )
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140, facecolor="white")
    plt.close(fig)
    return out


def render_best(reports: Path, data_root: Path, checkpoints: Path) -> Path | None:
    """Render the panel for whichever model scored best out-of-fold.

    Prefers the cross-validated results and their fold checkpoints; falls back to the
    single-split run when cross validation has not been run.
    """
    from .cv_report import load as load_cv

    cv = load_cv(reports)
    if cv and cv.get("models"):
        ranked = sorted(cv["models"],
                        key=lambda m: m.get("oof_metrics", {}).get("macro_f1", -1), reverse=True)
        for m in ranked:
            # Use the fold whose own macro-F1 was closest to the out-of-fold average, so the
            # panel shows a typical model rather than the luckiest one.
            folds = [f for f in m.get("folds", []) if f.get("ok")]
            if not folds:
                continue
            target = m.get("oof_metrics", {}).get("macro_f1", 0.0)
            best = min(folds, key=lambda f: abs(f["metrics"].get("macro_f1", 0.0) - target))
            ck = checkpoints / f"{m['model']}_fold{best['fold']}.pt"
            if ck.exists():
                return render_predictions(
                    ck, m["model"], data_root, reports / "figures" / "predictions.png",
                    size=cv.get("image_size", 192),
                )

    results_path = reports / "results.json"
    if not results_path.exists():
        return None
    results = json.loads(results_path.read_text())
    tiers = results.get("tiers", [])
    if not tiers:
        return None
    tier = tiers[-1]
    ranked = sorted(
        tier["models"], key=lambda m: (m.get("final_test") or {}).get("macro_f1", -1), reverse=True
    )
    for m in ranked:
        ck = checkpoints / f"{m['name']}_{tier['tier']}.pt"
        if ck.exists():
            return render_predictions(
                ck, m["name"], data_root, reports / "figures" / "predictions.png",
                size=tier["image_size"],
            )
    return None
