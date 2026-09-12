"""Streamlit demo: satellite tile in, crop damage assessment out.

Run with:  streamlit run app/streamlit_app.py

Deliberately runnable with no trained checkpoint: without one it still demonstrates the
preprocessing, the severity rule and the fusion arithmetic, which is most of what a reviewer
wants to interrogate. With a checkpoint it runs the real model.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fcda import SEVERITY_CLASSES  # noqa: E402
from fcda.data.severity import (  # noqa: E402
    DEFAULT_THRESHOLDS,
    THRESHOLD_RATIONALE,
    severity_from_masks,
)
from fcda.models.registry import DISPLAY_NAMES, MODELS  # noqa: E402

st.set_page_config(page_title="Crop Flood Damage Assessment", page_icon="🌊", layout="wide")

SEVERITY_COLOUR = {
    "Healthy": "#16a34a",
    "Mild": "#ca8a04",
    "Moderate": "#ea580c",
    "Severe": "#dc2626",
}


@st.cache_data(show_spinner=False)
def load_tabular():
    from fcda.data.india_agri import build_tabular_dataset

    csv = ROOT / "data" / "india_crops.csv"
    if not csv.exists():
        return None
    return build_tabular_dataset(csv)


@st.cache_resource(show_spinner=False)
def load_model(name: str, ckpt: Path):
    import torch

    from fcda.models.registry import build_model

    model = build_model(name, pretrained=False)
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    model.eval()
    return model


@st.cache_data(show_spinner=False)
def load_temperature(name: str) -> float:
    """Mean calibration temperature fitted across the cross-validation folds.

    The confidence shown below is the calibrated one. An uncalibrated softmax is a score that
    happens to sum to one, not a probability; temperature scaling is argmax-invariant, so this
    changes only how honest the number is, never which class is predicted.
    """
    import numpy as np

    from fcda.eval.cv_report import load as load_cv

    # Use the same loader the report uses: it merges the per-model cv_<model>.json files, which
    # is where the temperatures actually live when the models were run separately.
    cv = load_cv(ROOT / "reports")
    if not cv:
        return 1.0
    for m in cv.get("models", []):
        if m["model"] == name:
            temps = [f["calibration"]["temperature"] for f in m.get("folds", [])
                     if f.get("ok") and f.get("calibration")]
            if temps:
                return float(np.mean(temps))
    return 1.0


def find_checkpoints() -> dict[str, Path]:
    """Best available checkpoint per architecture.

    Ranked by the tier it was trained at, then preferring the corrected run. Plain
    alphabetical order would pick `T1_tiny` over `T2_small` and silently demo the weakest
    model in the directory.
    """
    from fcda.data.tiers import TIERS_BY_NAME, tier_index

    out: dict[str, Path] = {}
    ck = ROOT / "checkpoints"
    if not ck.exists():
        return out

    # Which run the pipeline actually selected per (model, tier). The corrected run is kept
    # only when it improved validation, which it usually does not -- so preferring
    # "_corrected" blindly would demo a model the benchmark itself rejected.
    selected: dict[tuple[str, str], str] = {}
    res = ROOT / "reports" / "results.json"
    if res.exists():
        import json

        for t in json.loads(res.read_text()).get("tiers", []):
            for m in t.get("models", []):
                selected[(m["name"], t["tier"])] = m.get("selected_run", "initial")

    def rank(path: Path, model: str) -> tuple[int, int]:
        stem = path.stem[len(model) + 1 :]           # e.g. "T2_small" or "T2_small_corrected"
        corrected = stem.endswith("_corrected")
        tier = stem.removesuffix("_corrected")
        want = selected.get((model, tier), "initial")
        preferred = int(corrected == (want == "corrected"))
        return (tier_index(tier) if tier in TIERS_BY_NAME else -1, preferred)

    for name in MODELS:
        candidates = [p for p in ck.glob("*.pt") if p.stem.startswith(name + "_")]
        if candidates:
            out[name] = max(candidates, key=lambda p: rank(p, name))
    return out


def colourise(gray: np.ndarray, mask: np.ndarray, colour=(220, 38, 38), alpha=0.45) -> np.ndarray:
    """Overlay a damage mask on the SAR backscatter image."""
    rgb = np.stack([gray] * 3, axis=-1).astype(np.float32)
    tint = np.zeros_like(rgb)
    tint[..., 0], tint[..., 1], tint[..., 2] = colour
    m = (mask > 0)[..., None].astype(np.float32)
    return np.clip(rgb * (1 - alpha * m) + tint * (alpha * m), 0, 255).astype(np.uint8)


st.title("🌊 AI-Based Crop Flood Damage Assessment")
st.caption(
    "Problem Statement 6 · Sentinel-1 SAR imagery → Healthy / Mild / Moderate / Severe "
    "→ district-level crop loss"
)

with st.sidebar:
    st.header("Configuration")
    ckpts = find_checkpoints()
    if ckpts:
        model_name = st.selectbox(
            "Trained model", list(ckpts), format_func=lambda n: DISPLAY_NAMES.get(n, n)
        )
        st.success(f"Checkpoint: `{ckpts[model_name].name}`")
        st.caption("Best available checkpoint per architecture: highest tier, and the run the benchmark actually selected.")
    else:
        model_name = None
        st.info("No checkpoint found — running in rule-based mode. Train with `make train`.")

    st.divider()
    st.subheader("Severity thresholds")
    st.caption("Net inundated fraction, permanent water excluded.")
    t1 = st.slider("Healthy → Mild", 0.0, 0.2, DEFAULT_THRESHOLDS[0], 0.005)
    t2 = st.slider("Mild → Moderate", 0.0, 0.3, DEFAULT_THRESHOLDS[1], 0.01)
    t3 = st.slider("Moderate → Severe", 0.1, 0.6, DEFAULT_THRESHOLDS[2], 0.01)
    st.caption(f"**Severe boundary.** {THRESHOLD_RATIONALE['severe']}")

tab_demo, tab_data, tab_about = st.tabs(["Assess a tile", "District loss estimate", "How it works"])

with tab_demo:
    src = st.radio(
        "Input", ["Use a sample tile from the dataset", "Upload my own SAR tile"], horizontal=True
    )

    planes = None
    if src.startswith("Use a sample"):
        idx_files = sorted((ROOT / "data").glob("index_*.json"))
        etci = ROOT / "data" / "etci"
        if not etci.exists():
            st.warning("No local tiles. Run `fcda download --tier T1_tiny` first.")
        else:
            import json

            from fcda.data.etci import load_planes

            recs = json.loads(idx_files[-1].read_text()) if idx_files else []
            if recs:
                labels = {r["tile_id"]: r["label"] for r in recs}
                order = sorted(recs, key=lambda r: -r["net_flood_fraction"])
                choice = st.selectbox(
                    "Tile",
                    [r["tile_id"] for r in order[:60]],
                    format_func=lambda t: f"{t}  ({SEVERITY_CLASSES[labels[t]]})",
                )
                planes = load_planes(ROOT / "data", choice)
    else:
        up = st.file_uploader("Grayscale SAR tile (PNG/JPG)", type=["png", "jpg", "jpeg"])
        if up:
            from PIL import Image

            g = np.array(Image.open(up).convert("L"))
            planes = {
                "post_vv": g,
                "post_vh": g,
                "post_flood": np.zeros_like(g),
                "post_water_body": np.zeros_like(g),
            }
            st.caption("Uploaded tiles have no ground-truth mask; the model supplies the mask.")

    if planes is not None:
        thresholds = (t1, t2, t3)
        truth = severity_from_masks(planes["post_flood"], planes["post_water_body"], thresholds)

        pred_mask, pred_name, pred_conf = None, None, None
        if model_name:
            import torch

            from fcda.data.etci import _stack_sar
            from fcda.data.severity import severity_from_probability_map
            from fcda.models.registry import is_temporal, wants_change_input
            from fcda.preprocess.transforms import build_transform

            model = load_model(model_name, ckpts[model_name])
            tf = build_transform(train=False, use_clahe=True)
            x = tf(_stack_sar(planes["post_vv"], planes["post_vh"], 256))
            xt = torch.from_numpy(x).float()[None]
            has_pre = "pre_vv" in planes
            if is_temporal(model_name) and has_pre:
                pre = tf(_stack_sar(planes["pre_vv"], planes["pre_vh"], 256))
                xt = torch.stack([torch.from_numpy(pre).float(), xt[0]], dim=0)[None]
            elif wants_change_input(model_name):
                # 6-channel change stack; an uploaded tile with no pre-flood frame gets zeros,
                # which is honest (no observed change) rather than silently wrong.
                pre = tf(_stack_sar(planes["pre_vv"], planes["pre_vh"], 256)) if has_pre else x
                xt = torch.from_numpy(np.concatenate([x, x - pre], axis=0)).float()[None]
            with torch.no_grad():
                seg, cls = model(xt)
            prob = torch.sigmoid(seg)[0, 0].numpy()
            temperature = load_temperature(model_name)
            probs = torch.softmax(cls / max(temperature, 1e-3), dim=1)[0].numpy()
            pred_mask = (prob >= 0.5).astype(np.uint8)
            pred = severity_from_probability_map(prob, thresholds=thresholds)
            pred_name, pred_conf = SEVERITY_CLASSES[int(probs.argmax())], float(probs.max())

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**SAR backscatter (VV)**")
            st.image(planes["post_vv"], width="stretch", clamp=True)
        with c2:
            st.markdown("**Reference flood mask**")
            st.image(
                colourise(planes["post_vv"], planes["post_flood"]),
                width="stretch",
            )
        with c3:
            st.markdown("**Predicted damage**" if pred_mask is not None else "**Permanent water**")
            import cv2

            show = pred_mask if pred_mask is not None else planes["post_water_body"]
            show = cv2.resize(show, planes["post_vv"].shape[::-1], interpolation=cv2.INTER_NEAREST)
            st.image(
                colourise(planes["post_vv"], show, colour=(37, 99, 235)),
                width="stretch",
            )

        st.divider()
        m1, m2, m3, m4 = st.columns(4)
        label = pred_name or truth.name
        conf = pred_conf
        m1.markdown(
            f"### <span style='color:{SEVERITY_COLOUR[label]}'>{label}</span>",
            unsafe_allow_html=True,
        )
        m1.caption("Predicted severity" if pred_name else "Rule-based severity")
        m2.metric("Net inundated", f"{truth.net_flood_fraction:.1%}")
        m3.metric("Permanent water", f"{truth.permanent_water_fraction:.1%}", help="Excluded from damage")
        m4.metric("Confidence", f"{conf:.1%}" if conf else "—",
                  help="Calibrated by temperature scaling fitted on held-out folds")
        if model_name:
            t = load_temperature(model_name)
            if abs(t - 1.0) > 1e-3:
                m4.caption(f"calibrated (T={t:.2f})")

        if pred_name and pred_name != truth.name:
            st.warning(f"Model says **{pred_name}**; the threshold rule on the reference mask says **{truth.name}**.")

        st.session_state["last"] = {
            "severity": label,
            "confidence": conf or 1.0,
            "flooded_fraction": truth.net_flood_fraction,
        }

with tab_data:
    df = load_tabular()
    if df is None:
        st.warning("Indian crop statistics not downloaded yet. Run `fcda download`.")
    else:
        from fcda.fusion import DEFAULT_LOSS_FRACTION, estimate_district_loss, fuse

        last = st.session_state.get("last", {"severity": "Moderate", "confidence": 0.8, "flooded_fraction": 0.2})
        c1, c2, c3 = st.columns(3)
        state = c1.selectbox("State", sorted(df["state"].unique()))
        district = c2.selectbox("District", sorted(df[df.state == state]["district"].unique()))
        severity = c3.selectbox("Observed severity", SEVERITY_CLASSES,
                                index=SEVERITY_CLASSES.index(last["severity"]))
        frac = st.slider("Net inundated fraction of the district", 0.0, 1.0,
                         float(min(last["flooded_fraction"], 1.0)), 0.01)

        rows = estimate_district_loss(df, district, severity, frac)
        if not rows:
            st.info("No records for that district.")
        else:
            import pandas as pd

            out = pd.DataFrame([r.to_dict() for r in rows])
            total = out["indicative_value_inr"].sum()
            a, b = st.columns(2)
            a.metric("Estimated production loss", f"{out['production_loss_tonnes'].sum():,.0f} t")
            b.metric("Indicative value", f"₹ {total / 1e7:,.1f} crore")
            st.dataframe(
                out[["crop", "cropped_area_kha", "affected_area_kha",
                     "assumed_loss_fraction", "production_loss_tonnes", "indicative_value_inr"]],
                width="stretch", hide_index=True,
            )
            st.caption(
                f"Assumed loss fraction at {severity}: {DEFAULT_LOSS_FRACTION[severity]:.0%}. "
                "These are planning coefficients, not measured agronomic values — the corpus "
                "has no ground-truth crop loss to calibrate against."
            )
            st.divider()
            st.subheader("Fusion")
            st.json(fuse(severity, last["confidence"], frac,
                         tabular_label=SEVERITY_CLASSES.index(severity), tabular_confidence=0.6))

with tab_about:
    st.markdown(
        """
### The sequence

**Image pipeline** — Load → Preprocess → Leakage checks → CLAHE → Split / 10-fold →
**GAN augmentation (training split only)** → Initial training → Over/under-fit detection →
Apply correction and retrain → Final test evaluation.

**Tabular pipeline** — Load CSV → Preprocess → Leakage checks → Split / 10-fold →
**SMOTE (training split only)** → Initial training → Over/under-fit detection →
Apply correction and retrain → Final test evaluation.

### Why Sentinel-1 SAR rather than optical imagery
Floods happen under cloud. Optical sensors cannot see through it, which is the first
limitation FLNet (arXiv 2601.03884) names in its own future work. SAR is unaffected by cloud.

### Why permanent water is subtracted
A tile containing a river is not a damaged tile. The reviewed literature measures flood IoU
at roughly half the IoU of permanent water precisely because the two get confused.

### How the models are evaluated
**5-fold cross validation**, not a single holdout. Every tile is scored exactly once by a model
that never saw it, so the evaluation covers all 900 tiles — including **all 20 Severe tiles in
the corpus**, where a single 15% holdout would score about three.

### Scope boundary — read this before trusting a number
- **Agricultural fields are not identified.** This system segments *water*. ETCI-2021 has no
  parcel labels, so YOLO12's detection head is present but unsupervised. The crop context comes
  entirely from the district statistics, not from the imagery.
- **Severity is derived from flood extent**, not ground-truth agronomic damage.
- **Only 20 Severe tiles exist** in the whole corpus, so Severe-class numbers are thin even
  under cross validation.
- **Rupee figures use assumed loss coefficients**, exposed as parameters rather than fitted.
        """
    )
    st.caption("Source: github.com/tejgokani/flood-crop-damage-ai")
