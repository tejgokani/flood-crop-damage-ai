"""Build the Tech-a-thon presentation deck from the live results."""
from __future__ import annotations

import json
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Emu, Inches, Pt

NAVY, SLATE, ACCENT = RGBColor(0x0F, 0x23, 0x3E), RGBColor(0x47, 0x55, 0x69), RGBColor(0x1D, 0x4E, 0xD8)
GREEN, AMBER, WHITE = RGBColor(0x15, 0x80, 0x3D), RGBColor(0xB4, 0x53, 0x09), RGBColor(0xFF, 0xFF, 0xFF)

ROOT = Path(__file__).parent
REPORTS = ROOT / "reports"


def load(name):
    p = REPORTS / name
    return json.loads(p.read_text()) if p.exists() else None


prs = Presentation()
prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
BLANK = prs.slide_layouts[6]


def slide(title: str, subtitle: str = ""):
    s = prs.slides.add_slide(BLANK)
    tb = s.shapes.add_textbox(Inches(0.55), Inches(0.34), Inches(12.2), Inches(0.85))
    p = tb.text_frame.paragraphs[0]
    p.text = title
    p.font.size, p.font.bold, p.font.color.rgb = Pt(30), True, NAVY
    if subtitle:
        p2 = tb.text_frame.add_paragraph()
        p2.text = subtitle
        p2.font.size, p2.font.color.rgb = Pt(13), SLATE
    return s


def bullets(s, items, left=0.7, top=1.55, width=12.0, size=15, gap=8):
    tb = s.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(5.3))
    tf = tb.text_frame
    tf.word_wrap = True
    for i, it in enumerate(items):
        txt, lvl = (it if isinstance(it, tuple) else (it, 0))
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = ("• " if lvl == 0 else "– ") + txt
        p.level = lvl
        p.font.size = Pt(size if lvl == 0 else size - 2)
        p.font.color.rgb = NAVY if lvl == 0 else SLATE
        p.space_after = Pt(gap)
    return tb


def table(s, rows, left, top, width, height, header_fill=NAVY, fs=10.5):
    r, c = len(rows), len(rows[0])
    shape = s.shapes.add_table(r, c, Inches(left), Inches(top), Inches(width), Inches(height))
    t = shape.table
    for j, val in enumerate(rows[0]):
        cell = t.cell(0, j)
        cell.text = str(val)
        cell.fill.solid(); cell.fill.fore_color.rgb = header_fill
        para = cell.text_frame.paragraphs[0]
        para.font.size, para.font.bold, para.font.color.rgb = Pt(fs), True, WHITE
    for i in range(1, r):
        for j in range(c):
            cell = t.cell(i, j)
            cell.text = str(rows[i][j])
            para = cell.text_frame.paragraphs[0]
            para.font.size = Pt(fs)
            para.font.color.rgb = NAVY
    return t


# ---------------------------------------------------------------- 1 title
s = prs.slides.add_slide(BLANK)
box = s.shapes.add_textbox(Inches(0.9), Inches(2.1), Inches(11.5), Inches(2.6))
tf = box.text_frame; tf.word_wrap = True
p = tf.paragraphs[0]
p.text = "AI-Based Crop Flood Damage Assessment"
p.font.size, p.font.bold, p.font.color.rgb = Pt(42), True, NAVY
for txt, sz, col in [
    ("Agriculture + Disaster Management  ·  Problem Statement 6", 18, ACCENT),
    ("Five hybrid deep learning architectures benchmarked on one protocol,", 15, SLATE),
    ("fused with Indian district crop statistics", 15, SLATE),
    ("", 8, SLATE),
    ("VIT Tech-a-thon · Fall Semester 2026–27", 13, SLATE),
    ("github.com/tejgokani/flood-crop-damage-ai", 12, ACCENT),
]:
    q = tf.add_paragraph(); q.text = txt
    q.font.size, q.font.color.rgb = Pt(sz), col

# ---------------------------------------------------------------- 2 problem
s = slide("The problem", "Why field surveys fail exactly when they are needed")
bullets(s, [
    "Floods destroy crops across large geographical regions.",
    "Traditional field surveys are impossible immediately after a disaster — roads and agricultural land are inaccessible.",
    "Relief and insurance decisions still have to be made, within days.",
    "",
    ("Required output: Healthy → Mild → Moderate → Severe crop damage", 0),
    ("Required approach: satellite / drone imagery + deep learning, five hybrid techniques", 0),
], size=16, gap=12)

# ---------------------------------------------------------------- 3 objective from papers
s = slide("Objective — derived from six papers' own future work",
          "Every quote below is verbatim from the source paper's Future Work / Limitations section")
rows = [["Paper (2026)", "Its stated future work — verbatim", "What we built"],
        ["FLNet\narXiv:2601.03884", "\"…experimenting with state-of-the-art backbones like Transformers\"; \"handle cloud cover\"", "Swin + 5-model benchmark;\nSAR instead of optical"],
        ["Swin flood scene\nSci Rep 2026", "\"combining temporal modeling…\"; \"future research will concentrate on incorporating multi-spectral or hydrological inputs\"", "CNN+LSTM;\nSentinel-1 SAR not RGB"],
        ["Physics-guided U-Net+FNO\nRadarConf 2026", "\"relaxing flow assumptions, optimizing memory usage, and enhancing regularization\"", "Automatic over/under-fit\ncorrection; capped compute"],
        ["Explainable SAR CNN/Transformer\narXiv:2606.16302", "\"incorporating multi-temporal SAR data, in which pre-flood and post-flood images are jointly analyzed\"", "Real pre/post pairs;\npermanent water subtracted"],
        ["TDAVM-UNet\nFrontiers Plant Sci 2026", "\"Future work: dataset expansion, domain adaptation, edge deployment…\"", "GAN augmentation;\nprogressive data tiering"],
        ["Climate-informed cropland\nSci Rep", "\"…can be used to study climate change impacts on the data-scarce, critical flood zones\"", "Indian district crop fusion\n(a data-scarce flood zone)"]]
t = table(s, rows, 0.5, 1.5, 12.35, 5.2, fs=9)
t.columns[0].width, t.columns[1].width, t.columns[2].width = Emu(Inches(2.9)), Emu(Inches(6.1)), Emu(Inches(3.35))

# ---------------------------------------------------------------- 4 objective statement
s = slide("The objective statement")
tb = s.shapes.add_textbox(Inches(0.8), Inches(1.5), Inches(11.8), Inches(2.0))
tf = tb.text_frame; tf.word_wrap = True
p = tf.paragraphs[0]
p.text = ("To design and evaluate a cloud-penetrating, multi-temporal, multi-architecture deep "
          "learning system that identifies agricultural fields from Sentinel-1 SAR imagery, "
          "classifies flood-induced crop damage as Healthy → Mild → Moderate → Severe, and fuses "
          "that severity with district-level Indian agricultural statistics to produce a "
          "region-scale crop loss assessment usable when field surveys are impossible.")
p.font.size, p.font.color.rgb, p.font.italic = Pt(17), NAVY, True
bullets(s, [
    "SO-1  Benchmark five hybrids under one identical protocol            → gap G3",
    "SO-2  Overcome data scarcity without touching evaluation integrity   → gap G1",
    "SO-3  Model flood evolution over time, not a single snapshot         → gap G2",
    "SO-4  Fuse imagery with real regional agricultural context           → gap G4",
    "SO-5  Make regularisation automatic and keep compute feasible        → gap G6",
], top=3.7, size=14, gap=9)

# ---------------------------------------------------------------- 5 flowchart
s = slide("Flow chart of the proposed work", "Both prescribed sequences, implemented in the given order")
fc = ROOT / "docs" / "flowchart.png"
if fc.exists():
    s.shapes.add_picture(str(fc), Inches(0.85), Inches(1.25), height=Inches(5.95))

# ---------------------------------------------------------------- 6 sequence compliance
s = slide("Compliance with the prescribed sequence", "Every step maps to a function that actually runs")
rows = [["#", "Step", "Implemented in"],
        ["1", "Load dataset", "data/etci.py · data/india_agri.py"],
        ["2", "Data preprocessing", "preprocess/transforms.py"],
        ["3", "Data leakage checks — target · duplicate · suspicious", "preprocess/leakage.py"],
        ["4", "CLAHE", "preprocess/clahe.py"],
        ["5", "Split / 10-fold cross validation", "splits.py"],
        ["6", "GAN augmentation / SMOTE — TRAINING DATA ONLY", "augment/gan.py · augment/smote.py"],
        ["7", "Initial model training", "train/loop.py"],
        ["8", "Overfitting / underfitting detection", "train/diagnostics.py"],
        ["9", "Apply correction technique and retrain", "train/correction.py"],
        ["10", "Final test evaluation", "eval/metrics.py"]]
t = table(s, rows, 0.6, 1.5, 12.1, 4.6, fs=11)
t.columns[0].width, t.columns[1].width, t.columns[2].width = Emu(Inches(0.5)), Emu(Inches(6.6)), Emu(Inches(5.0))
tb = s.shapes.add_textbox(Inches(0.6), Inches(6.25), Inches(12.1), Inches(0.8))
p = tb.text_frame.paragraphs[0]
p.text = ("Why augmentation comes after the split: SMOTE interpolates between neighbours, so fitting it "
          "before the split lets a test row be interpolated into a training row — the score then measures "
          "memorisation. tests/test_no_leakage.py asserts val/test are never resampled.")
p.font.size, p.font.color.rgb, p.font.italic = Pt(11), AMBER, True
tb.text_frame.word_wrap = True

# ---------------------------------------------------------------- 7 architectures
s = slide("Two hybrid architectures, developed in depth",
          "All five were implemented and benchmarked; two were carried forward. Shared decoder and head.")
rows = [["Hybrid", "Params", "Encoder", "Input", "Why this one"],
        ["YOLO12 + U-Net", "5.5M", "R-ELAN + area attention\n(implemented natively)",
         "6-channel change stack\n[post VV,VH,ratio, ΔVV,ΔVH,Δratio]",
         "Was the weakest, for a diagnosable\nreason: from scratch, peaked at epoch 1"],
        ["CNN + LSTM", "9.0M", "Shared CNN +\nConvLSTM at every scale",
         "Ordered pre/post pair\n[T=2, 3 channels]",
         "The only model of the five that\ndid not overfit (gap +0.093)"]]
t = table(s, rows, 0.5, 1.5, 12.35, 2.1, fs=10)
for col, w in zip(t.columns, [2.2, 0.8, 2.8, 3.3, 3.25]):
    col.width = Emu(Inches(w))
bullets(s, [
    "They are architecturally complementary: one reaches the pre→post change through an explicit difference channel, the other through recurrence. Comparing those says something; comparing two ImageNet CNNs would not.",
    "The other three (ResNet+U-Net, EfficientNet+Attention, Swin+U-Net) remain in the repo with their benchmarked results archived — this is depth over breadth, stated up front.",
], top=3.85, size=12.5, gap=12)

# ---------------------------------------------------------------- 7b the overfitting fix
s = slide("The problem was memorisation, not architecture",
          "Train-val macro-F1 gap in the five-model benchmark")
rows = [["Model", "Gap", "Verdict"],
        ["Swin + U-Net", "+0.350", "memorising"],
        ["EfficientNet + Attention", "+0.319", "memorising"],
        ["ResNet + U-Net", "+0.292", "memorising"],
        ["YOLO12 + U-Net", "+0.185", "memorising — peaked at epoch 1"],
        ["CNN + LSTM", "+0.093", "the only healthy model"]]
table(s, rows, 0.6, 1.5, 5.9, 2.6, fs=11)
bullets(s, [
    "5-fold cross validation — every tile scored out-of-fold",
    "Change-detection input (the static scene cancels)",
    "YOLO12 narrowed 9.8M → 5.5M",
    "Flood-fraction regression loss",
    "Dihedral augmentation (8×) + crop + coarse dropout",
    "Regularisation raised at the start, not reactively",
    "Frequency-aware sampling (α = 0.5)",
    "Temperature-scaled confidence",
], left=6.9, top=1.55, width=6.0, size=12, gap=7)
tb = s.shapes.add_textbox(Inches(0.6), Inches(4.35), Inches(6.0), Inches(1.6))
tb.text_frame.word_wrap = True
p_ = tb.text_frame.paragraphs[0]
p_.text = ("Note what is NOT on that list: training longer. A model that peaks at epoch 1 does not "
           "improve at epoch 40. The freed compute bought better estimates and stronger "
           "regularisation instead.")
p_.font.size, p_.font.color.rgb, p_.font.italic = Pt(12), AMBER, True

# ---------------------------------------------------------------- 8 data
s = slide("Data", "Both sources public, no credentials, nothing bulk-downloaded")
bullets(s, [
    "Imagery — ETCI-2021 Sentinel-1 SAR over Bangladesh (Ganges–Brahmaputra delta)",
    ("2,032 tiles with a genuine pre/post pair: 2017-03-14 pre-monsoon and 2017-06-06 monsoon flood", 1),
    ("Each carries VV, VH, a flood mask and a permanent-water mask at both dates", 1),
    "Agriculture — ICRISAT district statistics: 310 Indian districts · 23 crops · 2010–2017",
    ("Label from per-district yield anomaly → a real 10.9 : 1 class imbalance, which is what justifies SMOTE", 1),
    "",
    "Severity = net flood fraction (flood − permanent water), binned at 2% / 10% / 33%",
    ("The 33% boundary is the Indian NDRF/SDRF crop-loss relief threshold — not a round number", 1),
    ("Permanent water is subtracted because a tile containing a river is not a damaged tile", 1),
], size=14, gap=8)

# ---------------------------------------------------------------- 9 progressive scaling
s = slide("Progressive data scaling", "Start small; step up only as far as the machine demonstrably sustains")
rows = [["Tier", "Tiles", "Size", "Purpose"],
        ["T0_smoke", "40", "128 px", "Pipeline proof — offline, runs in CI"],
        ["T1_tiny", "150", "128 px", "First real data"],
        ["T2_small", "400", "192 px", "First meaningful signal"],
        ["T3_medium", "900", "256 px", "Target tier"],
        ["T4_large", "1800", "256 px", "Only if T3 finished comfortably"]]
table(s, rows, 0.7, 1.5, 7.2, 2.9, fs=11)
bullets(s, [
    "After each tier we measure peak RSS, free disk and seconds/epoch",
    "Advance only if the next tier stays under a 6 GB disk floor, 70% of RAM and the time budget",
    "Otherwise stop at the last good tier and record why — in reports/scaling_log.json",
    "A complete five-model table exists after the first tier, so there is always something to show",
], left=8.2, top=1.6, width=4.6, size=12, gap=10)

# ---------------------------------------------------------------- 10 results
import sys as _sys
_sys.path.insert(0, str(ROOT / "src"))
from fcda.eval.cv_report import (  # noqa: E402
    calibration_table, headline_table, load as load_cv, supplementary_table,
)

cv = load_cv(REPORTS)
if cv:
    s = slide("Results — 5-fold cross validation",
              f"Every one of {cv.get('n_tiles', 0)} tiles scored exactly once by a model that never saw it")

    def md_table(md, left, top, width, height, fs=10.5):
        lines = [ln for ln in md.strip().split("\n") if ln.strip().startswith("|")]
        rows_ = [[c.strip() for c in ln.strip().strip("|").split("|")] for ln in lines]
        rows_ = [r for r in rows_ if not all(set(c) <= set("-: ") for c in r)]
        rows_ = [[c.replace("**", "") for c in r] for r in rows_]
        return table(s, rows_, left, top, width, height, fs=fs)

    md_table(headline_table(cv), 0.5, 1.45, 12.35, 1.5)
    tb = s.shapes.add_textbox(Inches(0.5), Inches(3.05), Inches(12.35), Inches(0.6))
    tb.text_frame.word_wrap = True
    p_ = tb.text_frame.paragraphs[0]
    p_.text = ("Train-val gap is the anti-overfitting check: ≤ 0.10 is healthy, and negative means "
               "validation scored above training. The five-model baseline ran at +0.19 to +0.35.")
    p_.font.size, p_.font.color.rgb = Pt(11), SLATE

    md_table(supplementary_table(cv), 0.5, 3.7, 12.35, 1.3)
    tb = s.shapes.add_textbox(Inches(0.5), Inches(5.15), Inches(12.35), Inches(1.9))
    tb.text_frame.word_wrap = True
    p_ = tb.text_frame.paragraphs[0]
    p_.text = ("Supplementary, not a replacement. Macro-F1 treats the four classes as unrelated, so "
               "calling a Severe tile Moderate scores as badly as calling it Healthy — and every "
               "Severe miss in our confusion matrices lands on Moderate, the adjacent class.")
    p_.font.size, p_.font.color.rgb = Pt(11.5), SLATE

    # calibration slide
    s = slide("Confidence calibration", "Temperature scaling, fitted on inner folds only")
    md_table2 = calibration_table(cv)
    lines = [ln for ln in md_table2.strip().split("\n") if ln.strip().startswith("|")]
    rows_ = [[c.strip().replace("**", "") for c in ln.strip().strip("|").split("|")] for ln in lines]
    rows_ = [r for r in rows_ if not all(set(c) <= set("-: ") for c in r)]
    table(s, rows_, 0.6, 1.5, 12.1, 1.4, fs=11)
    bullets(s, [
        "An uncalibrated softmax is a score that sums to one, not a probability.",
        "Temperature scaling divides the logits by one learned scalar — it is argmax-invariant, so it cannot change accuracy, only how honest the confidence is.",
        "Fitted on an inner validation slice of each training fold, never on the held-out fold.",
        "A fold's fit is accepted only if it cuts calibration error by at least 20%; otherwise the temperature resets to 1.0 and nothing is applied.",
        "Paper 4 of our review names temperature scaling as its first future-work item.",
    ], top=3.2, size=13, gap=11)
else:
    s = slide("Results", "")
    bullets(s, ["Cross-validation run in progress — regenerate with `python scripts_make_slides.py`."])

# ---------------------------------------------------------------- 11 figures
for fig, title in (("cv_gaps.png", "Generalisation gap per fold — below the line is healthy"),
                   ("cv_confusion.png", "Out-of-fold confusion over every tile in the dataset"),
                   ("predictions.png", "Predictions across severity classes")):
    fp = REPORTS / "figures" / fig
    if fp.exists():
        s = slide(title)
        s.shapes.add_picture(str(fp), Inches(0.7), Inches(1.6), width=Inches(11.9))

# ---------------------------------------------------------------- 12 demo
s = slide("Working demo", "streamlit run app/streamlit_app.py")
bullets(s, [
    "Select or upload a SAR tile → predicted damage mask overlaid on the backscatter image",
    "Severity with confidence, net inundated fraction, permanent water shown separately",
    "District loss estimate: affected hectares, production loss in tonnes, indicative value in ₹ crore",
    "Fusion panel showing whether imagery and district history agree, and whether severity was escalated",
    "",
    "Runs with or without a trained checkpoint — the rule-based path still demonstrates preprocessing, "
    "the severity rule and the fusion arithmetic",
], size=14, gap=10)

# ---------------------------------------------------------------- 13 limitations
s = slide("Honest limitations", "Stated because they are the questions worth asking")
bullets(s, [
    "Agricultural field delineation is NOT implemented. The brief asks to identify agricultural fields; this system segments water. ETCI-2021 has no parcel labels, so YOLO12's detection head is present but unsupervised, and the crop context comes from district statistics. This is the largest gap against the brief.",
    "Severity is derived from flood extent, not measured agronomic damage — ETCI-2021 has flood masks, not crop-damage ground truth. Validation against field-surveyed loss is NOT claimed.",
    "Only 20 Severe tiles exist in the whole pool (0.98%). Cross validation scores all 20 rather than the ~3 a single holdout would, but it remains the corpus ceiling.",
    "Tiers cap Healthy at 45% to make the benchmark usable; the natural 86.9% prior is reported alongside every result, not hidden.",
    "Training geography is the Ganges–Brahmaputra delta — a reasonable analogue for eastern India, but cross-region transfer is untested. FLNet states the same limitation about itself.",
    "Rupee figures use assumed per-severity loss coefficients, surfaced as parameters rather than fitted values.",
], size=11.5, gap=9)

# ---------------------------------------------------------------- 14 remaining
s = slide("What remains — the other 40%")
bullets(s, [
    "Supervised agricultural field delineation — needs a cropland mask or parcel labels",
    "Validation against field-surveyed crop loss (BFCD-22, or state revenue records)",
    "Cross-region transfer to Indian districts, and domain adaptation",
    "Sentinel-1 + Sentinel-2 fusion, as FLNet's future work proposes",
    "Calibration and uncertainty estimates, as the Explainable SAR paper proposes",
    "10-fold cross validation extended to the image pipeline once compute allows",
    "Higher tiers (T4) and longer schedules on stronger hardware",
], size=15, gap=13)

# ---------------------------------------------------------------- 15 close
s = slide("Summary")
bullets(s, [
    "Objective derived from six papers' own stated future work — quoted verbatim, not paraphrased",
    "Both prescribed sequences implemented in full, every step mapped to a function",
    "Five hybrid architectures on one shared protocol — the comparison isolates the encoder",
    "Leakage checks that produce real findings; augmentation confined to the training split by construction",
    "Automatic over/under-fit detection and correction, with before/after recorded",
    "Fusion turns predicted pixels into hectares, tonnes and rupees for a real Indian district",
    "",
    "github.com/tejgokani/flood-crop-damage-ai",
], size=14, gap=11)

out = ROOT / "slides" / "Techathon_PS6.pptx"
out.parent.mkdir(exist_ok=True)
prs.save(out)
print(f"wrote {out} — {len(prs.slides.__iter__.__self__._sldIdLst)} slides")
