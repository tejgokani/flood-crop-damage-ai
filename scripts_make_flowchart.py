"""Render the proposed-work flow chart to PNG/SVG for the slide deck and report."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

C_STEP, C_ONLY, C_DEC, C_OUT = "#e8eef7", "#fde68a", "#e0e7ff", "#bbf7d0"
C_EDGE, C_TXT = "#334155", "#0f172a"


def box(ax, x, y, w, h, text, fc=C_STEP, fs=8.2, bold=False, style="round,pad=0.02"):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle=style,
                                linewidth=1.3, edgecolor=C_EDGE, facecolor=fc))
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, color=C_TXT,
            fontweight="bold" if bold else "normal", linespacing=1.45)
    return (x, y, w, h)


def arrow(ax, a, b, label="", style="-|>", ls="-", color=C_EDGE, rad=0.0, dx=0.0):
    ax.add_patch(FancyArrowPatch((a[0] + dx, a[1] - a[3] / 2), (b[0] + dx, b[1] + b[3] / 2),
                                 arrowstyle=style, linestyle=ls, mutation_scale=12,
                                 linewidth=1.2, color=color,
                                 connectionstyle=f"arc3,rad={rad}"))
    if label:
        ax.text((a[0] + b[0]) / 2 + dx + 0.12, (a[1] - a[3] / 2 + b[1] + b[3] / 2) / 2,
                label, fontsize=6.6, color="#475569", ha="left", va="center")


fig, ax = plt.subplots(figsize=(15.5, 12.2))
ax.set_xlim(0, 16); ax.set_ylim(-0.75, 12.4); ax.axis("off")

ax.text(8, 12.05, "Flow Chart of the Proposed Work", ha="center", fontsize=16, fontweight="bold", color=C_TXT)
ax.text(8, 11.68, "AI-Based Crop Flood Damage Assessment  ·  Problem Statement 6  ·  sequence as prescribed",
        ha="center", fontsize=9.5, color="#475569")

# ---------------- Pipeline A (left)
ax.add_patch(FancyBboxPatch((0.35, 1.45), 6.5, 9.85, boxstyle="round,pad=0.05",
                            linewidth=1.5, edgecolor="#94a3b8", facecolor="#f8fafc", zorder=0))
ax.text(3.6, 11.05, "PIPELINE A — IMAGE  (Sentinel-1 SAR, ETCI-2021)",
        ha="center", fontsize=10, fontweight="bold", color="#1e293b")

xa, w, h = 3.6, 5.4, 0.62
ys = [10.35, 9.45, 8.55, 7.65, 6.75, 5.8, 4.85, 3.85, 2.85, 1.95]
A = []
A.append(box(ax, xa, ys[0], w, h, "Load Image Dataset\n2,032 pre/post tile pairs"))
A.append(box(ax, xa, ys[1], w, h, "Data Preprocessing\nVV + VH + derived ratio · resize · normalise"))
A.append(box(ax, xa, ys[2], w, h, "Data Leakage Checks\nduplicate · near-duplicate pHash · split overlap"))
A.append(box(ax, xa, ys[3], w, h, "CLAHE\nclip 2.0 · 8×8 grid · SAR channels only"))
A.append(box(ax, xa, ys[4], w, h, "Train / Validation / Test Split\n+ 10-fold Cross Validation"))
A.append(box(ax, xa, ys[5], w, h + 0.12, "GAN Augmentation — TRAINING DATA ONLY\nconditional DCGAN, balances Mild/Moderate/Severe", fc=C_ONLY, bold=True))
A.append(box(ax, xa, ys[6], w, h, "Initial Model Training\n5 hybrid architectures, shared decoder + head"))
A.append(box(ax, xa, ys[7], w, h + 0.05, "Overfitting / Underfitting Detection\ngeneralisation gap + learning-curve slope", fc=C_DEC, bold=True))
A.append(box(ax, xa, ys[8], w, h + 0.05, "Apply Correction Technique and Retrain\n↑dropout ↑decay ↑augment  |  ↓reg ↑LR ↑epochs"))
A.append(box(ax, xa, ys[9], w, h, "Final Test Evaluation", bold=True))
for i in range(len(A) - 1):
    arrow(ax, A[i], A[i + 1])
ax.text(xa + 0.18, (ys[7] + ys[8]) / 2, "OVERFIT / UNDERFIT", fontsize=6.6, color="#b45309",
        va="center", ha="left", fontweight="bold")
# OK bypass
ax.add_patch(FancyArrowPatch((xa - w / 2, ys[7]), (xa - w / 2, ys[9]), arrowstyle="-|>",
                             mutation_scale=12, linewidth=1.2, color="#15803d",
                             connectionstyle="arc3,rad=0.42"))
ax.text(xa - w / 2 - 0.75, (ys[7] + ys[9]) / 2, "OK", fontsize=7, color="#15803d", fontweight="bold")

# ---------------- Pipeline B (right)
ax.add_patch(FancyBboxPatch((7.3, 2.78), 6.1, 8.52, boxstyle="round,pad=0.05",
                            linewidth=1.5, edgecolor="#94a3b8", facecolor="#f8fafc", zorder=0))
ax.text(10.35, 11.05, "PIPELINE B — TABULAR  (ICRISAT Indian district crop data)",
        ha="center", fontsize=10, fontweight="bold", color="#1e293b")

xb, wb = 10.35, 5.1
ysb = [10.35, 9.45, 8.55, 7.65, 6.7, 5.75, 4.8, 3.85, 3.1]
B = []
B.append(box(ax, xb, ysb[0], wb, h, "Load CSV Dataset\n310 districts · 23 crops · 2010–2017"))
B.append(box(ax, xb, ysb[1], wb, h, "Data Preprocessing\nwide→long melt · yield-anomaly z-score"))
B.append(box(ax, xb, ysb[2], wb, h, "Data Leakage Checks\ntarget · duplicate · suspicious features"))
B.append(box(ax, xb, ysb[3], wb, h, "Train / Val / Test Split — grouped by district\n+ 10-fold Cross Validation"))
B.append(box(ax, xb, ysb[4], wb, h + 0.12, "SMOTE — TRAINING DATA ONLY\nclass imbalance 10.9 : 1  →  1 : 1", fc=C_ONLY, bold=True))
B.append(box(ax, xb, ysb[5], wb, h, "Initial Model Training"))
B.append(box(ax, xb, ysb[6], wb, h + 0.05, "Overfitting / Underfitting Detection", fc=C_DEC, bold=True))
B.append(box(ax, xb, ysb[7], wb, h, "Apply Correction Technique and Retrain"))
B.append(box(ax, xb, ysb[8], wb, h * 0.8, "Final Test Evaluation", bold=True))
for i in range(len(B) - 1):
    arrow(ax, B[i], B[i + 1])

# ---------------- Models panel, placed in the clear space beneath Pipeline B so the
# connector does not cut diagonally across it.
ax.add_patch(FancyBboxPatch((7.3, 1.42), 8.4, 1.02, boxstyle="round,pad=0.05",
                            linewidth=1.4, edgecolor="#7c3aed", facecolor="#faf5ff", zorder=0))
ax.text(11.5, 2.26, "5 HYBRID ARCHITECTURES  —  shared U-Net decoder + shared severity head",
        ha="center", fontsize=8.2, fontweight="bold", color="#5b21b6")
for i, m in enumerate(["YOLO12\n+ U-Net", "ResNet\n+ U-Net", "EfficientNet\n+ Attention",
                       "Swin Transformer\n+ U-Net", "CNN + LSTM\n(pre/post temporal)"]):
    box(ax, 7.92 + i * 1.63, 1.80, 1.52, 0.56, m, fc="#ede9fe", fs=6.4)
ax.add_patch(FancyArrowPatch((xa + w / 2, ys[6]), (7.4, 2.05), arrowstyle="-|>", linestyle=":",
                             mutation_scale=10, linewidth=1.2, color="#7c3aed",
                             connectionstyle="arc3,rad=-0.25"))
ax.text(6.95, 3.15, "trains", fontsize=6.6, color="#7c3aed", ha="center", style="italic")

# ---------------- Fusion + output
fus = box(ax, 5.4, 0.42, 6.6, 0.62, "FUSION LAYER   ·   image-derived severity  ×  district crop context",
          fc="#dbeafe", bold=True, fs=9)
out = box(ax, 7.4, -0.42, 8.8, 0.6,
          "OUTPUT:  Healthy → Mild → Moderate → Severe   +   damage mask   +   district-level crop loss estimate",
          fc=C_OUT, bold=True, fs=9)
ax.add_patch(FancyArrowPatch((xa, ys[9] - h / 2), (4.2, 0.73), arrowstyle="-|>", mutation_scale=12,
                             linewidth=1.2, color=C_EDGE, connectionstyle="arc3,rad=0.1"))
ax.add_patch(FancyArrowPatch((xb, ysb[8] - h * 0.4), (7.0, 0.73), arrowstyle="-|>", mutation_scale=12,
                             linewidth=1.2, color=C_EDGE, connectionstyle="arc3,rad=-0.12"))
ax.add_patch(FancyArrowPatch((5.4, 0.11), (6.4, -0.12), arrowstyle="-|>", mutation_scale=12,
                             linewidth=1.2, color=C_EDGE))

ax.text(8.0, -0.88, "Yellow = applied to the TRAINING split ONLY (this is what prevents leakage)   ·   "
                  "Blue = automatic diagnosis step   ·   Green = final output",
        fontsize=7.2, color="#64748b", ha="center")

fig.tight_layout()
out_dir = Path("docs")
fig.savefig(out_dir / "flowchart.png", dpi=175, bbox_inches="tight", facecolor="white")
fig.savefig(out_dir / "flowchart.svg", bbox_inches="tight", facecolor="white")
print("wrote docs/flowchart.png and docs/flowchart.svg")
