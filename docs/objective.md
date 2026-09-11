# Objective of the Proposed Work

> Derived from the **stated future work of six recent papers** — see
> [`literature_review.md`](literature_review.md), where every future-work statement is quoted verbatim
> from its source paper. Gap identifiers (G1–G6) below refer to the convergent-gap table in that document.

---

## Primary objective

> **To design and evaluate a cloud-penetrating, multi-temporal, multi-architecture deep learning system
> that identifies agricultural fields from Sentinel-1 SAR satellite imagery, classifies flood-induced
> crop damage into four severity levels — Healthy → Mild → Moderate → Severe — and fuses that
> image-derived severity with district-level Indian agricultural statistics to produce a
> region-scale crop loss assessment that is usable when field surveys are impossible.**

The three design commitments in that sentence each answer a gap the reviewed papers named themselves:

- **"cloud-penetrating"** — FLNet's first-named future-work item is "finding ways to handle cloud
  cover" (G5). We avoid the failure mode rather than patch it, by working on SAR instead of optical.
- **"multi-temporal"** — three papers independently ask for temporal modelling to distinguish transient
  flooding from permanent water (G2).
- **"multi-architecture"** — FLNet asks for "state-of-the-art backbones like Transformers" and Paper 4
  is itself a two-architecture comparison; neither delivers a unified benchmark (G3).

---

## Sub-objectives

### SO-1 — Benchmark five hybrid architectures under one identical protocol *(addresses G3)*
Implement and compare **YOLO12 + U-Net**, **ResNet + U-Net**, **EfficientNet + Attention**,
**Swin Transformer + U-Net**, and **CNN + LSTM** on a single shared data split, loss function and
metric set, so that architectural differences are actually attributable to the architecture.

*Motivated by:* FLNet — "experimenting with state-of-the-art backbones like Transformers and Diffusion
models." Paper 4 compares only CNN vs Transformer and calls for wider operational evaluation.

### SO-2 — Overcome labelled-data scarcity without touching evaluation integrity *(addresses G1)*
Apply **GAN-based augmentation to the training split only**, never to validation or test, and grow the
dataset through **hardware-gated progressive tiers** so the system trains on the largest volume the
available machine can actually sustain.

*Motivated by:* Paper 3 — "the scarcity of temporally and spatially aligned SAR and optical data posed
a challenge." Paper 5 — "Future work: dataset expansion, domain adaptation…"

### SO-3 — Model flood evolution over time, not a single snapshot *(addresses G2)*
Use **pre-flood and post-flood image sequences** through a CNN + LSTM hybrid, and exploit permanent
water-body labels, so that transient inundation is separated from standing water.

*Motivated by:* Paper 4 — "incorporating multi-temporal SAR data, in which pre-flood and post-flood
images are jointly analyzed, could help models better distinguish temporary flooding from permanent
water bodies." Paper 2 — "combining temporal modeling to capture changing patterns of floods."
Paper 3 — "integrating time-series imagery."

### SO-4 — Fuse imagery with real regional agricultural context *(addresses G4)*
Join the image-derived severity map with **district-level, per-crop, multi-year Indian agricultural
statistics** to translate flooded pixels into an agriculturally meaningful, economically
interpretable loss estimate.

*Motivated by:* Paper 6 nominates "data-scarce, critical flood zones" as the target for future
application, and identifies static crop masks as a limitation — "The model can be improved by
considering changes in the future crop pattern, rather than considering the constant crop masks."
Paper 4 — "integrating additional data sources… could provide complementary information."

### SO-5 — Make regularisation diagnostic and automatic, and keep compute feasible *(addresses G6)*
Detect over-fitting and under-fitting from the train/validation gap and learning-curve behaviour,
**automatically apply the matching correction and retrain**, and keep the whole pipeline inside a
wall-clock budget on consumer hardware (Apple M3, 16 GB).

*Motivated by:* Paper 3 — "Future work will focus on relaxing flow assumptions, optimizing memory
usage, and enhancing regularization… Computational demands also hinder large-scale real-time
deployment." Paper 5 — "edge deployment."

---

## Expected contributions

1. The **first unified benchmark** of these five specific hybrid architectures on a single SAR
   flood-damage protocol with a shared four-class severity definition.
2. A **leakage-audited pipeline**: duplicate/target/suspicious-feature checks run before any training,
   with findings reported rather than silently dropped.
3. An **automatic over/under-fit correction loop** that reports before/after metrics, making the
   regularisation decision reproducible instead of hand-tuned.
4. A **dual-modality fusion** of SAR-derived damage severity with Indian district crop statistics.
5. A **hardware-honest training protocol** — progressive tiering with a recorded scaling log — so the
   result is reproducible on a laptop, not only on a cluster.

---

## Scope boundary (stated honestly)

Severity labels in this work are **derived from flood extent** within each tile, using published
thresholds documented in `src/fcda/data/severity.py` — the ETCI-2021 corpus provides flood and
water-body masks, not agronomic crop-damage ground truth. Validation against field-surveyed crop loss
(as FLNet achieves with BFCD-22 for Bihar) is **not claimed here and is listed as remaining work**.
This mirrors the limitation Paper 1 states for itself: "further validation will be needed to test how
well the model generalizes to different seasons, crop types, and geographies."
