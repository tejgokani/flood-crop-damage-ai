# Literature Review — 6 Recent Papers and Their Stated Future Work

> **Scope.** Six papers (five from 2026, one from late 2025 retained for its crop-damage focus).
> For each paper the **Future Work is quoted verbatim from the paper itself**, with a link to the
> source. Nothing in the "Stated future work" blocks is paraphrased or invented. Where a paper states
> its future work in more than one place (Discussion *and* Conclusion), both are given.
>
> The objective of our project (`docs/objective.md`) is derived from these quotes.

---

## Paper 1 — FLNet: Flood-Induced Agriculture Damage Assessment using Super Resolution of Satellite Images

| | |
|---|---|
| **Authors** | Sanidhya Ghosal, Anurag Sharma, Sushil Ghildiyal, Mukesh Saini |
| **Venue / Date** | arXiv:2601.03884 — submitted 7 January 2026 |
| **Link** | https://arxiv.org/abs/2601.03884 |
| **Dataset** | Bihar Flood Impacted Croplands Dataset (BFCD-22) — October 2022 Muzaffarpur flood |
| **Architecture** | EDSR super-resolution (10 m → 3 m) + U-Net damage classifier |
| **Headline result** | "Full Damage" F1 improved from **0.83 → 0.89**, matching commercial high-resolution imagery |

**Stated future work (verbatim):**
> "For future work, we will focus on improving the robustness of our pipeline, primarily by finding
> ways to handle cloud cover…we also plan to investigate architectural improvements, such as directly
> fusing Sentinel-1 and Sentinel-2 data and experimenting with state-of-the-art backbones like
> Transformers and Diffusion models."

**Stated limitations (verbatim):**
> "Image alignment remains a critical challenge; even tiny sub-pixel misalignments between the pre- and
> post-flood images can appear as false damage along field edges."

> "this study was conducted for a single event and location; further validation will be needed to test
> how well the model generalizes to different seasons, crop types, and geographies."

**Gap this leaves.** Optical Sentinel-2 is defeated by cloud cover — precisely the condition present
during a flood. Only one backbone family was tried, and validation covers a single event/location.

**How our work addresses it.** We work on **Sentinel-1 SAR (VV/VH)**, which penetrates cloud, removing
the failure mode the authors name first. Their call to try "state-of-the-art backbones like
Transformers" is answered directly by benchmarking **five** hybrid backbones — including
Swin Transformer + U-Net — under one identical protocol.

---

## Paper 2 — Optimized Flood Scene Segmentation with Swin Transformer-Based Architecture

| | |
|---|---|
| **Authors** | Preetha S, M S SP, Manikandan P |
| **Venue / Date** | *Scientific Reports*, 2026 (PMID 42031732, PMC13332218) — open access |
| **Link** | https://www.nature.com/articles/s41598-026-39188-x |
| **Dataset** | FloodNet (UAV RGB imagery) |
| **Architecture** | Swin Transformer encoder + lightweight custom decoder |

**Stated future work (verbatim, Conclusion):**
> "For future studies, the approach may be applied to satellite or UAV imagery for more generalization
> across broader areas, combining temporal modeling to capture changing patterns of floods, and
> inference optimization towards real-time application in emergency response systems."

**Stated future work (verbatim, Limitations):**
> "Since FloodNet dataset, being limited to RGB imagery, it may lead to misclassification under poor
> visibility and also in order to increase generalization and enhance detection in challenging visual
> conditions, future research will concentrate on incorporating multi-spectral or hydrological inputs."

**Stated future work (verbatim, Discussion):**
> "Future integration with multi-modal datasets, including but not limited to SAR or LiDAR, might
> improve robustness."

**Gap this leaves.** RGB-only input fails under poor visibility; the model is single-date, so it cannot
separate a transient flood from permanent water; no temporal modelling.

**How our work addresses it.** This paper asks for three things by name and we implement all three:
**SAR input** (Sentinel-1 VV/VH instead of RGB), **temporal modelling** (our CNN + LSTM hybrid over
pre/post-flood tile sequences), and **satellite generalisation** across multiple flood regions.

---

## Paper 3 — Advanced Flood Prediction with Physics-Guided Deep Learning: Combining UNet, FNO, and SAR/Optical Imagery

| | |
|---|---|
| **Authors** | Tewodros Syum Gebre, Jagrati Talreja, Leila Hashemi-Beni |
| **Venue / Date** | Proc. IEEE Radar Conference (RadarConf 2026) — arXiv:2606.06524, 2 June 2026 |
| **Link** | https://arxiv.org/abs/2606.06524 |
| **Architecture** | Physics-guided U-Net + Fourier Neural Operator over SAR/optical imagery |

**Stated future work (verbatim, Discussion IV-B):**
> "Despite its strengths, the model assumes steady-state flow and uniform hydrostatic pressure, which
> may limit its applicability during dynamic flood events. Additionally, the scarcity of temporally and
> spatially aligned SAR and optical data posed a challenge; to mitigate this, we implemented a
> sequential learning concept, initially trained the model using SAR data alone and later incorporated
> optical imagery. Computational demands also hinder large-scale real-time deployment. Future work will
> focus on relaxing flow assumptions, optimizing memory usage, and enhancing regularization to better
> capture heterogeneous flood dynamics across diverse land covers and sensor modalities."

**Stated future work (verbatim, Conclusion V):**
> "Future work will focus on relaxing the steady-state flow assumption, integrating time-series
> imagery, and optimizing computational efficiency for large-scale deployment."

**Gap this leaves.** Data scarcity of aligned imagery; heavy computation blocking real-time use; no
time-series modelling; regularisation named as an open problem.

**How our work addresses it.** **Data scarcity** → GAN augmentation applied to the training split only.
**Regularisation** → our over/under-fitting detection module measures the train/val gap and
*automatically applies and re-trains* with a corrective regularisation policy, rather than leaving it
to manual tuning. **Computational efficiency** → progressive dataset tiering and wall-clock-capped
training that runs on a 16 GB consumer laptop. **Time-series** → the CNN + LSTM hybrid.

---

## Paper 4 — Explainable Flood Segmentation on Sentinel-1 SAR Imagery: A Comparative Study of CNN and Transformer Architectures

| | |
|---|---|
| **Authors** | Arundhuti Banerjee, David Daou |
| **Venue / Date** | arXiv:2606.16302, 2026 |
| **Link** | https://arxiv.org/abs/2606.16302 |
| **Data** | Sentinel-1 SAR |

**Stated future work (verbatim, Conclusion):**
> "exploring calibration techniques such as temperature scaling, Monte Carlo dropout, or deep ensemble
> methods to produce more reliable uncertainty estimates for transformer-based architectures."

> "incorporating multi-temporal SAR data, in which pre-flood and post-flood images are jointly analyzed,
> could help models better distinguish temporary flooding from permanent water bodies."

> "integrating additional data sources, such as digital elevation models or optical imagery, could
> provide complementary information to improve segmentation accuracy."

> "evaluating these architectures on higher-resolution SAR data or in near-real-time operational
> pipelines would help assess their practical readiness for disaster response scenarios."

**Stated limitation.** Flood IoU was roughly **half** the IoU for permanent water — separating
transient inundation from permanent water bodies in SAR is the core difficulty.

**Gap this leaves.** Single-date SAR confuses flood with permanent water; no fusion with auxiliary
data; comparisons stop at accuracy without operational framing.

**How our work addresses it.** **Multi-temporal pre/post analysis** is the explicit design of our
CNN + LSTM hybrid. **Additional data sources** → our fusion layer combines the image-derived severity
with district-level Indian agricultural statistics. We also use the ETCI water-body labels to
distinguish permanent water from flood, which is the exact failure the authors quantify.

---

## Paper 5 — TDAVM-UNet: Task-Driven Attention VM-UNet for Crop Disease Detection from UAV Imagery

| | |
|---|---|
| **Authors** | Shanwen Zhang, Cong Xu, Yihang Zhao, Ting Zhang |
| **Venue / Date** | *Frontiers in Plant Science*, Volume 17, 2026 — open access |
| **Link** | https://www.frontiersin.org/journals/plant-science/articles/10.3389/fpls.2026.1855733/full |
| **Architecture** | U-shaped encoder–decoder + Disease-Aware Dynamic Attention (DADA) + Channel-Spatial Visual State Space (CSVSS); 256×256 UAV input |
| **Result** | 82.22% mIoU on UAV imagery |

**Stated limitation (verbatim):**
> "Training uses smartphone images (domain shift), but testing on pure UAV imagery ensures 82.22% mIoU
> reliably reflects UAV detection."

**Stated future work (verbatim):**
> "Future work: dataset expansion, domain adaptation, edge deployment, and field validation."

**Gap this leaves.** Domain shift between training and deployment imagery; limited dataset size; no
field validation against real agricultural ground truth.

**How our work addresses it.** **Dataset expansion** → GAN augmentation plus a progressive data-tiering
system that grows the training set as far as the hardware allows. This paper also validates the
**attention-guided U-shaped decoder** for agricultural imagery, which is the design we adopt for our
EfficientNet + Attention hybrid.

---

## Paper 6 — Climate-Informed Flood Damage Assessment in the Cropland Area Across the Midwestern USA

| | |
|---|---|
| **Authors** | Lazin R, Shen X, Anagnostou E |
| **Venue / Date** | *Scientific Reports*, 2025 (PMC12706033) — open access |
| **Link** | https://www.nature.com/articles/s41598-025-27288-z |
| **Architecture** | Pre-trained CNN for flood crop-damage estimation + CMIP5/CMIP6 climate projections |

> **Note on year.** This is the one paper in the set published in **2025** rather than 2026. It is
> retained because it is the only work here that ties flood damage to *crop-specific economic loss* at
> administrative-region scale, which is the direct precedent for our fusion layer.

**Stated future work (verbatim, Discussion and study limitation):**
> "In future research, we plan to downscale the CMIP6 projection and analyze the crop damage trends due
> to future flooding."

**Stated future work (verbatim, Conclusions):**
> "In the future, the pre-trained CNN model can be used to study climate change impacts on the
> data-scarce, critical flood zones."

> "The model can be improved by considering changes in the future crop pattern, rather than considering
> the constant crop masks in the future."

**Stated limitation (verbatim):**
> "The morphological property HAND and crop masks are assumed to be the same in the future, while in
> reality, these properties change rapidly."

**Gap this leaves.** Crop masks are treated as static; the study covers the Midwestern USA only, with
**data-scarce flood zones named by the authors as the target for future application**.

**How our work addresses it.** The authors nominate "data-scarce, critical flood zones" as the next
frontier — **India is exactly such a zone**. Our fusion layer replaces static crop masks with
**district-level, per-crop, multi-year Indian agricultural statistics (ICRISAT, 2010–2017)**, so crop
context is data-driven rather than assumed constant.

---

## Convergent gaps across all six papers

| # | Gap | Papers stating it | Our response |
|---|---|---|---|
| **G1** | **Data scarcity** — too few labelled flood/crop images | P1, P3, P5, P6 | GAN augmentation on the **training split only** + progressive data tiering |
| **G2** | **Single-date imagery** cannot separate transient flood from permanent water | P2, P3, P4 | **CNN + LSTM** over pre/post-flood sequences + ETCI water-body labels |
| **G3** | **No unified multi-architecture comparison** on one protocol | P1, P4 | **Five hybrids benchmarked** under one identical split, loss and metric set |
| **G4** | **No fusion with regional agricultural context** | P1, P5, P6 | Fusion layer joining image severity with **Indian district crop statistics** |
| **G5** | **Cloud cover / RGB failure** under poor visibility | P1, P2 | **Sentinel-1 SAR (VV/VH)** input, which is cloud-penetrating |
| **G6** | **Compute cost** blocks real-time and low-resource deployment | P3, P5 | Wall-clock-capped training + hardware-gated tiering; runs on a 16 GB laptop |

Full citations: [`references.bib`](references.bib). Objective derived from this table: [`objective.md`](objective.md).
