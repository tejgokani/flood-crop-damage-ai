# Team Brief — presenting this at 9 AM

Read this cold and you can defend the work. It takes about ten minutes.

**The split.** Tej demos the code and the app. The second presenter owns the **research framing** —
the objective derived from the six papers, and the flow chart. That division is deliberate: it
looks like two people who actually divided the work, and the research half is where the
questions will land.

---

## What Ma'am asked for, and where each piece is

| Asked for | Artefact | Presented by |
|---|---|---|
| 60% of the code | The repo + live Streamlit demo | Tej |
| Objective from 5–6 papers' **future work** | [`literature_review.md`](literature_review.md) + [`objective.md`](objective.md) | Second presenter |
| Flow chart of the proposed work | [`flowchart.png`](flowchart.png) + [`sequence_compliance.md`](sequence_compliance.md) | Second presenter |

---

## The 60-second version

> Floods destroy crops over large areas, and field surveys are impossible in the days that
> matter because the roads are underwater. We estimate damage from **Sentinel-1 SAR satellite
> imagery**, which sees through cloud — and cloud is exactly what you get during a flood. We
> classify each tile as **Healthy, Mild, Moderate or Severe**, and then fuse that with real
> **Indian district crop statistics** to say how many hectares and tonnes were lost.
>
> We benchmarked **five hybrid architectures** on one identical protocol, and the whole thing
> follows the sequence you gave us — including leakage checks and applying GAN/SMOTE to the
> training split only.

---

## The six papers — the one thing to actually memorise

Each paper's **own future work** is quoted verbatim in `literature_review.md`. They converge on
six gaps, and **every gap maps to a component we built**. If you remember nothing else, remember
this table.

| Gap | What the literature says is missing | What we built |
|---|---|---|
| **G1** | Labelled data scarcity | GAN augmentation (training split only) + progressive data tiering |
| **G2** | Single-date imagery confuses flood with permanent water | **CNN + LSTM** on real pre/post pairs; permanent water subtracted |
| **G3** | Nobody compares architectures on one protocol | **Five hybrids, one shared split, loss and metric set** |
| **G4** | No fusion with regional agricultural context | Fusion with **ICRISAT Indian district crop data** |
| **G5** | Cloud defeats optical imagery | **Sentinel-1 SAR** |
| **G6** | Compute cost blocks deployment | Wall-clock-capped training; runs on a 16 GB laptop |

### Two quotes worth being able to say out loud

> "…experimenting with state-of-the-art backbones like Transformers" — **FLNet** (arXiv:2601.03884)

> "incorporating multi-temporal SAR data, in which pre-flood and post-flood images are jointly
> analyzed, could help models better distinguish temporary flooding from permanent water bodies."
> — **Explainable Flood Segmentation on Sentinel-1 SAR** (arXiv:2606.16302)

The first is why we have Swin + a five-model benchmark. The second is why we have CNN + LSTM.

---

## Likely questions, with answers

**"Why these papers?"**
They are the 2026 work closest to our problem: flood segmentation, crop damage assessment, and
hybrid CNN/Transformer architectures. Five are from 2026; the sixth is late 2025 and we kept it
because it is the only one that ties flood damage to crop-specific economic loss — the direct
precedent for our fusion layer. We label its year honestly rather than rounding it up.

**"Why is GAN/SMOTE applied after the split?"**
Because applying it before leaks. SMOTE creates new minority rows by interpolating between
neighbours — if it runs before the split, a test row can be interpolated into a training row,
and the score measures memorisation rather than learning. Same argument for a GAN that has seen
the test tiles. Your sequence puts both after the split for exactly this reason, and our code
enforces it structurally: the augmentation functions receive only training indices.

**"Did your leakage check actually find anything?"**
Yes, on the tabular side, and it is a real finding. Our damage label is derived from the yield
anomaly, and `YIELD = PRODUCTION / AREA`. So `yield`, `production` and the anomaly columns are
direct target leakage — six columns, dropped before modelling. Leaving them in gives a
near-perfect and completely worthless model.

**"Which model is best?"**
Point at the results table in the README. Say what the table says — do **not** inflate it. Then
add the honest caveat below about the Severe class.

**"Why is the Severe F1 low / unstable?"**
Because the entire corpus contains only **20 Severe tiles** — 0.98% of the pool. With a 70/15/15
split that is roughly 14 for training and 3 each for validation and test, so the Severe F1 is
computed over a handful of tiles and moves a lot. That is the ceiling of this dataset, not a bug
in the method, and it is precisely the data-scarcity gap the GAN augmentation targets.

**"Is your severity label real ground truth?"**
No, and we say so on the front page. ETCI-2021 gives flood masks, not agronomic crop-damage
labels. Severity is **derived** from the net inundated fraction per tile. The thresholds are
2% / 10% / 33%, and the 33% boundary is not arbitrary — Indian NDRF/SDRF relief practice treats
33% crop loss as the compensation threshold, so a "Severe" prediction lines up with a decision a
revenue officer actually makes. Validating against field-surveyed loss is the top item in our
remaining work.

**"Why subtract permanent water?"**
A tile containing a river is not a damaged tile. One of our reviewed papers measures flood IoU at
roughly half the IoU of permanent water precisely because models confuse the two. ETCI ships a
permanent-water mask, so we subtract it before labelling.

**"Why Bangladesh data for an Indian problem?"**
It is the Ganges–Brahmaputra delta — the same river system, the same monsoon, and agronomically
very close to the flood-prone eastern Indian belt. It is also the only region in the corpus with
a complete pre/post pair, which the temporal model needs. Cross-region transfer to Indian
districts is untested and listed as remaining work. FLNet states the same limitation about its
own single-location study.

**"How much is actually done?"**
Both pipelines end to end, all five architectures, the leakage checks, the diagnosis-and-
correction loop, the fusion layer, a working demo, and 37 passing tests with CI. What remains is
validation against real crop-loss records, cross-region transfer, S1+S2 fusion, calibration, and
longer training on better hardware.

---

## Walking the flow chart

Open `docs/flowchart.png` and go top to bottom on the left column:

1. **Load** — 2,032 tile pairs
2. **Preprocess** — VV, VH and their ratio
3. **Leakage checks** — *before* CLAHE, because CLAHE changes pixel values and would mask a
   duplicate pair that differs only in contrast
4. **CLAHE** — SAR sits in a narrow dark band; this lifts local contrast at the flood boundary
5. **Split + 10-fold**
6. **GAN augmentation — training only** (the yellow box; this is the leakage-critical step)
7. **Initial training** — the five hybrids
8. **Over/under-fit detection** — gap *and* learning-curve slope
9. **Correction + retrain** — and we keep the corrected model only if validation improved
10. **Final test evaluation** — the test split is touched exactly once

Then the right column is the same sequence for the CSV, with SMOTE in place of the GAN, and both
feed the fusion layer.

---

## If something breaks during the demo

- The Streamlit app runs **without** a checkpoint — it still shows preprocessing, the severity
  rule and the fusion arithmetic.
- `make smoke` runs the entire pipeline offline in under two minutes with no network.
- Every number in the README came from `reports/results.json`, which is committed.

---

## One thing that would genuinely improve the work

Our per-severity loss coefficients (10% / 25% / 60% yield loss for Mild / Moderate / Severe) are
**planning assumptions**, not measured values. If you can find a citable agronomic source —
inundation duration versus paddy yield loss, or state disaster-relief norms — send it over and it
goes straight into `src/fcda/fusion.py`. That turns an assumption into a citation.
