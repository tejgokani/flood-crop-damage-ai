# Compliance with the Prescribed Sequence

`Sequence.docx` prescribes two sequences. This project has both an imagery side and an Indian
agricultural CSV side, so **both are implemented in full**. Every step below maps to a named
function that actually runs — the numbered console output of `fcda train` prints these step
headings in order, so the code can be read against the flow chart line by line.

---

## Pipeline A — Image classification sequence

| # | Step as prescribed | Where it lives | What it actually does |
|---|---|---|---|
| 1 | **Load Image Dataset** | [`data/etci.py`](../src/fcda/data/etci.py) → `build_index`, `load_planes` | Loads ETCI-2021 Sentinel-1 SAR tiles: 2,032 pre/post pairs, each with VV, VH, flood mask and permanent-water mask |
| 2 | **Data Preprocessing** | [`preprocess/transforms.py`](../src/fcda/preprocess/transforms.py) → `build_transform` | Stacks VV + VH + derived VV/VH ratio, resizes to the tier resolution, standardises per channel |
| 3 | **Data Leakage Checks** | [`preprocess/leakage.py`](../src/fcda/preprocess/leakage.py) | All three families Ma'am names — see the breakdown below |
| 3a | · duplicate leakage | `check_duplicate_leakage`, `average_hash` | 64-bit perceptual average hash per tile; flags near-duplicates across splits within a Hamming radius, not just byte-identical files |
| 3b | · target leakage | `check_target_leakage` | Declared leaky columns plus any feature correlating with the target above threshold |
| 3c | · suspicious features | `check_suspicious_features` | Near-constant columns, identifier-like columns, suspiciously strong predictors |
| 4 | **CLAHE** | [`preprocess/clahe.py`](../src/fcda/preprocess/clahe.py) → `CLAHE` | Clip limit 2.0, 8×8 tile grid, applied to the SAR polarisation channels only — the derived ratio channel is left untouched so it keeps its physical meaning |
| 5 | **Train / Val / Test Split**<br>**or 10-fold cross validation** | [`splits.py`](../src/fcda/splits.py) → `stratified_split`, `kfold_indices`; [`train/crossval.py`](../src/fcda/train/crossval.py) | Both branches implemented. **The cross-validation branch is the headline** — see the note below. `_assert_disjoint` raises on any overlap rather than training silently |
| 6 | **GAN Augmentation on TRAINING data only** | [`augment/gan.py`](../src/fcda/augment/gan.py) → `train_gan`, `generate_samples` | Conditional DCGAN generating image **and** mask together, conditioned on severity class. The function receives only training-split samples, so it has no route to val or test |
| 7 | **Initial Model Training** | [`train/loop.py`](../src/fcda/train/loop.py) → `train_model` | All five hybrids, wall-clock capped, best checkpoint retained |
| 8 | **Overfitting / Underfitting Detection** | [`train/diagnostics.py`](../src/fcda/train/diagnostics.py) → `diagnose` | Generalisation gap **and** learning-curve slope → `OVERFIT` / `UNDERFIT` / `OK`, with a stated reason |
| 9 | **Apply Correction Technique and Retrain** | [`train/correction.py`](../src/fcda/train/correction.py) → `plan_correction` | Emits a concrete config diff, then genuinely retrains. Keeps the corrected model **only if validation improved** |
| 10 | **Final Test Evaluation** | [`eval/metrics.py`](../src/fcda/eval/metrics.py) → `compute_metrics` | Test split touched exactly once, at the end. Macro-F1, accuracy, Cohen's κ, flood IoU, Dice, per-class F1, confusion matrix |

**Orchestrated by** [`pipeline.py`](../src/fcda/pipeline.py) → `run_image_pipeline` (single
split, used for the demo checkpoints) and `run_cv_pipeline` (cross-validated, the headline).

### Why step 5 takes the cross-validation branch

`Sequence.docx` offers "Train / Validation / Test Split **or** 10-fold cross validation". We take
the second branch for the headline numbers, and the reason is specific to this corpus rather
than a general preference:

- A 15% test split of 900 tiles is 135 samples, of which roughly **three** are Severe. Any
  number computed on three samples is noise, and a Severe F1 of 0.000 on three tiles says
  almost nothing about the model.
- Cross-validation scores **every tile exactly once, by a model that never saw it**. The
  evaluation set becomes 900 tiles and **all 20 Severe tiles in the corpus** get scored.
- It also yields a fold-to-fold standard deviation, so a modest number arrives with error bars.

Five folds rather than ten, because ten would not fit the wall-clock budget for two deep models
on a laptop. That trade is recorded rather than glossed over.

Each fold splits its training portion again into an inner train/validation pair. Early stopping
and the calibration temperature are fitted on the **inner** validation only; the held-out fold is
touched exactly once, for scoring. Fitting anything on the held-out fold would reintroduce the
overfitting the cross-validation exists to measure.

---

## Pipeline B — CSV classification with class imbalance handling

| # | Step as prescribed | Where it lives | What it actually does |
|---|---|---|---|
| 1 | **Load CSV Dataset** | [`data/india_agri.py`](../src/fcda/data/india_agri.py) → `load_raw` | ICRISAT district crop statistics: 310 districts, 23 crops, 2010–2017 |
| 2 | **Data Preprocessing** | `melt_to_long`, `add_yield_anomaly`, `engineer_features` | Wide (crop × metric) → long rows; per-district-crop yield z-score; feature engineering |
| 3 | **Data Leakage Checks** | `check_target_leakage`, `check_suspicious_features` | **Fires on real data**: `YIELD = PRODUCTION / AREA` and the label is derived from yield, so six columns are direct target leakage and are dropped before modelling |
| 4 | **Train / Val / Test Split**<br>**or 10-fold cross validation** | `grouped_stratified_split`, `kfold_indices` | Grouped by **district**, so no district spans train and test — otherwise the model memorises a district's yield level instead of learning damage. Verified overlap = 0 |
| 5 | **SMOTE on TRAINING data only** | [`augment/smote.py`](../src/fcda/augment/smote.py) → `balance_training_split` | 10.9 : 1 → 1 : 1. Validation and test row counts are asserted unchanged by `assert_untouched` |
| 6 | **Initial Model Training** | `run_tabular_pipeline` | RandomForest on the SMOTE-balanced training split |
| 7 | **Overfitting / Underfitting Detection** | `diagnose` | Same diagnostic used by the image pipeline |
| 8 | **Apply Correction Technique and Retrain** | `run_tabular_pipeline` | Over-fit → depth limit, minimum leaf size, feature subsampling, balanced class weights. Under-fit → more and deeper trees |
| 9 | **10-fold cross validation** | `kfold_indices` | SMOTE is refitted **inside each fold**, on that fold's training portion only — refitting outside the fold loop is the classic way this step leaks |
| 10 | **Final Test Evaluation** | `compute_metrics` | Held-out districts, scored once |

**Orchestrated by** [`pipeline.py`](../src/fcda/pipeline.py) → `run_tabular_pipeline`.

---

## The three points most likely to be questioned

**Why is GAN/SMOTE after the split rather than before?**
Because resampling before the split leaks. SMOTE synthesises minority rows by interpolating
between neighbours; if it runs before the split, a test row can be interpolated into a
training row, and the reported score measures memorisation. The same argument applies to a
GAN that has seen the test tiles. Both sequences in `Sequence.docx` place these steps after
the split for exactly this reason, and our code enforces it structurally — the augmentation
functions are handed training indices and have no access to the rest.

**Why do the leakage checks run before CLAHE?**
CLAHE changes pixel values, so running duplicate detection afterwards would compare enhanced
images and could mask a duplicate pair that differs only in contrast. Checking first means we
are fingerprinting the data as it arrived.

**Why five folds on the image pipeline and ten on the tabular one?**
Cost, and it is stated rather than hidden. A RandomForest fold takes seconds, so the tabular
pipeline runs the full ten that `Sequence.docx` names. A deep model fold takes ~13 minutes on an
M3, so ten folds × two architectures would not fit the budget; five folds still scores every
tile out-of-fold, which is the property that matters.

**How do you know the models are not overfitting?**
It is measured, not asserted. Every fold records final training macro-F1 minus best validation
macro-F1 (`eval/metrics.py` → `generalisation_gap`), the per-fold values are in the README, and
`reports/figures/cv_gaps.png` plots them against a 0.10 threshold. The earlier five-model run sat
at +0.19 to +0.35; the fixes in `preprocess/transforms.py` (dihedral augmentation),
`models/yolo12_unet.py` (narrower, change-detection input) and `train/loop.py` (regularised
defaults, frequency-aware sampling) are what moved it.

---

## Verifying these claims

```bash
make smoke     # runs every step above offline in under two minutes, printing each heading
make test      # asserts split disjointness and that val/test are never resampled
```

`tests/test_no_leakage.py` is the machine-checkable version of this document.
