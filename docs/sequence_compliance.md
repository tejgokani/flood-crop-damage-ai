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
| 5 | **Train / Val / Test Split**<br>**or 10-fold cross validation** | [`splits.py`](../src/fcda/splits.py) → `stratified_split`, `kfold_indices` | Both provided. Stratified 70/15/15 plus stratified 10-fold. `_assert_disjoint` raises on any overlap rather than training silently |
| 6 | **GAN Augmentation on TRAINING data only** | [`augment/gan.py`](../src/fcda/augment/gan.py) → `train_gan`, `generate_samples` | Conditional DCGAN generating image **and** mask together, conditioned on severity class. The function receives only training-split samples, so it has no route to val or test |
| 7 | **Initial Model Training** | [`train/loop.py`](../src/fcda/train/loop.py) → `train_model` | All five hybrids, wall-clock capped, best checkpoint retained |
| 8 | **Overfitting / Underfitting Detection** | [`train/diagnostics.py`](../src/fcda/train/diagnostics.py) → `diagnose` | Generalisation gap **and** learning-curve slope → `OVERFIT` / `UNDERFIT` / `OK`, with a stated reason |
| 9 | **Apply Correction Technique and Retrain** | [`train/correction.py`](../src/fcda/train/correction.py) → `plan_correction` | Emits a concrete config diff, then genuinely retrains. Keeps the corrected model **only if validation improved** |
| 10 | **Final Test Evaluation** | [`eval/metrics.py`](../src/fcda/eval/metrics.py) → `compute_metrics` | Test split touched exactly once, at the end. Macro-F1, accuracy, Cohen's κ, flood IoU, Dice, per-class F1, confusion matrix |

**Orchestrated by** [`pipeline.py`](../src/fcda/pipeline.py) → `run_image_pipeline`.

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

**Why is 10-fold CV on the tabular pipeline but not the headline image results?**
Cost. Ten folds × five deep architectures does not fit the compute budget, so the image
pipeline uses a stratified 70/15/15 split and CV indices are prepared and available
(`kfold_indices`). The tabular models are cheap enough that full 10-fold runs, and it does —
its mean and standard deviation across folds are reported in `reports/tabular.json`.

---

## Verifying these claims

```bash
make smoke     # runs every step above offline in under two minutes, printing each heading
make test      # asserts split disjointness and that val/test are never resampled
```

`tests/test_no_leakage.py` is the machine-checkable version of this document.
