"""Indian district-level crop statistics: the tabular half of the project.

Source: ICRISAT district-level data for all Indian states (2010-2017), redistributed at
``nileshely/Crop-Datasets-for-All-Indian-States``. The raw file is *wide* -- one row per
(district, year) with an AREA/PRODUCTION/YIELD triple for each of ~20 crops -- so the first
job is to melt it into one row per (district, year, crop).

Why this dataset earns its place, rather than being decoration:

* It supplies the **regional agricultural context** that Papers 1, 5 and 6 all name as
  missing (gap G4). Paper 6 in particular nominates "data-scarce, critical flood zones" as
  the target for future work, and replaces static crop masks with real crop patterns.
* It produces a genuinely **imbalanced four-class problem**, which is what justifies SMOTE
  in Ma'am's CSV sequence rather than applying it for its own sake.
* It contains a **textbook target leakage trap**: YIELD = PRODUCTION / AREA, and our label is
  derived from yield. Leaving those columns in would give a near-perfect and completely
  worthless model. Our leakage check catches it; see ``preprocess/leakage.py``.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from .. import SEVERITY_CLASSES

ID_COLS = ["Dist Code", "Year", "State Code", "State Name", "Dist Name"]

#: Damage severity from yield anomaly (z-score against the district-crop baseline).
#: Negative z means the harvest came in below what that district normally achieves.
#: The Severe cut at -1.5 sigma corresponds to roughly the worst 7% of district-years.
YIELD_Z_THRESHOLDS: tuple[float, float, float] = (-1.5, -0.75, -0.25)

_TRIPLE_RE = re.compile(r"^(?P<crop>.+?) (?P<kind>AREA|PRODUCTION|YIELD) \(.*\)$")


def load_raw(csv_path: Path) -> pd.DataFrame:
    return pd.read_csv(csv_path)


def melt_to_long(df: pd.DataFrame) -> pd.DataFrame:
    """Wide (crop x metric) columns -> long rows of (district, year, crop, area, prod, yield)."""
    records: dict[tuple, dict] = {}
    for col in df.columns:
        m = _TRIPLE_RE.match(col)
        if not m:
            continue
        crop, kind = m.group("crop").strip(), m.group("kind").lower()
        for idx, val in df[col].items():
            key = (idx, crop)
            rec = records.setdefault(key, {})
            rec[kind] = val

    rows = []
    for (idx, crop), vals in records.items():
        base = df.loc[idx, ID_COLS].to_dict()
        base.update({"crop": crop, **vals})
        rows.append(base)
    long = pd.DataFrame(rows)
    long = long.rename(columns={"Year": "year", "State Name": "state", "Dist Name": "district"})
    # A district that did not grow a crop that year is not a damaged district.
    long = long[(long["area"].fillna(0) > 0) & (long["production"].fillna(0) >= 0)]
    return long.reset_index(drop=True)


def add_yield_anomaly(long: pd.DataFrame, min_years: int = 4) -> pd.DataFrame:
    """Z-score each district-crop yield against that district-crop's own history.

    Comparing a district only against itself is deliberate: absolute yield varies enormously
    between, say, Punjab wheat and Odisha wheat for reasons that have nothing to do with
    flooding. What signals damage is a district doing badly *relative to its own norm*.
    """
    g = long.groupby(["district", "crop"])["yield"]
    long = long.assign(
        yield_mean=g.transform("mean"),
        yield_std=g.transform("std"),
        n_years=g.transform("count"),
    )
    long = long[long["n_years"] >= min_years].copy()
    long["yield_z"] = (long["yield"] - long["yield_mean"]) / long["yield_std"].replace(0, np.nan)
    return long.dropna(subset=["yield_z"]).reset_index(drop=True)


def assign_damage_class(
    long: pd.DataFrame, thresholds: tuple[float, float, float] = YIELD_Z_THRESHOLDS
) -> pd.DataFrame:
    """Map the yield anomaly onto the same four severity classes as the imagery."""
    z = long["yield_z"].to_numpy()
    # searchsorted over ascending cuts: most negative z -> Severe (index 3).
    idx = np.searchsorted(np.asarray(thresholds), z, side="right")
    label = (len(SEVERITY_CLASSES) - 1) - np.clip(idx, 0, len(SEVERITY_CLASSES) - 1)
    long = long.copy()
    long["label"] = label.astype(int)
    long["label_name"] = [SEVERITY_CLASSES[i] for i in long["label"]]
    return long


FEATURE_COLS = ["area", "crop_share", "state_enc", "crop_enc", "year", "area_change", "area_z"]

#: Columns that must never reach the model: the label is computed from them.
LEAKY_COLS = ["yield", "production", "yield_z", "yield_mean", "yield_std", "label_name"]


def engineer_features(long: pd.DataFrame) -> pd.DataFrame:
    """Build the modelling frame, deliberately excluding the leaky columns."""
    df = long.copy()
    total_area = df.groupby(["district", "year"])["area"].transform("sum")
    df["crop_share"] = df["area"] / total_area.replace(0, np.nan)
    df["state_enc"] = df["state"].astype("category").cat.codes
    df["crop_enc"] = df["crop"].astype("category").cat.codes
    g_area = df.sort_values("year").groupby(["district", "crop"])["area"]
    df["area_change"] = g_area.pct_change().fillna(0.0)
    df["area_z"] = (df["area"] - g_area.transform("mean")) / g_area.transform("std").replace(0, np.nan)
    return df.fillna({"crop_share": 0.0, "area_z": 0.0})


def build_tabular_dataset(csv_path: Path) -> pd.DataFrame:
    """Full Pipeline-B data preparation, up to but not including the split."""
    raw = load_raw(csv_path)
    long = melt_to_long(raw)
    long = add_yield_anomaly(long)
    long = assign_damage_class(long)
    return engineer_features(long)
