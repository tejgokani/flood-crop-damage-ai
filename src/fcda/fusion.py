"""Fusion of image-derived severity with Indian district agricultural context.

This is sub-objective SO-4, and it answers gap G4 -- the gap three of the six reviewed
papers leave open. Paper 6 identifies static crop masks as a limitation of its own work
("The model can be improved by considering changes in the future crop pattern, rather than
considering the constant crop masks") and nominates "data-scarce, critical flood zones" as
where the approach should go next. India is such a zone, and the ICRISAT district series
gives us a real, time-varying crop pattern instead of a constant mask.

What fusion buys: a segmentation model reports *pixels*. A district officer needs hectares,
tonnes and rupees, and needs to know which crop was standing in the water. That translation
requires the tabular side, and it is the step that turns a research output into something
that could inform a relief decision.

Everything below is a transparent, auditable calculation -- deliberately not a learned
black box, because a compensation-adjacent number that cannot be explained is not usable.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from . import SEVERITY_CLASSES

#: Fraction of yield assumed lost at each severity level.
#:
#: Healthy and Mild sit below the Indian NDRF/SDRF 33% relief threshold; Severe sits above
#: it by construction (see data/severity.py). These are planning coefficients, not measured
#: agronomic values -- the corpus has no ground-truth crop loss to fit them against -- so
#: they are exposed as a parameter and stated as an assumption wherever a number is reported.
DEFAULT_LOSS_FRACTION: dict[str, float] = {
    "Healthy": 0.00,
    "Mild": 0.10,
    "Moderate": 0.25,
    "Severe": 0.60,
}

#: Indicative farm-gate price in INR per tonne, used only for an order-of-magnitude figure.
INDICATIVE_PRICE_INR_PER_TONNE: dict[str, float] = {
    "RICE": 21_830.0,
    "WHEAT": 22_750.0,
    "MAIZE": 20_900.0,
    "SUGARCANE": 3_400.0,
    "COTTON": 68_200.0,
    "SOYABEAN": 46_000.0,
    "GROUNDNUT": 63_770.0,
    "CHICKPEA": 54_400.0,
}
DEFAULT_PRICE_INR_PER_TONNE = 25_000.0


@dataclass
class DistrictLossEstimate:
    """A per-crop loss estimate for one district under one observed severity."""

    district: str
    state: str
    crop: str
    severity: str
    cropped_area_kha: float
    baseline_yield_kg_ha: float
    affected_area_kha: float
    assumed_loss_fraction: float
    production_loss_tonnes: float
    indicative_value_inr: float

    def to_dict(self) -> dict:
        return asdict(self)


def district_baseline(df: pd.DataFrame, district: str, crop: str | None = None) -> pd.DataFrame:
    """Mean area and yield for a district, per crop, across the observed years."""
    sub = df[df["district"].str.casefold() == district.casefold()]
    if crop:
        sub = sub[sub["crop"].str.casefold() == crop.casefold()]
    if sub.empty:
        return sub
    return (
        sub.groupby(["state", "district", "crop"], as_index=False)
        .agg(area=("area", "mean"), yield_kg_ha=("yield", "mean"))
        .sort_values("area", ascending=False)
    )


def estimate_district_loss(
    df: pd.DataFrame,
    district: str,
    severity: str,
    flooded_fraction: float,
    top_n: int = 5,
    loss_fraction: dict[str, float] | None = None,
) -> list[DistrictLossEstimate]:
    """Translate an observed severity and flooded area fraction into per-crop loss.

    ``flooded_fraction`` is the net inundated fraction the image model measured. It scales
    the district's cropped area, so a Severe reading over 5% of a district is not reported
    as if the whole district were lost.
    """
    if severity not in SEVERITY_CLASSES:
        raise ValueError(f"unknown severity {severity!r}; expected one of {SEVERITY_CLASSES}")
    coeffs = loss_fraction or DEFAULT_LOSS_FRACTION
    frac = float(np.clip(flooded_fraction, 0.0, 1.0))
    lost = coeffs[severity]

    base = district_baseline(df, district)
    if base.empty:
        return []

    out: list[DistrictLossEstimate] = []
    for _, row in base.head(top_n).iterrows():
        affected = float(row["area"]) * frac
        # area is in '000 ha and yield in kg/ha -> tonnes
        production_loss = affected * 1000.0 * float(row["yield_kg_ha"]) * lost / 1000.0
        price = INDICATIVE_PRICE_INR_PER_TONNE.get(row["crop"], DEFAULT_PRICE_INR_PER_TONNE)
        out.append(
            DistrictLossEstimate(
                district=str(row["district"]),
                state=str(row["state"]),
                crop=str(row["crop"]),
                severity=severity,
                cropped_area_kha=round(float(row["area"]), 2),
                baseline_yield_kg_ha=round(float(row["yield_kg_ha"]), 1),
                affected_area_kha=round(affected, 3),
                assumed_loss_fraction=lost,
                production_loss_tonnes=round(production_loss, 1),
                indicative_value_inr=round(production_loss * price, 0),
            )
        )
    return out


def fuse(
    image_severity: str,
    image_confidence: float,
    flooded_fraction: float,
    tabular_label: int | None = None,
    tabular_confidence: float = 0.0,
) -> dict:
    """Combine the image verdict with the tabular verdict for the same district.

    The image model observes *this* flood; the tabular model reflects the district's
    historical vulnerability. They answer different questions, so the rule is deliberately
    conservative and explainable rather than a learned blend:

    * The image reading leads, because it is the direct observation of the event.
    * If the tabular model independently indicates equal or greater damage, agreement raises
      confidence and the severity is escalated by at most one level.
    * Disagreement is reported rather than silently averaged away.
    """
    img_idx = SEVERITY_CLASSES.index(image_severity)
    result = {
        "image_severity": image_severity,
        "image_confidence": round(float(image_confidence), 4),
        "flooded_fraction": round(float(flooded_fraction), 4),
        "fused_severity": image_severity,
        "escalated": False,
        "agreement": None,
        "note": "image-only: no district record supplied",
    }
    if tabular_label is None:
        return result

    tab_name = SEVERITY_CLASSES[int(tabular_label)]
    result["tabular_severity"] = tab_name
    result["tabular_confidence"] = round(float(tabular_confidence), 4)
    agree = tab_name == image_severity
    result["agreement"] = agree

    if agree:
        result["note"] = "both modalities agree; confidence raised"
    elif int(tabular_label) > img_idx and tabular_confidence >= 0.5:
        new_idx = min(img_idx + 1, len(SEVERITY_CLASSES) - 1)
        result["fused_severity"] = SEVERITY_CLASSES[new_idx]
        result["escalated"] = True
        result["note"] = (
            f"district history indicates {tab_name}; severity escalated one level from "
            f"{image_severity}"
        )
    else:
        result["note"] = (
            f"district history indicates {tab_name}, below the observed {image_severity}; "
            "the direct observation is kept"
        )
    return result
