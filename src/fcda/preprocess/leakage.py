"""Data leakage checks -- step 3 of both of Ma'am's sequences.

Three families, matching the three she names (target leakage, duplicate leakage,
suspicious features):

* **Duplicate leakage** -- the same or near-identical image appearing in more than one
  split, which lets a model score well by memorisation. Detected with an average-hash
  perceptual fingerprint, so near-duplicates are caught, not just byte-identical files.
* **Target leakage** -- a feature that encodes the answer. Our tabular label is derived from
  the yield anomaly, so ``yield``, ``production`` and the anomaly columns are all direct
  leaks. This is the real thing, not a drill: left in, they produce a near-perfect and
  completely worthless model.
* **Suspicious features** -- columns implausibly correlated with the target, near-constant
  columns, and identifier-like columns that let a model memorise rows.

Findings are *reported*, then acted on. A check that silently drops columns teaches nobody
anything, so every function returns a structured finding list.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class LeakageFinding:
    kind: str
    severity: str  # "critical" | "warning" | "info"
    detail: str
    items: list = field(default_factory=list)

    def __str__(self) -> str:
        n = f" ({len(self.items)} items)" if self.items else ""
        return f"[{self.severity.upper():8s}] {self.kind}: {self.detail}{n}"


@dataclass
class LeakageReport:
    findings: list[LeakageFinding] = field(default_factory=list)

    def add(self, f: LeakageFinding) -> None:
        self.findings.append(f)

    @property
    def critical(self) -> list[LeakageFinding]:
        return [f for f in self.findings if f.severity == "critical"]

    @property
    def clean(self) -> bool:
        return not self.critical

    def to_dict(self) -> dict:
        return {
            "clean": self.clean,
            "n_findings": len(self.findings),
            "findings": [
                {"kind": f.kind, "severity": f.severity, "detail": f.detail, "items": f.items[:50]}
                for f in self.findings
            ],
        }

    def render(self) -> str:
        if not self.findings:
            return "No leakage findings."
        return "\n".join(str(f) for f in self.findings)


# --------------------------------------------------------------------------- images


def average_hash(img: np.ndarray, size: int = 8) -> int:
    """64-bit perceptual average hash of a single image plane.

    Downsample to 8x8, threshold at the mean, read the bits out. Two images that differ only
    by compression, mild noise or a small shift collapse to the same hash, which is what
    makes this catch *near* duplicates rather than only exact ones.
    """
    import cv2

    if img.ndim == 3:
        img = img.mean(axis=0)
    small = cv2.resize(img.astype(np.float32), (size, size), interpolation=cv2.INTER_AREA)
    bits = (small > small.mean()).flatten()
    out = 0
    for b in bits:
        out = (out << 1) | int(b)
    return out


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def check_duplicate_leakage(
    hashes_by_split: dict[str, dict[str, int]], max_distance: int = 3
) -> LeakageReport:
    """Flag images that appear in two different splits.

    ``hashes_by_split`` maps split name -> {tile_id: hash}.
    """
    report = LeakageReport()
    splits = sorted(hashes_by_split)

    for i, a in enumerate(splits):
        for b in splits[i + 1 :]:
            collisions = []
            hb = hashes_by_split[b]
            # Exact matches first: a dict lookup rather than an O(n*m) scan.
            inv_b: dict[int, list[str]] = {}
            for tid, h in hb.items():
                inv_b.setdefault(h, []).append(tid)
            for tid_a, ha in hashes_by_split[a].items():
                if ha in inv_b:
                    collisions.append((tid_a, inv_b[ha][0], 0))
                elif max_distance > 0:
                    for hbv, tids in inv_b.items():
                        if hamming(ha, hbv) <= max_distance:
                            collisions.append((tid_a, tids[0], hamming(ha, hbv)))
                            break
            if collisions:
                report.add(
                    LeakageFinding(
                        kind="duplicate_leakage",
                        severity="critical",
                        detail=f"{len(collisions)} near-duplicate tiles shared between {a} and {b}",
                        items=[f"{x}~{y}(d={d})" for x, y, d in collisions[:50]],
                    )
                )

    # Duplicates *within* a split are not leakage, but they do skew class balance.
    for s, hs in hashes_by_split.items():
        counts: dict[int, int] = {}
        for h in hs.values():
            counts[h] = counts.get(h, 0) + 1
        dupes = {h: c for h, c in counts.items() if c > 1}
        if dupes:
            report.add(
                LeakageFinding(
                    kind="intra_split_duplicates",
                    severity="warning",
                    detail=f"{sum(dupes.values())} duplicated tiles inside split '{s}'",
                    items=[],
                )
            )
    return report


# -------------------------------------------------------------------------- tabular


def check_target_leakage(
    df: pd.DataFrame,
    target: str,
    known_leaky: list[str] | None = None,
    corr_threshold: float = 0.95,
) -> LeakageReport:
    """Find columns that encode the target, by declaration and by correlation."""
    report = LeakageReport()

    declared = [c for c in (known_leaky or []) if c in df.columns]
    if declared:
        report.add(
            LeakageFinding(
                kind="target_leakage",
                severity="critical",
                detail=(
                    "columns used to construct the label are present and must be dropped "
                    "before modelling"
                ),
                items=declared,
            )
        )

    numeric = df.select_dtypes(include=[np.number])
    if target in numeric.columns:
        corr = numeric.corr(numeric_only=True)[target].drop(labels=[target], errors="ignore")
        strong = corr[corr.abs() >= corr_threshold]
        extra = [c for c in strong.index if c not in declared]
        if extra:
            report.add(
                LeakageFinding(
                    kind="target_leakage",
                    severity="critical",
                    detail=f"|correlation| >= {corr_threshold} with the target",
                    items=[f"{c}({strong[c]:+.3f})" for c in extra],
                )
            )
    return report


def check_suspicious_features(
    df: pd.DataFrame, target: str, near_constant: float = 0.99, id_uniqueness: float = 0.95
) -> LeakageReport:
    """Near-constant columns, identifier-like columns, and suspiciously strong predictors."""
    report = LeakageReport()
    n = len(df)

    constants, ids = [], []
    for c in df.columns:
        if c == target:
            continue
        vc = df[c].value_counts(dropna=False, normalize=True)
        if len(vc) and vc.iloc[0] >= near_constant:
            constants.append(f"{c}({vc.iloc[0]:.1%})")
        if df[c].nunique(dropna=False) / max(n, 1) >= id_uniqueness:
            ids.append(c)

    if constants:
        report.add(
            LeakageFinding(
                "suspicious_feature", "warning", "near-constant columns carry no signal", constants
            )
        )
    if ids:
        report.add(
            LeakageFinding(
                "suspicious_feature",
                "warning",
                "identifier-like columns let a model memorise rows",
                ids,
            )
        )

    numeric = df.select_dtypes(include=[np.number])
    if target in numeric.columns and numeric.shape[1] > 1:
        corr = numeric.corr(numeric_only=True)[target].drop(labels=[target], errors="ignore")
        moderate = corr[(corr.abs() >= 0.80) & (corr.abs() < 0.95)]
        if len(moderate):
            report.add(
                LeakageFinding(
                    "suspicious_feature",
                    "info",
                    "strong but sub-threshold correlation with the target; verify these are causal",
                    [f"{c}({moderate[c]:+.3f})" for c in moderate.index],
                )
            )
    return report


def check_split_overlap(splits: dict[str, pd.DataFrame], key_cols: list[str]) -> LeakageReport:
    """Flag identical entity keys appearing in more than one tabular split."""
    report = LeakageReport()
    names = sorted(splits)
    keyed = {n: set(map(tuple, splits[n][key_cols].to_numpy())) for n in names}
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            shared = keyed[a] & keyed[b]
            if shared:
                report.add(
                    LeakageFinding(
                        "split_overlap",
                        "critical",
                        f"{len(shared)} identical {key_cols} keys shared between {a} and {b}",
                        [str(s) for s in list(shared)[:20]],
                    )
                )
    return report


def merge_reports(*reports: LeakageReport) -> LeakageReport:
    out = LeakageReport()
    for r in reports:
        out.findings.extend(r.findings)
    return out
