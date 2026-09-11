"""Progressive dataset tiers.

The image dataset is never downloaded in one shot. We start at a deliberately tiny tier,
prove the machine handles it, and only then step up. ``capacity.py`` owns the decision of
whether a step up is allowed; this module only declares what the rungs of the ladder are.

Rationale: the development machine is an Apple M3 with 16 GB unified memory and roughly
16 GB of free disk. A fixed large dataset would either exhaust the disk or thermally stall
the run overnight with nothing to show. Walking the ladder means there is a complete,
presentable set of results after the *first* tier, and every later tier is a bonus.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Tier:
    """One rung of the dataset ladder."""

    name: str
    n_tiles: int
    image_size: int
    epochs: int
    batch_size: int
    description: str

    @property
    def approx_disk_mb(self) -> float:
        """Rough on-disk cost: 3 PNG planes (vv, vh, mask) per tile at ~45 KB each."""
        return self.n_tiles * 3 * 45 / 1024


#: Ordered from smallest to largest. Index 0 must always be runnable offline.
#:
#: Resolution climbs only once, at T2, and then holds at 192 px. The reason is a lesson from
#: the first run: per-epoch cost scales with the *square* of the image size, so a tier that
#: raised both the tile count and the resolution blew through the per-model wall-clock cap
#: and trained for fewer epochs than the tier below it. Flood IoU went *down* from T1 to T2,
#: which inverts the whole point of a scaling ladder.
#:
#: Holding resolution fixed above T2 means each rung varies only the amount of data, which is
#: both the honest experiment ("does more data help?") and affordable enough for every model
#: to actually finish its epochs.
TIERS: tuple[Tier, ...] = (
    Tier("T0_smoke", 40, 128, 2, 8, "Pipeline proof. Synthetic-capable, used by CI."),
    Tier("T1_tiny", 150, 128, 8, 8, "First real ETCI data; confirms download and labels are sane."),
    Tier("T2_small", 400, 192, 14, 8, "First meaningful training signal."),
    Tier("T3_medium", 900, 192, 14, 8, "Target tier: 2.25x the data at the same resolution."),
    Tier("T4_large", 2000, 224, 14, 4, "Only if T3 finished comfortably inside budget."),
)

TIERS_BY_NAME: dict[str, Tier] = {t.name: t for t in TIERS}


def get_tier(name: str) -> Tier:
    if name not in TIERS_BY_NAME:
        raise KeyError(f"Unknown tier {name!r}. Available: {sorted(TIERS_BY_NAME)}")
    return TIERS_BY_NAME[name]


def next_tier(name: str) -> Tier | None:
    """The next rung up, or ``None`` if already at the top."""
    names = [t.name for t in TIERS]
    idx = names.index(name)
    return TIERS[idx + 1] if idx + 1 < len(TIERS) else None


def tier_index(name: str) -> int:
    return [t.name for t in TIERS].index(name)
