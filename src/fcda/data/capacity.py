"""Hardware capacity gate for progressive dataset scaling.

Ma'am's instruction: *start with a smaller number of images and gradually increase in such a
manner that our PC can withstand it.* This module is the "can withstand it" half.

After each tier finishes we record what it actually cost — peak resident memory, free disk,
seconds per epoch — and only then decide whether the next tier is affordable. If it is not,
we stop at the last good tier and write down why. Stopping early is a successful outcome,
not a failure: the results table from the last completed tier is still complete.
"""

from __future__ import annotations

import json
import shutil
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import psutil

from .tiers import Tier, next_tier

#: Never let free disk fall below this. The dev machine starts with ~16 GB free.
MIN_FREE_DISK_GB = 6.0
#: Never let peak process memory exceed this fraction of total RAM.
MAX_MEM_FRACTION = 0.70
#: A tier projected to take longer than this is refused.
DEFAULT_BUDGET_MINUTES = 45.0


@dataclass
class TierMeasurement:
    """What one tier actually cost when it ran."""

    tier: str
    n_tiles: int
    image_size: int
    seconds_per_epoch: float
    epochs_run: int
    peak_rss_gb: float
    free_disk_gb_after: float
    total_seconds: float
    ok: bool = True
    note: str = ""


@dataclass
class ScalingLog:
    """The full audit trail of the tier ladder, written to reports/scaling_log.json."""

    machine: dict = field(default_factory=dict)
    measurements: list[TierMeasurement] = field(default_factory=list)
    final_tier: str = ""
    stopped_because: str = ""

    def to_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "machine": self.machine,
            "measurements": [asdict(m) for m in self.measurements],
            "final_tier": self.final_tier,
            "stopped_because": self.stopped_because,
        }
        path.write_text(json.dumps(payload, indent=2))


def machine_profile() -> dict:
    """Snapshot of the host, recorded so results are reproducible in context."""
    import platform

    vm = psutil.virtual_memory()
    return {
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "cpu_count": psutil.cpu_count(logical=True),
        "total_ram_gb": round(vm.total / 1024**3, 2),
        "free_disk_gb": round(free_disk_gb(), 2),
        "python": platform.python_version(),
    }


def free_disk_gb(path: str | Path = ".") -> float:
    return shutil.disk_usage(Path(path).resolve()).free / 1024**3


def peak_rss_gb() -> float:
    """Resident memory of this process right now, in GB."""
    return psutil.Process().memory_info().rss / 1024**3


def total_ram_gb() -> float:
    return psutil.virtual_memory().total / 1024**3


class MemoryWatermark:
    """Tracks the high-water mark of RSS across a training tier."""

    def __init__(self) -> None:
        self.peak = peak_rss_gb()
        self._t0 = time.time()

    def sample(self) -> float:
        self.peak = max(self.peak, peak_rss_gb())
        return self.peak

    @property
    def elapsed(self) -> float:
        return time.time() - self._t0


def can_advance(
    measurement: TierMeasurement,
    current: Tier,
    budget_minutes: float = DEFAULT_BUDGET_MINUTES,
) -> tuple[bool, str]:
    """Decide whether stepping up from ``current`` is safe.

    Returns ``(allowed, reason)``. The reason is recorded either way so the scaling log
    explains itself without needing the code.
    """
    nxt = next_tier(current.name)
    if nxt is None:
        return False, "already at the largest tier"

    if not measurement.ok:
        return False, f"tier {current.name} did not complete cleanly: {measurement.note}"

    # 1. Disk: would the next tier's download push us under the floor?
    projected_free = measurement.free_disk_gb_after - (nxt.approx_disk_mb / 1024)
    if projected_free < MIN_FREE_DISK_GB:
        return False, (
            f"disk floor: next tier needs ~{nxt.approx_disk_mb:.0f} MB, leaving "
            f"{projected_free:.1f} GB free (floor is {MIN_FREE_DISK_GB} GB)"
        )

    # 2. Memory: scale the observed peak by tile count and pixel area.
    tile_ratio = nxt.n_tiles / max(current.n_tiles, 1)
    area_ratio = (nxt.image_size / current.image_size) ** 2
    # Only the activation-ish part of memory scales with image area; the model weights do not.
    projected_mem = measurement.peak_rss_gb * (0.35 + 0.65 * area_ratio)
    mem_ceiling = total_ram_gb() * MAX_MEM_FRACTION
    if projected_mem > mem_ceiling:
        return False, (
            f"memory ceiling: projected {projected_mem:.1f} GB exceeds "
            f"{mem_ceiling:.1f} GB ({MAX_MEM_FRACTION:.0%} of {total_ram_gb():.0f} GB RAM)"
        )

    # 3. Time: seconds/epoch scales with tiles and pixel area.
    projected_spe = measurement.seconds_per_epoch * tile_ratio * area_ratio
    projected_minutes = projected_spe * nxt.epochs / 60.0
    if projected_minutes > budget_minutes:
        return False, (
            f"time budget: next tier projected at {projected_minutes:.0f} min, "
            f"budget is {budget_minutes:.0f} min"
        )

    return True, (
        f"advancing to {nxt.name}: projected {projected_minutes:.0f} min, "
        f"{projected_mem:.1f} GB peak, {projected_free:.1f} GB disk free"
    )
