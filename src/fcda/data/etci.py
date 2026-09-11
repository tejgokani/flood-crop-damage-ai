"""ETCI-2021 tile index and PyTorch datasets.

The corpus we train on is the Bangladesh subset of ETCI-2021, which is the only region in
the train split carrying all four raster planes at more than one acquisition date:

* ``2017-03-14`` -- pre-monsoon, the dry reference.
* ``2017-06-06`` -- the monsoon flood.

That gives 2,068 tiles with a genuine pre/post pair. Single-date models consume the
post-flood acquisition; the CNN+LSTM hybrid consumes the ordered pair. The Ganges-
Brahmaputra delta is also a reasonable analogue for the flood-prone eastern Indian
agricultural belt, which is what the fusion layer is aimed at.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .download import PLANES, POST_DATE, PRE_DATE, build_pair_index, download_tiles, fetch_manifest
from .severity import DEFAULT_THRESHOLDS, severity_from_masks
from .synthetic import make_dataset as make_synthetic


@dataclass
class TileRecord:
    """One tile: where its files are, and what class it belongs to."""

    tile_id: str
    label: int
    net_flood_fraction: float
    permanent_water_fraction: float


def _read_gray(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("L"))


def load_planes(root: Path, tile_id: str) -> dict[str, np.ndarray]:
    """Read all eight planes (four per date) for a tile from local cache."""
    out: dict[str, np.ndarray] = {}
    for region, prefix in ((PRE_DATE, "pre"), (POST_DATE, "post")):
        for plane in PLANES:
            key = f"{prefix}_{plane.replace('_label', '')}"
            out[key] = _read_gray(root / "etci" / region / plane / f"{tile_id}.png")
    return out


def build_index(
    root: Path,
    n_tiles: int,
    thresholds: tuple[float, float, float] = DEFAULT_THRESHOLDS,
    seed: int = 42,
    progress: bool = True,
) -> list[TileRecord]:
    """Fetch (if needed) and label ``n_tiles`` real tiles.

    Tiles are sampled with a fixed seed from the full pair list, so growing a tier is
    strictly additive: T2 contains every tile T1 had. That keeps the tier comparison honest,
    because a later tier never trains on an easier sample of the data.
    """
    root.mkdir(parents=True, exist_ok=True)
    cache = root / f"index_{n_tiles}.json"
    if cache.exists():
        return [TileRecord(**r) for r in json.loads(cache.read_text())]

    files = fetch_manifest(root)
    all_ids = build_pair_index(files)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(all_ids))
    chosen = [all_ids[i] for i in order[: min(n_tiles, len(all_ids))]]

    if progress:
        print(f"[etci] requesting {len(chosen)} tiles (of {len(all_ids)} available pairs)")
    got = download_tiles(root, chosen, progress=progress)

    records: list[TileRecord] = []
    for tid in got:
        planes = load_planes(root, tid)
        sev = severity_from_masks(planes["post_flood"], planes["post_water_body"], thresholds)
        records.append(
            TileRecord(tid, sev.label, sev.net_flood_fraction, sev.permanent_water_fraction)
        )
    cache.write_text(json.dumps([r.__dict__ for r in records], indent=2))
    return records


def _stack_sar(vv: np.ndarray, vh: np.ndarray, size: int) -> np.ndarray:
    """Two SAR polarisations plus their ratio -> a 3-channel tensor.

    The VV/VH ratio is a standard derived band: open water suppresses both polarisations but
    not equally, so the ratio carries information neither channel has alone. It also lets us
    feed ImageNet-pretrained 3-channel encoders without wasting a channel.
    """
    import cv2

    vv = cv2.resize(vv, (size, size), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    vh = cv2.resize(vh, (size, size), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    ratio = vv / (vh + 1e-3)
    ratio = np.clip(ratio / 4.0, 0.0, 1.0)
    return np.stack([vv, vh, ratio], axis=0)


class FloodTileDataset(Dataset):
    """Single-date (post-flood) tiles for the four spatial hybrids.

    Yields ``(image[3,H,W], mask[1,H,W], label)``. ``transform`` receives and returns the
    stacked image and is where CLAHE / augmentation is injected.
    """

    def __init__(
        self,
        records: list[TileRecord],
        root: Path | None,
        size: int = 256,
        transform=None,
        synthetic: list[dict] | None = None,
        temporal: bool = False,
    ) -> None:
        self.records = records
        self.root = root
        self.size = size
        self.transform = transform
        self.synthetic = synthetic
        self.temporal = temporal

    def __len__(self) -> int:
        return len(self.synthetic) if self.synthetic is not None else len(self.records)

    def _planes(self, i: int) -> tuple[dict[str, np.ndarray], int]:
        if self.synthetic is not None:
            s = self.synthetic[i]
            return s["planes"], s["severity"].label
        rec = self.records[i]
        assert self.root is not None
        return load_planes(self.root, rec.tile_id), rec.label

    def __getitem__(self, i: int):
        import cv2

        planes, label = self._planes(i)
        post = _stack_sar(planes["post_vv"], planes["post_vh"], self.size)

        mask = cv2.resize(
            planes["post_flood"], (self.size, self.size), interpolation=cv2.INTER_NEAREST
        )
        water = cv2.resize(
            planes["post_water_body"], (self.size, self.size), interpolation=cv2.INTER_NEAREST
        )
        # Target is *net* flood: permanent water is not damage.
        target = ((mask > 0) & ~(water > 0)).astype(np.float32)[None]

        if self.transform is not None:
            post = self.transform(post)

        if self.temporal:
            pre = _stack_sar(planes["pre_vv"], planes["pre_vh"], self.size)
            if self.transform is not None:
                pre = self.transform(pre)
            seq = np.stack([pre, post], axis=0)  # [T=2, C, H, W]
            return torch.from_numpy(seq).float(), torch.from_numpy(target), label

        return torch.from_numpy(post).float(), torch.from_numpy(target), label


def synthetic_dataset(n: int, size: int, seed: int = 0, temporal: bool = False, transform=None):
    """Offline dataset with the identical interface, used by the smoke tier and CI."""
    return FloodTileDataset(
        records=[],
        root=None,
        size=size,
        transform=transform,
        synthetic=make_synthetic(n, size=size, seed=seed),
        temporal=temporal,
    )
