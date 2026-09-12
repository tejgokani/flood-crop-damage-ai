"""v2 dataset: all real ETCI pairs, plus procedural synthetic for training only.

The whole design rests on one separation:

* **Real tiles are the only thing ever scored.** Every number in the v2 report comes from real
  ETCI tiles evaluated out of fold.
* **Synthetic tiles exist only inside the training split.** They are generated per fold from
  that fold's seed and never enter validation or test.

``assert_no_synthetic_in_eval`` makes that checkable rather than trusted, and the tests call it.
Without that boundary "expanding the dataset" and "inventing a result" look identical from the
outside.

v1 used 900 of the 2,032 available ETCI pairs because a single split wasted 15% on test and the
tier ladder capped the rest. v2 uses all of them, and commissions synthetic tiles to fill the
classes the corpus cannot supply -- above all Severe, which has 20 real examples.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from fcda import NUM_CLASSES, SEVERITY_CLASSES
from fcda.data.etci import _stack_sar, load_planes
from fcda.data.severity import severity_from_masks
from fcda.preprocess.transforms import apply_geometric_pair

from .procedural import make_for_class


@dataclass
class TileRef:
    """A pointer to one sample. ``synthetic`` decides which loader is used."""

    key: str
    label: int
    synthetic: bool = False


def _refs(path: Path) -> list[TileRef]:
    return [TileRef(r["tile_id"], int(r["label"]), False) for r in json.loads(path.read_text())]


def load_benchmark(data_root: Path) -> list[TileRef]:
    """The evaluation benchmark: v1's stratified 900-tile set.

    Deliberately *not* the full 2,032-tile pool, and the reason is a measurement rather than a
    preference. The stratified set already contains **every** Mild, Moderate and Severe tile in
    the corpus -- the extra 1,132 tiles in the full pool are 100% Healthy. Evaluating on the full
    pool therefore adds no minority data, raises the always-Healthy baseline from 0.703 to 0.869,
    and drops macro-F1 from 0.462 to 0.354 purely by changing the test distribution. That is a
    different question being answered, not a worse model.

    Keeping v1's benchmark means v2's numbers are comparable to v1's. The extra Healthy tiles are
    still useful -- as *training* data, via ``load_extra_healthy``.
    """
    index = data_root / "index_900_strat.json"
    if not index.exists():
        raise FileNotFoundError(f"{index} not found -- run the v1 downloader first")
    return _refs(index)


def load_extra_healthy(data_root: Path) -> list[TileRef]:
    """Real Healthy tiles outside the benchmark. Training-only, like the synthetic ones.

    They cannot sharpen the minority classes, but more real negatives do help Healthy precision,
    and they are real -- no domain gap to pay for.
    """
    pool_path = data_root / "pool_index.json"
    if not pool_path.exists():
        return []
    bench = {r.key for r in load_benchmark(data_root)}
    return [r for r in _refs(pool_path) if r.key not in bench]


def load_real_pool(data_root: Path) -> list[TileRef]:
    """Backwards-compatible alias for the evaluation benchmark."""
    return load_benchmark(data_root)


def class_counts(refs: list[TileRef]) -> dict[str, int]:
    return {
        SEVERITY_CLASSES[c]: sum(1 for r in refs if r.label == c) for c in range(NUM_CLASSES)
    }


def plan_synthetic(
    real_train: list[TileRef],
    per_class_target: int = 600,
    total_cap: int | None = None,
) -> dict[int, int]:
    """How many synthetic tiles of each class to commission for one training split.

    Tops each class up to ``per_class_target`` real-plus-synthetic examples. Healthy is already
    far above it, so in practice this buys Mild, Moderate and above all Severe -- the class the
    corpus caps at 20 and the one that binds macro-F1.

    Deliberately *not* levelled all the way up to the Healthy count (1,765). Doing so would
    commission ~5,000 synthetic tiles against 2,032 real ones, so most of what the model saw
    would be generated -- trading a class-imbalance problem for a domain-gap one, which is very
    close to the mistake the GAN made in v1.
    """
    counts = np.bincount([r.label for r in real_train], minlength=NUM_CLASSES)
    plan = {c: max(0, int(per_class_target - counts[c])) for c in range(NUM_CLASSES)}
    if total_cap is not None and sum(plan.values()) > total_cap:
        scale = total_cap / sum(plan.values())
        plan = {c: int(v * scale) for c, v in plan.items()}
    return plan


def build_synthetic(counts: dict[int, int], seed: int) -> list[TileRef]:
    """Deterministic synthetic refs. The key encodes class and seed, so it is reproducible."""
    out: list[TileRef] = []
    n = 0
    for label, want in sorted(counts.items()):
        for _ in range(want):
            out.append(TileRef(f"syn:{label}:{seed}:{n}", label, True))
            n += 1
    return out


def assert_no_synthetic_in_eval(refs: list[TileRef], split_name: str) -> None:
    """Evaluation splits must contain real tiles only."""
    bad = [r.key for r in refs if r.synthetic]
    if bad:
        raise AssertionError(
            f"{len(bad)} synthetic tiles found in the '{split_name}' split "
            f"(e.g. {bad[:3]}); synthetic data is training-only"
        )


class FloodDataset(Dataset):
    """Yields ``(image, mask, label, fraction)`` for any of the three v2 input layouts.

    ``mode`` selects the layout:
      * ``"change"``   -> ``[post(3), post - pre(3)]``   (YOLO12 + U-Net)
      * ``"temporal"`` -> ``[pre, post]`` as ``[T=2,3,H,W]`` (CNN + LSTM)
      * ``"single"``   -> ``post(3)``                     (EfficientNet + Attention)
    """

    def __init__(
        self,
        refs: list[TileRef],
        data_root: Path,
        size: int = 256,
        mode: str = "change",
        transform=None,
    ) -> None:
        self.refs = refs
        self.data_root = data_root
        self.size = size
        self.mode = mode
        self.transform = transform

    def __len__(self) -> int:
        return len(self.refs)

    def _planes(self, ref: TileRef) -> dict[str, np.ndarray]:
        if not ref.synthetic:
            return load_planes(self.data_root, ref.key)
        _, label, seed, n = ref.key.split(":")
        return make_for_class(int(seed) * 1_000_003 + int(n), int(label), size=self.size).as_planes()

    def __getitem__(self, i: int):
        import cv2

        ref = self.refs[i]
        planes = self._planes(ref)

        post = _stack_sar(planes["post_vv"], planes["post_vh"], self.size)
        pre = _stack_sar(planes["pre_vv"], planes["pre_vh"], self.size)

        flood = cv2.resize(planes["post_flood"], (self.size, self.size), interpolation=cv2.INTER_NEAREST)
        water = cv2.resize(planes["post_water_body"], (self.size, self.size), interpolation=cv2.INTER_NEAREST)
        target = ((flood > 0) & ~(water > 0)).astype(np.float32)[None]

        if self.transform is not None:
            post, pre, target = apply_geometric_pair(self.transform, post, pre, target)

        # Computed after augmentation: a crop changes how much of the tile is under water.
        fraction = np.float32(target.mean())

        if self.mode == "temporal":
            img = np.stack([pre, post], axis=0)
        elif self.mode == "change":
            img = np.concatenate([post, post - pre], axis=0)
        else:
            img = post

        return (
            torch.from_numpy(np.ascontiguousarray(img)).float(),
            torch.from_numpy(target),
            int(ref.label),
            fraction,
        )


def verify_synthetic_labels(n_per_class: int = 50, size: int = 128) -> dict[str, float]:
    """Confirm commissioned tiles land in the class they were asked for.

    A generator that drifts would quietly relabel the training set, so this is checked rather
    than assumed -- it is the synthetic equivalent of a leakage check.
    """
    out: dict[str, float] = {}
    for label in range(NUM_CLASSES):
        hits = 0
        for i in range(n_per_class):
            p = make_for_class(i, label, size=size).as_planes()
            if severity_from_masks(p["post_flood"], p["post_water_body"]).label == label:
                hits += 1
        out[SEVERITY_CLASSES[label]] = hits / n_per_class
    return out
