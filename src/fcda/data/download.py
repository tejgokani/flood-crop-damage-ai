"""On-demand fetching of ETCI-2021 tiles and the Indian agricultural CSV.

Nothing is bulk-downloaded. The HuggingFace manifest is cached once (~4 MB of JSON), and
individual PNG tiles are pulled only when a tier actually asks for them. This keeps the
footprint proportional to the tier we reached rather than to the size of the corpus, which
matters on a machine with limited free disk.

Sources (both public, no authentication required):
  * ``blanchon/ETCI-2021-Flood-Detection`` -- Sentinel-1 SAR flood tiles.
  * ``nileshely/Crop-Datasets-for-All-Indian-States`` -- ICRISAT district crop statistics.
"""

from __future__ import annotations

import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
from pathlib import Path

import requests

ETCI_REPO = "blanchon/ETCI-2021-Flood-Detection"
ETCI_API = f"https://huggingface.co/api/datasets/{ETCI_REPO}?full=true"
ETCI_RESOLVE = f"https://huggingface.co/datasets/{ETCI_REPO}/resolve/main/"

INDIA_CSV_URL = (
    "https://raw.githubusercontent.com/nileshely/"
    "Crop-Datasets-for-All-Indian-States/main/Crops_data.csv"
)

#: The two Bangladesh acquisitions that carry all four raster planes.
#: 2017-03-14 is pre-monsoon (dry); 2017-06-06 is the monsoon flood.
PRE_DATE = "bangladesh_20170314t115609"
POST_DATE = "bangladesh_20170606t115613"
PLANES = ("vv", "vh", "flood_label", "water_body_label")

_XY_RE = re.compile(r"_(x-\d+_y-\d+)(?:_vv|_vh)?$")


class DownloadError(RuntimeError):
    """Raised when a remote source cannot be reached."""


def manifest_path(root: Path) -> Path:
    return root / "etci_manifest.json"


def fetch_manifest(root: Path, timeout: int = 90, force: bool = False) -> list[str]:
    """Fetch (and cache) the list of every file in the ETCI repo."""
    root.mkdir(parents=True, exist_ok=True)
    path = manifest_path(root)
    if path.exists() and not force:
        return json.loads(path.read_text())
    try:
        resp = requests.get(ETCI_API, timeout=timeout)
        resp.raise_for_status()
        files = [s["rfilename"] for s in resp.json()["siblings"]]
    except Exception as exc:  # noqa: BLE001 - surfaced to the caller as a clean failure
        raise DownloadError(f"could not fetch ETCI manifest: {exc}") from exc
    path.write_text(json.dumps(files))
    return files


def build_pair_index(files: list[str]) -> list[str]:
    """Return the tile ids (``x-N_y-M``) that are complete at *both* dates.

    A tile qualifies only if all four planes exist for the pre-flood and the post-flood
    acquisition, so every sample supports single-date models and the temporal model alike.
    """
    have: dict[tuple[str, str], set[str]] = defaultdict(set)
    for f in files:
        parts = f.split("/")
        if len(parts) < 6 or parts[1] != "train":
            continue
        region, plane, stem = parts[2], parts[4], parts[5].removesuffix(".png")
        m = _XY_RE.search(stem)
        if m:
            have[(region, m.group(1))].add(plane)
    need = set(PLANES)
    pre = {xy for (r, xy), p in have.items() if r == PRE_DATE and need <= p}
    post = {xy for (r, xy), p in have.items() if r == POST_DATE and need <= p}
    return sorted(pre & post)


def tile_url(region: str, plane: str, tile_id: str) -> str:
    """Build the resolve URL for one raster plane of one tile."""
    suffix = f"_{plane}" if plane in ("vv", "vh") else ""
    return f"{ETCI_RESOLVE}data/train/{region}/tiles/{plane}/{region}_{tile_id}{suffix}.png"


def local_tile_path(root: Path, region: str, plane: str, tile_id: str) -> Path:
    return root / "etci" / region / plane / f"{tile_id}.png"


def download_tile_plane(
    root: Path, region: str, plane: str, tile_id: str, retries: int = 3, timeout: int = 45
) -> Path:
    """Download one plane of one tile, skipping the request if it is already cached."""
    dest = local_tile_path(root, region, plane, tile_id)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = tile_url(region, plane, tile_id)
    last: Exception | None = None
    for attempt in range(retries):
        try:
            resp = requests.get(url, timeout=timeout)
            resp.raise_for_status()
            dest.write_bytes(resp.content)
            return dest
        except Exception as exc:  # noqa: BLE001 - retried, then reported
            last = exc
            time.sleep(1.5 * (attempt + 1))
    raise DownloadError(f"failed to download {url}: {last}")


def download_tiles(
    root: Path, tile_ids: list[str], progress: bool = True, workers: int = 16
) -> list[str]:
    """Fetch every plane of every requested tile at both dates, in parallel.

    Each tile needs eight small PNGs, so the job is latency-bound rather than
    bandwidth-bound: serially this runs at well under one file per second, which would put a
    900-tile tier out of reach in a single session. A modest thread pool turns that into
    minutes. Individual failures are skipped rather than fatal, so a flaky connection
    degrades the tier size instead of killing the run.
    """
    tasks = [
        (region, plane, tid)
        for tid in tile_ids
        for region in (PRE_DATE, POST_DATE)
        for plane in PLANES
    ]
    failed: set[str] = set()
    done = 0
    total = len(tasks)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(download_tile_plane, root, region, plane, tid): tid
            for region, plane, tid in tasks
        }
        for fut in as_completed(futures):
            tid = futures[fut]
            try:
                fut.result()
            except DownloadError:
                failed.add(tid)
            done += 1
            if progress and (done % 400 == 0 or done == total):
                print(
                    f"  [download] {done}/{total} files  ({len(failed)} tiles failed)",
                    flush=True,
                )

    return [t for t in tile_ids if t not in failed]


def download_india_csv(root: Path, timeout: int = 90, force: bool = False) -> Path:
    """Fetch the ICRISAT district-level Indian crop statistics CSV."""
    root.mkdir(parents=True, exist_ok=True)
    dest = root / "india_crops.csv"
    if dest.exists() and dest.stat().st_size > 0 and not force:
        return dest
    try:
        resp = requests.get(INDIA_CSV_URL, timeout=timeout)
        resp.raise_for_status()
        dest.write_bytes(resp.content)
    except Exception as exc:  # noqa: BLE001
        raise DownloadError(f"could not fetch Indian crop CSV: {exc}") from exc
    return dest
