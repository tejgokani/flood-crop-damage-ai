"""CLAHE contrast enhancement -- step 4 of Ma'am's image sequence.

Contrast-Limited Adaptive Histogram Equalisation equalises within small tiles rather than
globally, and clips the histogram before equalising so that flat regions do not have their
noise amplified into false texture.

It earns its place here specifically because of SAR: Sentinel-1 amplitude images are
dominated by multiplicative speckle and most of a flood scene sits in a narrow, dark band of
the dynamic range. Global equalisation would blow out the speckle; CLAHE lifts the local
contrast at the flood boundary, which is the part of the image the segmentation head has to
get right.

Applied to the SAR polarisation channels only. The derived VV/VH ratio channel is left alone
-- it is a physical quantity, and equalising it would destroy the ratio's meaning.
"""

from __future__ import annotations

import cv2
import numpy as np


class CLAHE:
    """Callable CLAHE transform over a ``[C, H, W]`` float image in ``[0, 1]``."""

    def __init__(
        self, clip_limit: float = 2.0, tile_grid: int = 8, channels: tuple[int, ...] = (0, 1)
    ) -> None:
        self.clip_limit = clip_limit
        self.tile_grid = tile_grid
        self.channels = channels
        self._op = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid, tile_grid))

    def __call__(self, img: np.ndarray) -> np.ndarray:
        out = img.copy()
        for c in self.channels:
            if c >= out.shape[0]:
                continue
            plane = np.clip(out[c] * 255.0, 0, 255).astype(np.uint8)
            out[c] = self._op.apply(plane).astype(np.float32) / 255.0
        return out

    def __repr__(self) -> str:
        return f"CLAHE(clip_limit={self.clip_limit}, tile_grid={self.tile_grid})"


def apply_clahe_uint8(gray: np.ndarray, clip_limit: float = 2.0, tile_grid: int = 8) -> np.ndarray:
    """Convenience wrapper for a single 8-bit plane (used by the Streamlit app)."""
    op = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid, tile_grid))
    return op.apply(gray)
