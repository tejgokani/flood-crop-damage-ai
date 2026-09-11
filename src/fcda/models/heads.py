"""The shared output head.

Every hybrid in the benchmark ends in this identical head, which is what makes the
comparison apples-to-apples: only the encoder differs.

It is deliberately dual-output. The segmentation branch produces the damage mask a relief
officer would look at; the classification branch produces the Healthy/Mild/Moderate/Severe
label the problem statement asks for. Training both together is not just convenience --
the segmentation loss is a dense, per-pixel supervisory signal that stabilises the
classification branch, which only ever sees one label per tile.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .. import NUM_CLASSES


class SegSeverityHead(nn.Module):
    """Shared dual head: per-pixel flood mask + per-tile severity class."""

    def __init__(self, in_channels: int, n_classes: int = NUM_CLASSES, dropout: float = 0.1):
        super().__init__()
        self.seg = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(in_channels, 128),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, n_classes),
        )

    def set_dropout(self, p: float) -> None:
        """Used by the over-fitting correction policy to raise regularisation and retrain."""
        for m in self.classifier:
            if isinstance(m, nn.Dropout):
                m.p = p

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.seg(x), self.classifier(x)


class FloodModel(nn.Module):
    """Common wrapper so every architecture exposes the same interface.

    Subclasses implement ``encode`` and set ``self.decoder`` / ``self.head``.
    """

    name: str = "base"
    temporal: bool = False

    def encode(self, x: torch.Tensor) -> list[torch.Tensor]:
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        size = x.shape[-2:]
        seg, cls = self.head(self.decoder(self.encode(x)))
        if seg.shape[-2:] != size:
            # Encoders differ in stem stride, so the decoder lands at different resolutions.
            # Resizing here keeps every architecture's output aligned with the target mask.
            seg = F.interpolate(seg, size=size, mode="bilinear", align_corners=False)
        return seg, cls

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def set_dropout(self, p: float) -> None:
        if hasattr(self.head, "set_dropout"):
            self.head.set_dropout(p)
