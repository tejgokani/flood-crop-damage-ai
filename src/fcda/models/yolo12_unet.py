"""Hybrid 1: YOLO12 + U-Net.

Implemented natively rather than by importing ultralytics. Two reasons: the dependency is
heavy for a component we use only as an encoder, and an AGPL package inside a coursework
repository is a licensing question nobody needs. What we reproduce is the part that matters
for this task -- YOLO12's backbone design:

* **R-ELAN stages** -- residual efficient layer aggregation, which gives many gradient paths
  of differing length without widening the network.
* **Area attention** in the deep stages -- band-wise attention that keeps a global receptive
  field along one axis at a fraction of full self-attention's cost.

The detection side of YOLO is kept as a genuine second head: an anchor-free objectness and
box map over agricultural field parcels. That is what makes this a *hybrid* rather than a
U-Net with a different encoder -- detection localises the parcels, segmentation delineates
the water within them.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .blocks import AreaAttention, ConvBNAct, RELANBlock, UNetDecoder
from .heads import FloodModel, SegSeverityHead


class YOLO12Backbone(nn.Module):
    """Five-stage R-ELAN backbone with area attention in the two deepest stages."""

    def __init__(self, in_channels: int = 3, widths: tuple[int, ...] = (32, 64, 128, 256, 512)):
        super().__init__()
        self.stem = ConvBNAct(in_channels, widths[0], k=3, s=1)
        stages = []
        cin = widths[0]
        for i, w in enumerate(widths):
            layers: list[nn.Module] = []
            if i > 0:
                layers.append(ConvBNAct(cin, w, k=3, s=2))
                cin = w
            layers.append(RELANBlock(cin, w, depth=2 if i < 3 else 3))
            if i >= 3:  # area attention only where the map is small enough to be worth it
                layers.append(AreaAttention(w, num_heads=4, areas=4))
            stages.append(nn.Sequential(*layers))
            cin = w
        self.stages = nn.ModuleList(stages)
        self.channels = list(widths)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        x = self.stem(x)
        feats = []
        for stage in self.stages:
            x = stage(x)
            feats.append(x)
        return feats


class DetectionHead(nn.Module):
    """Anchor-free parcel detection: objectness + box regression on the stride-16 map."""

    def __init__(self, in_channels: int):
        super().__init__()
        self.stem = ConvBNAct(in_channels, in_channels // 2, k=3)
        self.obj = nn.Conv2d(in_channels // 2, 1, 1)
        self.box = nn.Conv2d(in_channels // 2, 4, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.stem(x)
        return self.obj(h), self.box(h)


class YOLO12UNet(FloodModel):
    name = "yolo12_unet"

    def __init__(
        self,
        in_channels: int = 3,
        widths: tuple[int, ...] = (32, 64, 128, 256, 512),
        decoder_channels: tuple[int, ...] = (256, 128, 64, 32, 16),
        dropout: float = 0.1,
        pretrained: bool = False,  # trained from scratch; accepted for a uniform constructor
    ):
        super().__init__()
        self.backbone = YOLO12Backbone(in_channels, widths)
        enc_ch = self.backbone.channels
        self.decoder = UNetDecoder(enc_ch, list(decoder_channels)[: len(enc_ch)])
        self.head = SegSeverityHead(self.decoder.out_channels, dropout=dropout)
        self.detect = DetectionHead(enc_ch[-2])
        self._last_detection: tuple[torch.Tensor, torch.Tensor] | None = None

    def encode(self, x: torch.Tensor) -> list[torch.Tensor]:
        feats = self.backbone(x)
        self._last_detection = self.detect(feats[-2])
        return feats

    @property
    def last_detection(self) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Objectness and box maps from the most recent forward pass."""
        return self._last_detection
