"""Hybrid 2: ResNet + U-Net.

The baseline of the benchmark and the most established combination in the segmentation
literature. A ResNet encoder supplies ImageNet-pretrained features and residual connections
that keep gradients healthy at depth; the U-Net decoder restores resolution through skip
connections. Everything after the encoder is the shared decoder and head.
"""

from __future__ import annotations

import timm
import torch

from .blocks import UNetDecoder
from .heads import FloodModel, SegSeverityHead


class ResNetUNet(FloodModel):
    name = "resnet_unet"

    def __init__(
        self,
        encoder: str = "resnet34",
        pretrained: bool = True,
        in_channels: int = 3,
        decoder_channels: tuple[int, ...] = (256, 128, 64, 32, 16),
        dropout: float = 0.1,
    ):
        super().__init__()
        self.encoder = timm.create_model(
            encoder, pretrained=pretrained, features_only=True, in_chans=in_channels
        )
        enc_ch = self.encoder.feature_info.channels()
        self.decoder = UNetDecoder(enc_ch, list(decoder_channels)[: len(enc_ch)])
        self.head = SegSeverityHead(self.decoder.out_channels, dropout=dropout)

    def encode(self, x: torch.Tensor) -> list[torch.Tensor]:
        return self.encoder(x)
