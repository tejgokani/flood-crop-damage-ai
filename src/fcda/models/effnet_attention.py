"""Hybrid 3: EfficientNet + Attention.

EfficientNet's compound scaling gives a strong accuracy-per-FLOP ratio, which matters given
that Papers 3 and 5 both name compute as a barrier to deployment. The attention here is two
layered mechanisms rather than one:

* **CBAM** on each encoder feature map -- channel gating then spatial gating.
* **Attention gates** on every skip connection, so the decoder filters the encoder's detail
  rather than accepting all of it.

Paper 5 (TDAVM-UNet) validates this exact shape for agricultural UAV imagery, pairing
multi-scale channel attention with texture-guided spatial attention.
"""

from __future__ import annotations

import timm
import torch
import torch.nn as nn

from .blocks import CBAM, UNetDecoder
from .heads import FloodModel, SegSeverityHead


class EfficientNetAttention(FloodModel):
    name = "effnet_attention"

    def __init__(
        self,
        encoder: str = "efficientnet_b0",
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
        self.cbam = nn.ModuleList([CBAM(c) for c in enc_ch])
        self.decoder = UNetDecoder(enc_ch, list(decoder_channels)[: len(enc_ch)], attention=True)
        self.head = SegSeverityHead(self.decoder.out_channels, dropout=dropout)

    def encode(self, x: torch.Tensor) -> list[torch.Tensor]:
        return [att(f) for att, f in zip(self.cbam, self.encoder(x), strict=False)]
