"""Hybrid 4: Swin Transformer + U-Net.

Motivated directly by the reviewed literature. FLNet's future work asks for "state-of-the-art
backbones like Transformers"; Paper 2 builds a Swin encoder for flood scenes and asks for it
to be applied to satellite imagery with SAR input, which is what we do here.

The shifted-window mechanism is the reason this backbone suits flood segmentation: self-
attention is computed inside local windows, and the windows shift between blocks so
information crosses window boundaries. That yields a global receptive field at linear rather
than quadratic cost -- the "long-range dependency" property Paper 2 argues is needed for
precise flood-boundary delineation.

Two practical adaptations:

* timm's Swin emits **NHWC** feature maps; the shared decoder expects NCHW, so features are
  permuted on the way out.
* Window partitioning fixes the input resolution at 256. Smaller tiers (128, 192 px) are
  bilinearly resized up for the encoder and the decoder output is resized back, so the model
  participates in every tier of the ladder rather than only the largest.
"""

from __future__ import annotations

import timm
import torch
import torch.nn.functional as F

from .blocks import UNetDecoder
from .heads import FloodModel, SegSeverityHead

NATIVE_SIZE = 256


class SwinUNet(FloodModel):
    name = "swin_unet"

    def __init__(
        self,
        encoder: str = "swinv2_tiny_window8_256",
        pretrained: bool = True,
        in_channels: int = 3,
        decoder_channels: tuple[int, ...] = (256, 128, 64, 32),
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
        feats = self.encoder(x)
        # timm's Swin returns NHWC; the shared decoder is NCHW throughout.
        return [f.permute(0, 3, 1, 2).contiguous() if f.ndim == 4 and f.shape[-1] != x.shape[1] else f
                for f in feats]

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h, w = x.shape[-2:]
        resized = (h, w) != (NATIVE_SIZE, NATIVE_SIZE)
        if resized:
            x = F.interpolate(x, size=(NATIVE_SIZE, NATIVE_SIZE), mode="bilinear", align_corners=False)
        seg, cls = self.head(self.decoder(self.encode(x)))
        seg = F.interpolate(seg, size=(h, w), mode="bilinear", align_corners=False)
        return seg, cls
