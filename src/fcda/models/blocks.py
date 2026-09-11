"""Building blocks shared by the five hybrid architectures.

Keeping the decoder, attention and head implementations in one place is what makes the
benchmark fair: when ResNet+U-Net and Swin+U-Net differ, the difference is the encoder,
because everything downstream is literally the same code.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNAct(nn.Sequential):
    def __init__(self, cin: int, cout: int, k: int = 3, s: int = 1, act: bool = True):
        layers: list[nn.Module] = [
            nn.Conv2d(cin, cout, k, s, k // 2, bias=False),
            nn.BatchNorm2d(cout),
        ]
        if act:
            layers.append(nn.SiLU(inplace=True))
        super().__init__(*layers)


class DoubleConv(nn.Sequential):
    """The standard U-Net pair of 3x3 convolutions."""

    def __init__(self, cin: int, cout: int):
        super().__init__(ConvBNAct(cin, cout), ConvBNAct(cout, cout))


class ChannelAttention(nn.Module):
    """Squeeze-and-excitation style channel gating (the CBAM channel branch)."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden), nn.ReLU(inplace=True), nn.Linear(hidden, channels)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        avg = self.mlp(x.mean(dim=(2, 3)))
        mx = self.mlp(x.amax(dim=(2, 3)))
        return x * torch.sigmoid(avg + mx).view(b, c, 1, 1)


class SpatialAttention(nn.Module):
    """CBAM spatial branch: where in the tile to look."""

    def __init__(self, kernel: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel, padding=kernel // 2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = x.mean(dim=1, keepdim=True)
        mx = x.amax(dim=1, keepdim=True)
        return x * torch.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))


class CBAM(nn.Module):
    """Convolutional Block Attention Module: channel gating then spatial gating.

    This is the attention used by the EfficientNet + Attention hybrid. Paper 5 in our review
    validates exactly this shape -- channel attention followed by texture-guided spatial
    attention -- for agricultural UAV imagery.
    """

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.channel = ChannelAttention(channels, reduction)
        self.spatial = SpatialAttention()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.spatial(self.channel(x))


class AttentionGate(nn.Module):
    """Attention U-Net gate: the decoder signal filters the skip connection.

    Skip connections carry high-resolution detail but also a lot of irrelevant background.
    Gating them on the coarser decoder feature suppresses the background before concatenation,
    which matters here because most of a flood tile is *not* flood.
    """

    def __init__(self, skip_ch: int, gate_ch: int, inter_ch: int | None = None):
        super().__init__()
        inter = inter_ch or max(skip_ch // 2, 8)
        self.w_skip = nn.Conv2d(skip_ch, inter, 1, bias=False)
        self.w_gate = nn.Conv2d(gate_ch, inter, 1, bias=False)
        self.psi = nn.Conv2d(inter, 1, 1)

    def forward(self, skip: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        g = F.interpolate(gate, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        a = torch.sigmoid(self.psi(F.relu(self.w_skip(skip) + self.w_gate(g))))
        return skip * a


class UpBlock(nn.Module):
    """One decoder stage: upsample, optionally gate the skip, concatenate, convolve."""

    def __init__(self, cin: int, skip_ch: int, cout: int, attention: bool = False):
        super().__init__()
        self.up = nn.ConvTranspose2d(cin, cin // 2, 2, 2)
        self.gate = AttentionGate(skip_ch, cin // 2) if (attention and skip_ch > 0) else None
        self.conv = DoubleConv(cin // 2 + skip_ch, cout)

    def forward(self, x: torch.Tensor, skip: torch.Tensor | None) -> torch.Tensor:
        x = self.up(x)
        if skip is not None:
            if self.gate is not None:
                skip = self.gate(skip, x)
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class UNetDecoder(nn.Module):
    """Shared U-Net decoder driven by a list of encoder channel widths.

    Every hybrid in the benchmark uses this same decoder, so encoder comparisons are not
    confounded by decoder differences.
    """

    def __init__(self, encoder_channels: list[int], decoder_channels: list[int], attention: bool = False):
        super().__init__()
        # encoder_channels is shallow -> deep; the decoder walks it in reverse.
        enc = list(encoder_channels)
        # One upsample per skip connection. Adding a block beyond that would take the output
        # past the shallowest encoder feature's resolution, which is how a decoder ends up
        # emitting maps at twice or half the input size depending on the encoder's stem
        # stride. The head resizes to the input resolution instead.
        skips = enc[:-1][::-1]
        cin = enc[-1]
        blocks = []
        for skip_ch, cout in zip(skips, decoder_channels, strict=False):
            blocks.append(UpBlock(cin, skip_ch, cout, attention=attention))
            cin = cout
        self.blocks = nn.ModuleList(blocks)
        self.out_channels = cin

    def forward(self, feats: list[torch.Tensor]) -> torch.Tensor:
        # feats shallow -> deep
        x = feats[-1]
        skips = feats[:-1][::-1]
        for i, blk in enumerate(self.blocks):
            skip = skips[i] if i < len(skips) else None
            x = blk(x, skip)
        return x


class AreaAttention(nn.Module):
    """YOLO12-style area attention.

    Full self-attention over a feature map is quadratic in the number of pixels. YOLO12's
    area attention splits the map into horizontal bands and attends within each, which keeps
    a global receptive field along one axis at a fraction of the cost. That trade is a good
    fit for flood imagery, where inundation spreads in large connected regions.
    """

    def __init__(self, channels: int, num_heads: int = 4, areas: int = 4):
        super().__init__()
        self.num_heads = num_heads
        self.areas = areas
        self.head_dim = max(channels // num_heads, 1)
        self.qkv = nn.Conv2d(channels, channels * 3, 1, bias=False)
        self.proj = nn.Conv2d(channels, channels, 1, bias=False)
        self.scale = self.head_dim**-0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        areas = max(1, min(self.areas, h))
        while h % areas != 0 and areas > 1:
            areas -= 1
        qkv = self.qkv(x).reshape(b, 3, self.num_heads, -1, h, w)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]

        def band(t: torch.Tensor) -> torch.Tensor:
            bb, nh, d, hh, ww = t.shape
            return t.reshape(bb, nh, d, areas, hh // areas * ww).permute(0, 3, 1, 4, 2)

        qb, kb, vb = band(q), band(k), band(v)
        attn = (qb @ kb.transpose(-2, -1)) * self.scale
        out = attn.softmax(dim=-1) @ vb
        out = out.permute(0, 2, 4, 1, 3).reshape(b, -1, h, w)
        return x + self.proj(out)


class RELANBlock(nn.Module):
    """Residual ELAN block, the YOLO12 backbone stage.

    ELAN splits the channels, runs a chain of convolutions on one half, and concatenates all
    intermediate outputs. That gives many gradient paths of differing length without the
    parameter cost of widening, which is why YOLO backbones stay small.
    """

    def __init__(self, cin: int, cout: int, depth: int = 2):
        super().__init__()
        hidden = cout // 2
        self.entry = ConvBNAct(cin, hidden * 2, k=1)
        self.chain = nn.ModuleList([ConvBNAct(hidden, hidden) for _ in range(depth)])
        self.exit = ConvBNAct(hidden * (2 + depth), cout, k=1)
        self.shortcut = (
            nn.Identity() if cin == cout else ConvBNAct(cin, cout, k=1, act=False)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.shortcut(x)
        a, b = self.entry(x).chunk(2, dim=1)
        outs = [a, b]
        h = b
        for blk in self.chain:
            h = blk(h)
            outs.append(h)
        return self.exit(torch.cat(outs, dim=1)) + res
