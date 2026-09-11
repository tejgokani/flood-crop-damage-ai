"""Hybrid 5: CNN + LSTM (temporal).

This is the hybrid our literature review most directly demands. Three of the six reviewed
papers ask for it in their own words:

* Paper 4: "incorporating multi-temporal SAR data, in which pre-flood and post-flood images
  are jointly analyzed, could help models better distinguish temporary flooding from
  permanent water bodies."
* Paper 2: "combining temporal modeling to capture changing patterns of floods."
* Paper 3: "integrating time-series imagery."

The input is the ordered pair (pre-monsoon 2017-03-14, monsoon 2017-06-06) for the same
tile. A shared CNN encoder embeds each timestep, a ConvLSTM carries state across them, and
the decoder reads the final hidden state.

Why a *Conv*LSTM rather than flattening to a vector LSTM: flood damage is a spatial
phenomenon, and a vector LSTM would discard the layout that the segmentation head needs.
The ConvLSTM keeps the feature map's spatial structure inside the recurrence.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .blocks import ConvBNAct, DoubleConv, UNetDecoder
from .heads import FloodModel, SegSeverityHead


class ConvLSTMCell(nn.Module):
    """One ConvLSTM step: gates computed by convolution, so state stays spatial."""

    def __init__(self, in_channels: int, hidden_channels: int, kernel: int = 3):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.conv = nn.Conv2d(
            in_channels + hidden_channels, hidden_channels * 4, kernel, padding=kernel // 2
        )
        # Standard LSTM practice: bias the forget gate open at initialisation. Left at zero
        # the gate sits at sigmoid(0) = 0.5 and the cell state decays by half every step, so
        # the pre-flood frame is already half-forgotten by the time the post frame arrives.
        with torch.no_grad():
            self.conv.bias[hidden_channels : 2 * hidden_channels].fill_(1.0)

    def forward(
        self, x: torch.Tensor, state: tuple[torch.Tensor, torch.Tensor] | None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        b, _, h, w = x.shape
        if state is None:
            zeros = torch.zeros(b, self.hidden_channels, h, w, device=x.device, dtype=x.dtype)
            state = (zeros, zeros)
        h_prev, c_prev = state
        gates = self.conv(torch.cat([x, h_prev], dim=1))
        i, f, o, g = gates.chunk(4, dim=1)
        i, f, o = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o)
        c = f * c_prev + i * torch.tanh(g)
        return o * torch.tanh(c), c


class CNNEncoder(nn.Module):
    """Compact CNN encoder applied identically to every timestep (shared weights)."""

    def __init__(self, in_channels: int = 3, widths: tuple[int, ...] = (32, 64, 128, 256)):
        super().__init__()
        stages = []
        cin = in_channels
        for i, w in enumerate(widths):
            stages.append(
                nn.Sequential(
                    ConvBNAct(cin, w, k=3, s=1 if i == 0 else 2),
                    DoubleConv(w, w),
                )
            )
            cin = w
        self.stages = nn.ModuleList(stages)
        self.channels = list(widths)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        feats = []
        for stage in self.stages:
            x = stage(x)
            feats.append(x)
        return feats


class CNNLSTM(FloodModel):
    name = "cnn_lstm"
    temporal = True

    def __init__(
        self,
        in_channels: int = 3,
        widths: tuple[int, ...] = (32, 64, 128, 256),
        decoder_channels: tuple[int, ...] = (128, 64, 32),
        dropout: float = 0.3,
        pretrained: bool = False,  # accepted for a uniform constructor signature
    ):
        super().__init__()
        self.encoder = CNNEncoder(in_channels, widths)
        enc_ch = self.encoder.channels
        # One ConvLSTM per encoder scale, not only at the bottleneck.
        #
        # With recurrence at the deepest scale alone, the temporal signal exists in exactly
        # one of the four feature maps handed to the decoder, and the three full-resolution
        # skips -- identical between the two timesteps -- dilute it away: measured end to
        # end, a 16% relative difference at the bottleneck arrived at the classifier as
        # 1e-8, i.e. the network was a single-frame CNN wearing an LSTM. Running the
        # recurrence at every scale means every skip carries pre-to-post change, so the
        # temporal claim holds structurally rather than by hope.
        self.lstms = nn.ModuleList([ConvLSTMCell(c, c) for c in enc_ch])
        self.decoder = UNetDecoder(enc_ch, list(decoder_channels)[: len(enc_ch) - 1])
        self.head = SegSeverityHead(self.decoder.out_channels, dropout=dropout)

    def encode(self, x: torch.Tensor) -> list[torch.Tensor]:
        """``x`` is ``[B, T, C, H, W]``; a ``[B, C, H, W]`` input is treated as T=1."""
        if x.ndim == 4:
            x = x[:, None]
        t = x.shape[1]
        states: list[tuple[torch.Tensor, torch.Tensor] | None] = [None] * len(self.lstms)

        for step in range(t):
            feats = self.encoder(x[:, step])
            for i, (cell, f) in enumerate(zip(self.lstms, feats, strict=True)):
                states[i] = cell(f, states[i])

        # Every level is now a recurrent state over the pre/post sequence.
        return [s[0] for s in states]

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        import torch.nn.functional as F

        size = x.shape[-2:]
        seg, cls = self.head(self.decoder(self.encode(x)))
        if seg.shape[-2:] != size:
            seg = F.interpolate(seg, size=size, mode="bilinear", align_corners=False)
        return seg, cls
