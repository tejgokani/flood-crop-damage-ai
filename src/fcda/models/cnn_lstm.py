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


class ChangeFusion(nn.Module):
    """Cheap temporal fusion for the shallow scales: concat(post, post - pre) -> 1x1 conv.

    Recurrence at full resolution is the single most expensive operation in this model --
    measured at 109 ms for a 0.07M-parameter cell, because it is bandwidth-bound at 192x192 --
    and it is also where recurrence earns the least: fine texture does not need memory, it needs
    to know what changed. An explicit difference gives the skip connection its temporal signal
    at a fraction of the cost, which buys the epochs the model was being starved of.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.mix = ConvBNAct(channels * 2, channels, k=1)

    def forward(self, pre: torch.Tensor, post: torch.Tensor) -> torch.Tensor:
        return self.mix(torch.cat([post, post - pre], dim=1))


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
        lstm_scales: int = 2,
        pretrained: bool = False,  # accepted for a uniform constructor signature
    ):
        super().__init__()
        self.encoder = CNNEncoder(in_channels, widths)
        enc_ch = self.encoder.channels

        # Every scale carries temporal information, but not all of it needs recurrence.
        #
        # Recurrence at the deepest scale *alone* was the original bug: the shallow skips are
        # identical between timesteps and diluted the signal from 16% relative at the bottleneck
        # to 1e-8 at the classifier, making this a single-frame CNN wearing an LSTM. Running a
        # ConvLSTM at all four scales fixed that but cost 356 ms per forward pass, of which 173 ms
        # went to the two shallowest scales -- and the model was then wall-clock-capped at nine
        # epochs with its loss still falling.
        #
        # So: ConvLSTM on the deep scales, where state across time is worth carrying, and an
        # explicit difference fusion on the shallow ones, where the useful signal is simply what
        # changed. Both paths keep every skip temporal.
        self.recurrent_from = max(0, len(enc_ch) - lstm_scales)
        self.lstms = nn.ModuleList([
            ConvLSTMCell(c, c) if i >= self.recurrent_from else nn.Identity()
            for i, c in enumerate(enc_ch)
        ])
        self.fusions = nn.ModuleList([
            ChangeFusion(c) if i < self.recurrent_from else nn.Identity()
            for i, c in enumerate(enc_ch)
        ])
        self.decoder = UNetDecoder(enc_ch, list(decoder_channels)[: len(enc_ch) - 1])
        self.head = SegSeverityHead(self.decoder.out_channels, dropout=dropout)

    def encode(self, x: torch.Tensor) -> list[torch.Tensor]:
        """``x`` is ``[B, T, C, H, W]``; a ``[B, C, H, W]`` input is treated as T=1."""
        if x.ndim == 4:
            x = x[:, None]
        t = x.shape[1]
        states: list[tuple[torch.Tensor, torch.Tensor] | None] = [None] * len(self.lstms)
        first_feats: list[torch.Tensor] = []
        last_feats: list[torch.Tensor] = []

        for step in range(t):
            feats = self.encoder(x[:, step])
            if step == 0:
                first_feats = feats
            last_feats = feats
            for i in range(self.recurrent_from, len(self.lstms)):
                states[i] = self.lstms[i](feats[i], states[i])

        out: list[torch.Tensor] = []
        for i in range(len(self.lstms)):
            if i >= self.recurrent_from:
                out.append(states[i][0])          # recurrent state over the sequence
            else:
                pre = first_feats[i] if t > 1 else last_feats[i]
                out.append(self.fusions[i](pre, last_feats[i]))   # explicit pre->post change
        return out

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        import torch.nn.functional as F

        size = x.shape[-2:]
        seg, cls = self.head(self.decoder(self.encode(x)))
        if seg.shape[-2:] != size:
            seg = F.interpolate(seg, size=size, mode="bilinear", align_corners=False)
        return seg, cls
