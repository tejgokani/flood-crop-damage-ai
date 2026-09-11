"""Every hybrid must accept every tier resolution and emit aligned outputs."""

from __future__ import annotations

import pytest
import torch

from fcda import NUM_CLASSES
from fcda.models.registry import MODELS, build_model, is_temporal


@pytest.mark.parametrize("name", list(MODELS))
@pytest.mark.parametrize("size", [128, 192, 256])
def test_output_shapes_match_input_resolution(name, size):
    """A decoder emitting the wrong resolution silently breaks the segmentation loss."""
    model = build_model(name, pretrained=False)
    model.eval()
    x = torch.randn(1, 2, 3, size, size) if is_temporal(name) else torch.randn(1, 3, size, size)
    with torch.no_grad():
        seg, cls = model(x)
    assert seg.shape == (1, 1, size, size), f"{name}: segmentation map misaligned with target"
    assert cls.shape == (1, NUM_CLASSES)


@pytest.mark.parametrize("name", list(MODELS))
def test_dropout_is_adjustable(name):
    """The over-fitting correction raises dropout, so it has to actually reach the model."""
    model = build_model(name, pretrained=False)
    model.set_dropout(0.42)
    found = [m.p for m in model.head.classifier if isinstance(m, torch.nn.Dropout)]
    assert found and all(p == 0.42 for p in found)


def test_temporal_model_uses_both_timesteps():
    """CNN+LSTM must respond to the pre-flood frame, or the temporal claim is empty.

    This is a regression test for a real defect. With recurrence only at the bottleneck, the
    three full-resolution skips are identical between timesteps and dilute the temporal
    signal to about 1e-8 at the classifier -- the model was a single-frame CNN in all but
    name. Recurrence now runs at every encoder scale.

    BatchNorm running statistics are warmed first: in eval mode an untrained BatchNorm is
    the identity, activations vanish through twelve convolutions, and the test would measure
    initialisation rather than architecture.
    """
    model = build_model("cnn_lstm", pretrained=False)
    for _ in range(12):
        model(torch.stack([torch.randn(4, 3, 64, 64), torch.randn(4, 3, 64, 64)], dim=1))
    model.eval()

    post = torch.randn(1, 3, 64, 64)
    pre_a, pre_b = torch.randn(1, 3, 64, 64), torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        seg_a, cls_a = model(torch.stack([pre_a, post], dim=1))
        seg_b, cls_b = model(torch.stack([pre_b, post], dim=1))

    cls_rel = (cls_a - cls_b).abs().max() / cls_a.std()
    seg_rel = (seg_a - seg_b).abs().max() / seg_a.std()
    assert cls_rel > 1e-3, f"classifier barely sees the pre-flood frame ({cls_rel:.2e} relative)"
    assert seg_rel > 1e-2, f"segmentation barely sees the pre-flood frame ({seg_rel:.2e} relative)"


def test_severity_head_consumes_predicted_flood_fraction():
    """The classifier must receive the quantity the label is defined from."""
    from fcda.models.heads import SegSeverityHead

    head = SegSeverityHead(16)
    # in_channels*2 (avg + max pooling) + 1 (flood fraction)
    first_linear = next(m for m in head.classifier if isinstance(m, torch.nn.Linear))
    assert first_linear.in_features == 16 * 2 + 1
