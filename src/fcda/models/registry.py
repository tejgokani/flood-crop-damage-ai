"""Model registry: one name -> one architecture, used by the CLI and the training loop."""

from __future__ import annotations

from .cnn_lstm import CNNLSTM
from .effnet_attention import EfficientNetAttention
from .heads import FloodModel
from .resnet_unet import ResNetUNet
from .swin_unet import SwinUNet
from .yolo12_unet import YOLO12UNet

#: The two hybrids carried forward for depth.
#:
#: All five named in Problem Statement 6 were implemented and benchmarked; their T1/T2 results
#: are archived in reports/results_5model_baseline.json. Two were selected to be developed
#: properly rather than five left shallow:
#:
#:   * CNN + LSTM was the only model of the five that did not overfit (train-val gap +0.093
#:     against +0.19 to +0.35 for the rest), because the pre/post pair acts as a regulariser.
#:   * YOLO12 + U-Net was the weakest, and its failure mode was diagnosable rather than
#:     mysterious -- trained from scratch with no ImageNet initialisation, it peaked at epoch 1.
#:
#: They are also architecturally complementary: one reaches the pre-to-post change through an
#: explicit difference channel, the other through recurrence.
MODELS: dict[str, type[FloodModel]] = {
    "yolo12_unet": YOLO12UNet,
    "cnn_lstm": CNNLSTM,
}

#: Implemented and benchmarked, retained for reference but no longer trained by default.
BASELINE_MODELS: dict[str, type[FloodModel]] = {
    "resnet_unet": ResNetUNet,
    "effnet_attention": EfficientNetAttention,
    "swin_unet": SwinUNet,
}

DISPLAY_NAMES = {
    "yolo12_unet": "YOLO12 + U-Net",
    "resnet_unet": "ResNet + U-Net",
    "effnet_attention": "EfficientNet + Attention",
    "swin_unet": "Swin Transformer + U-Net",
    "cnn_lstm": "CNN + LSTM",
}


ALL_MODELS: dict[str, type[FloodModel]] = {**MODELS, **BASELINE_MODELS}


def build_model(name: str, **kwargs) -> FloodModel:
    if name not in ALL_MODELS:
        raise KeyError(f"Unknown model {name!r}. Available: {list(ALL_MODELS)}")
    return ALL_MODELS[name](**kwargs)


def is_temporal(name: str) -> bool:
    """True when the model consumes an ordered [B, T, C, H, W] sequence."""
    return bool(getattr(ALL_MODELS[name], "temporal", False))


def wants_change_input(name: str) -> bool:
    """True when the model takes the 6-channel post + (post - pre) difference stack.

    The temporal model does not: it receives both frames and derives the change itself, and
    handing it a pre-computed difference as well would erase the architectural distinction
    that makes comparing the two worthwhile.
    """
    return bool(getattr(ALL_MODELS[name], "change_input", False))


def all_model_names() -> list[str]:
    return list(MODELS)
