"""Model registry: one name -> one architecture, used by the CLI and the training loop."""

from __future__ import annotations

from .cnn_lstm import CNNLSTM
from .effnet_attention import EfficientNetAttention
from .heads import FloodModel
from .resnet_unet import ResNetUNet
from .swin_unet import SwinUNet
from .yolo12_unet import YOLO12UNet

#: The five hybrids named in Problem Statement 6, in the order the statement lists them.
MODELS: dict[str, type[FloodModel]] = {
    "yolo12_unet": YOLO12UNet,
    "resnet_unet": ResNetUNet,
    "effnet_attention": EfficientNetAttention,
    "swin_unet": SwinUNet,
    "cnn_lstm": CNNLSTM,
}

DISPLAY_NAMES = {
    "yolo12_unet": "YOLO12 + U-Net",
    "resnet_unet": "ResNet + U-Net",
    "effnet_attention": "EfficientNet + Attention",
    "swin_unet": "Swin Transformer + U-Net",
    "cnn_lstm": "CNN + LSTM",
}


def build_model(name: str, **kwargs) -> FloodModel:
    if name not in MODELS:
        raise KeyError(f"Unknown model {name!r}. Available: {list(MODELS)}")
    return MODELS[name](**kwargs)


def is_temporal(name: str) -> bool:
    return bool(getattr(MODELS[name], "temporal", False))


def all_model_names() -> list[str]:
    return list(MODELS)
