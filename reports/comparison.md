# Model comparison

| Hybrid architecture | Params | Test macro-F1 | Accuracy | Cohen's κ | Flood IoU | Dice | Diagnosis | Correction helped |
|---|---:|---:|---:|---:|---:|---:|---|:--:|
| EfficientNet + Attention | 5.7M | **0.507** | 0.667 | 0.500 | 0.206 | 0.342 | OVERFIT | no |
| ResNet + U-Net | 24.4M | **0.500** | 0.683 | 0.511 | 0.222 | 0.364 | OVERFIT | no |
| CNN + LSTM | 9.0M | **0.484** | 0.667 | 0.500 | 0.195 | 0.327 | OK | — |
| Swin Transformer + U-Net | 31.9M | **0.436** | 0.600 | 0.414 | 0.241 | 0.388 | OVERFIT | yes |
| YOLO12 + U-Net | 9.8M | **0.342** | 0.400 | 0.192 | 0.158 | 0.273 | OVERFIT | no |

## Per-class F1

| Hybrid | Healthy | Mild | Moderate | Severe |
|---|---:|---:|---:|---:|
| YOLO12 + U-Net | 0.553 | 0.417 | 0.400 | 0.000 |
| ResNet + U-Net | 0.820 | 0.560 | 0.621 | 0.000 |
| EfficientNet + Attention | 0.815 | 0.571 | 0.640 | 0.000 |
| Swin Transformer + U-Net | 0.783 | 0.588 | 0.375 | 0.000 |
| CNN + LSTM | 0.868 | 0.615 | 0.455 | 0.000 |
