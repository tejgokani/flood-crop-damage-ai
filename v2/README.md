# v2 — focused run at macro-F1 ≥ 0.70

A clean rebuild targeting **macro-F1 ≥ 0.70** on the four severity classes, with three
architectures instead of five.

## Why macro-F1 and not accuracy

The dataset is **70.3% Healthy**. A one-line model that always predicts Healthy scores
**0.703 accuracy** — so an accuracy target near 70% measures nothing. v1's best model reached
0.713, barely above that floor.

Macro-F1 averages the four classes equally, so it cannot be gamed by predicting the majority
class. v1 reached **0.462**. Getting to 0.70 needs roughly:

| Class | v1 F1 | needed for macro-F1 0.70 |
|---|---:|---:|
| Healthy | 0.855 | ~0.90 |
| Mild | 0.407 | ~0.65 |
| Moderate | 0.379 | ~0.65 |
| **Severe** | **0.205** | **~0.60** |

**Severe is the binding constraint.** There are 20 real Severe tiles in the entire corpus. That
is the problem v2 exists to attack.

## The three techniques

| Model | Input | Why |
|---|---|---|
| **YOLO12 + U-Net** | 6-ch `[post, post−pre]` change stack | Required by the problem statement; best v1 model after fixes |
| **CNN + LSTM** | ordered `[pre, post]` pair | Only v1 model that never overfit; temporal reasoning |
| **EfficientNet + Attention** | single-date 3-ch | Best accuracy-per-FLOP, **and the only one that can consume Sen1Floods11**, which has no pre-flood image |

## Data expansion

| Source | Tiles | Use |
|---|---:|---|
| ETCI-2021 Bangladesh (full pool) | 2,032 pre/post pairs | up from 900 in v1 |
| Procedural synthetic | generated on demand | **class-targeted** — we control flood fraction exactly, so Severe is no longer capped at 20 |
| Sen1Floods11 India | 68 hand-labelled chips | real Indian data; single-date model only |

**Synthetic tiles are training-only. Every reported number is measured on real tiles, out of
fold.** That separation is enforced in code and asserted in tests — it is the difference between
expanding a dataset and inventing a result.

> The v1 GAN ablation measured **−0.101 macro-F1** across 3 folds: the DCGAN reintroduced the
> overfitting that had just been removed. v2 therefore uses **procedural** synthesis (explicit
> flood physics, controllable severity) rather than a learned generator.

## Relationship to v1

v2 is a separate package with its own data layer, training loop and evaluation. It deliberately
reuses v1's *proven primitives* — the severity rule, metrics, calibration and split logic — by
importing them, rather than re-deriving code that already has tests and known-fixed bugs behind
it. Everything above that line is new.
