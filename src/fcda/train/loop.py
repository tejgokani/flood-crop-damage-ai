"""The training loop.

Design constraints that shaped this:

* **Wall-clock capped.** The whole benchmark has to finish overnight on one laptop, and a
  single slow model must not starve the other four. Every run carries a hard time budget and
  stops cleanly at the boundary, keeping its best checkpoint.
* **Never fatal.** A model that fails to train produces a logged row with its error, not a
  dead run. A five-model comparison that dies on model three is worth nothing at 9am.
* **Augmentation is injected, not assumed.** GAN samples are passed in as an explicit list
  drawn from the training split, so this function has no route to the evaluation data.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from ..eval.metrics import Metrics, compute_metrics
from .correction import TrainConfig


def pick_device(prefer: str = "auto") -> str:
    if prefer != "auto":
        return prefer
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class ListDataset(Dataset):
    """Wraps a list of ``(image, mask, label)`` triples, real or GAN-generated."""

    def __init__(self, items: list[tuple[np.ndarray, np.ndarray, int]]):
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i):
        img, mask, label = self.items[i]
        return torch.from_numpy(np.asarray(img)).float(), torch.from_numpy(np.asarray(mask)).float(), int(label)


class ConcatWithSynthetic(Dataset):
    """Real training tiles followed by GAN samples, presented as one dataset."""

    def __init__(self, base: Dataset, synthetic: list[tuple[np.ndarray, np.ndarray, int]]):
        self.base = base
        self.synth = ListDataset(synthetic)

    def __len__(self) -> int:
        return len(self.base) + len(self.synth)

    def __getitem__(self, i):
        if i < len(self.base):
            return self.base[i]
        return self.synth[i - len(self.base)]


@dataclass
class EpochRecord:
    epoch: int
    train_loss: float
    train_f1: float
    val_loss: float
    val_f1: float
    seconds: float


@dataclass
class TrainResult:
    model_name: str
    ok: bool = True
    error: str = ""
    epochs: list[EpochRecord] = field(default_factory=list)
    best_val_f1: float = 0.0
    best_epoch: int = -1
    seconds: float = 0.0
    stopped_reason: str = ""
    n_params: int = 0
    val_metrics: Metrics | None = None
    test_metrics: Metrics | None = None

    @property
    def train_curve(self) -> list[float]:
        return [e.train_f1 for e in self.epochs]

    @property
    def val_curve(self) -> list[float]:
        return [e.val_f1 for e in self.epochs]

    def to_dict(self) -> dict:
        return {
            "model": self.model_name,
            "ok": self.ok,
            "error": self.error,
            "best_val_f1": round(self.best_val_f1, 4),
            "best_epoch": self.best_epoch,
            "seconds": round(self.seconds, 1),
            "stopped_reason": self.stopped_reason,
            "n_params": self.n_params,
            "epochs": [e.__dict__ for e in self.epochs],
            "val_metrics": self.val_metrics.to_dict() if self.val_metrics else None,
            "test_metrics": self.test_metrics.to_dict() if self.test_metrics else None,
        }


class DualLoss(nn.Module):
    """Segmentation (BCE + soft Dice) plus severity cross-entropy.

    The Dice term matters because flood pixels are a small minority of most tiles: pure BCE
    is minimised well by predicting "no flood" everywhere, while Dice is driven by overlap
    and so refuses that solution. Class weights handle the same problem on the severity head.
    """

    def __init__(
        self,
        class_weights: torch.Tensor | None = None,
        seg_weight: float = 1.0,
        cls_weight: float = 1.0,
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.seg_weight = seg_weight
        self.cls_weight = cls_weight
        self.ce = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)

    @staticmethod
    def _dice(logits: torch.Tensor, target: torch.Tensor, eps: float = 1.0) -> torch.Tensor:
        prob = torch.sigmoid(logits)
        num = 2.0 * (prob * target).sum(dim=(1, 2, 3)) + eps
        den = prob.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + eps
        return (1.0 - num / den).mean()

    def forward(self, seg_logits, cls_logits, seg_target, cls_target):
        bce = F.binary_cross_entropy_with_logits(seg_logits, seg_target)
        dice = self._dice(seg_logits, seg_target)
        ce = self.ce(cls_logits, cls_target)
        return self.seg_weight * (bce + dice) + self.cls_weight * ce


def compute_class_weights(labels: np.ndarray, n_classes: int = 4) -> torch.Tensor:
    """Inverse-frequency weights, normalised to mean 1 so the loss scale is unchanged."""
    counts = np.bincount(np.asarray(labels), minlength=n_classes).astype(float)
    counts[counts == 0] = 1.0
    w = counts.sum() / (n_classes * counts)
    return torch.tensor(w / w.mean(), dtype=torch.float32)


@torch.no_grad()
def evaluate(model, loader, device, loss_fn=None, collect_seg: bool = True) -> tuple[Metrics, float]:
    """Run the model over a loader and score both heads."""
    model.eval()
    dev = torch.device(device)
    y_true, y_pred = [], []
    seg_logits, seg_targets = [], []
    total_loss, n_batches = 0.0, 0

    for x, mask, y in loader:
        x, mask, y = x.to(dev), mask.to(dev), y.to(dev)
        sl, cl = model(x)
        if loss_fn is not None:
            total_loss += float(loss_fn(sl, cl, mask, y).detach())
            n_batches += 1
        y_true.append(y.cpu().numpy())
        y_pred.append(cl.argmax(1).cpu().numpy())
        if collect_seg:
            # Subsample spatially: full-resolution logits for a whole split will not fit in
            # memory at the larger tiers, and IoU over a regular 4x grid is unbiased.
            seg_logits.append(sl[:, :, ::4, ::4].float().cpu().numpy())
            seg_targets.append(mask[:, :, ::4, ::4].float().cpu().numpy())

    metrics = compute_metrics(
        np.concatenate(y_true) if y_true else np.array([]),
        np.concatenate(y_pred) if y_pred else np.array([]),
        np.concatenate(seg_logits) if seg_logits else None,
        np.concatenate(seg_targets) if seg_targets else None,
    )
    return metrics, (total_loss / max(n_batches, 1))


def train_model(
    model,
    train_ds: Dataset,
    val_ds: Dataset,
    config: TrainConfig,
    device: str = "auto",
    batch_size: int = 8,
    max_minutes: float = 25.0,
    class_weights: torch.Tensor | None = None,
    checkpoint: Path | None = None,
    verbose: bool = True,
    tag: str = "",
) -> TrainResult:
    """Train one model under a hard wall-clock budget, keeping the best checkpoint."""
    name = getattr(model, "name", model.__class__.__name__)
    result = TrainResult(model_name=name, n_params=getattr(model, "n_params", lambda: 0)())
    dev = torch.device(pick_device(device))
    t_start = time.time()
    budget = max_minutes * 60.0

    try:
        model.to(dev)
        if hasattr(model, "set_dropout"):
            model.set_dropout(config.dropout)

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

        opt = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(config.epochs, 1))
        loss_fn = DualLoss(
            class_weights=class_weights.to(dev) if class_weights is not None else None,
            label_smoothing=config.label_smoothing,
        )

        best_state, epochs_without_gain = None, 0

        for epoch in range(1, config.epochs + 1):
            ep_t0 = time.time()
            model.train()
            losses, yt, yp = [], [], []

            for x, mask, y in train_loader:
                x, mask, y = x.to(dev), mask.to(dev), y.to(dev)
                sl, cl = model(x)
                loss = loss_fn(sl, cl, mask, y)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()
                losses.append(float(loss.detach()))
                yt.append(y.detach().cpu().numpy())
                yp.append(cl.detach().argmax(1).cpu().numpy())

                if time.time() - t_start > budget:
                    break

            from ..eval.metrics import classification_scores

            train_f1 = classification_scores(np.concatenate(yt), np.concatenate(yp))["macro_f1"]
            val_metrics, val_loss = evaluate(model, val_loader, dev, loss_fn)
            sched.step()

            rec = EpochRecord(
                epoch=epoch,
                train_loss=float(np.mean(losses)) if losses else 0.0,
                train_f1=train_f1,
                val_loss=val_loss,
                val_f1=val_metrics.macro_f1,
                seconds=time.time() - ep_t0,
            )
            result.epochs.append(rec)

            if verbose:
                print(
                    f"    [{tag or name}] ep {epoch:2d}/{config.epochs}  "
                    f"loss {rec.train_loss:.3f}  trF1 {rec.train_f1:.3f}  "
                    f"vaF1 {rec.val_f1:.3f}  IoU {val_metrics.iou:.3f}  {rec.seconds:.0f}s",
                    flush=True,
                )

            if val_metrics.macro_f1 > result.best_val_f1:
                result.best_val_f1 = val_metrics.macro_f1
                result.best_epoch = epoch
                result.val_metrics = val_metrics
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                epochs_without_gain = 0
            else:
                epochs_without_gain += 1

            if time.time() - t_start > budget:
                result.stopped_reason = f"wall-clock cap of {max_minutes:.0f} min reached"
                break
            if epochs_without_gain >= config.patience:
                result.stopped_reason = f"early stop: no gain for {config.patience} epochs"
                break

        if not result.stopped_reason:
            result.stopped_reason = "completed all epochs"

        if best_state is not None:
            model.load_state_dict(best_state)
            if checkpoint is not None:
                checkpoint.parent.mkdir(parents=True, exist_ok=True)
                torch.save(best_state, checkpoint)

    except Exception as exc:  # noqa: BLE001 - a failed model must not kill the benchmark
        result.ok = False
        result.error = f"{type(exc).__name__}: {exc}"
        result.stopped_reason = "failed"
        if verbose:
            print(f"    [{tag or name}] FAILED: {result.error}", flush=True)

    result.seconds = time.time() - t_start
    return result
