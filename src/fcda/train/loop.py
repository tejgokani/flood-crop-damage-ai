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


#: Exponent on inverse class frequency for the training sampler.
#:
#: Sampler weight is (1 / count)^alpha, so the resulting class share is count^(1 - alpha):
#: alpha = 0 is the natural distribution, alpha = 1 is fully balanced. On this corpus
#: (633/164/83/20) alpha = 1 gives Severe an 11.2x boost, and with only 20 unique Severe tiles
#: that means showing the same handful of images over and over -- trading a class-imbalance
#: problem for a memorisation one. alpha = 0.5 lifts Severe from 2.2% to 8.7% of each epoch,
#: a 3.9x boost, which is enough exposure without that.
SAMPLER_ALPHA = 0.5


def _sampler_kwargs(dataset, balanced: bool, samples_per_epoch: int | None = None) -> dict:
    """Frequency-aware sampling for the training split.

    With 20 Severe tiles against 633 Healthy, most minibatches under plain shuffling contain no
    Severe example at all, so the rare classes are barely *seen* however heavily they are
    weighted in the loss.

    Important: when this sampler is active the loss must NOT also apply inverse-frequency class
    weights. Doing both corrects the same imbalance twice -- measured on this corpus, Severe
    ended up boosted 32x, the model over-predicted rare classes, and fold-1 accuracy collapsed
    to 0.139 against a 0.400 baseline. The sampler owns exposure; the loss stays neutral.
    """
    if not balanced:
        return {"shuffle": True}
    try:
        labels = np.array([int(dataset[i][2]) for i in range(len(dataset))])
    except Exception:  # noqa: BLE001 - any dataset that cannot be indexed falls back to shuffling
        return {"shuffle": True}
    counts = np.bincount(labels, minlength=4).astype(float)
    counts[counts == 0] = 1.0
    weights = np.power(1.0 / counts, SAMPLER_ALPHA)[labels]
    sampler = torch.utils.data.WeightedRandomSampler(
        weights=torch.as_tensor(weights, dtype=torch.double),
        num_samples=samples_per_epoch or len(labels),
        replacement=True,
    )
    return {"sampler": sampler}


class ListDataset(Dataset):
    """Wraps GAN-generated samples so they are indistinguishable from real ones.

    Accepts either ``(image, mask, label)`` or ``(image, mask, label, fraction)`` and always
    yields the 4-tuple the real dataset yields, computing the flood fraction from the mask when
    it is not supplied.
    """

    def __init__(self, items: list[tuple]):
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i):
        item = self.items[i]
        img, mask, label = item[0], item[1], item[2]
        m = np.asarray(mask, dtype=np.float32)
        fraction = np.float32(item[3]) if len(item) > 3 else np.float32(m.mean())
        return (
            torch.from_numpy(np.asarray(img)).float(),
            torch.from_numpy(m),
            int(label),
            fraction,
        )


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
    #: Final training macro-F1 minus best validation macro-F1. <= 0.10 means healthy.
    gap: float = 0.0
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
            "gap": round(self.gap, 4),
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
        frac_weight: float = 2.0,
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.seg_weight = seg_weight
        self.cls_weight = cls_weight
        #: Weight on the flood-fraction regression term. See forward() for why it exists.
        self.frac_weight = frac_weight
        self.ce = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)

    @staticmethod
    def _dice(logits: torch.Tensor, target: torch.Tensor, eps: float = 1.0) -> torch.Tensor:
        prob = torch.sigmoid(logits)
        num = 2.0 * (prob * target).sum(dim=(1, 2, 3)) + eps
        den = prob.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + eps
        return (1.0 - num / den).mean()

    def forward(self, seg_logits, cls_logits, seg_target, cls_target, frac_target=None):
        """Segmentation + severity + flood-fraction regression.

        The third term is the important one for this dataset. Severity is a deterministic
        binning of the net flood fraction at 2%/10%/33%, so four sparse class labels are a
        lossy encoding of one dense continuous quantity. Supervising the predicted fraction
        directly means every tile contributes a gradient to the ordinal structure, instead of
        the Severe boundary having to be inferred from the 20 Severe tiles that exist.
        """
        bce = F.binary_cross_entropy_with_logits(seg_logits, seg_target)
        dice = self._dice(seg_logits, seg_target)
        ce = self.ce(cls_logits, cls_target)
        total = self.seg_weight * (bce + dice) + self.cls_weight * ce

        if frac_target is not None and self.frac_weight > 0:
            pred_frac = torch.sigmoid(seg_logits).mean(dim=(1, 2, 3))
            total = total + self.frac_weight * F.mse_loss(pred_frac, frac_target)
        return total


def compute_class_weights(labels: np.ndarray, n_classes: int = 4) -> torch.Tensor:
    """Inverse-frequency weights, normalised to mean 1 so the loss scale is unchanged."""
    counts = np.bincount(np.asarray(labels), minlength=n_classes).astype(float)
    counts[counts == 0] = 1.0
    w = counts.sum() / (n_classes * counts)
    return torch.tensor(w / w.mean(), dtype=torch.float32)


@torch.no_grad()
def evaluate(
    model, loader, device, loss_fn=None, collect_seg: bool = True, return_logits: bool = False
):
    """Run the model over a loader and score both heads.

    With ``return_logits`` the raw class logits and labels come back too, which is what
    temperature calibration needs to be fitted without a second forward pass.
    """
    model.eval()
    dev = torch.device(device)
    y_true, y_pred, all_logits = [], [], []
    seg_logits, seg_targets = [], []
    total_loss, n_batches = 0.0, 0

    for batch in loader:
        x, mask, y = batch[0].to(dev), batch[1].to(dev), batch[2].to(dev)
        frac = batch[3].to(dev) if len(batch) > 3 else None
        sl, cl = model(x)
        if loss_fn is not None:
            total_loss += float(loss_fn(sl, cl, mask, y, frac).detach())
            n_batches += 1
        y_true.append(y.cpu().numpy())
        y_pred.append(cl.argmax(1).cpu().numpy())
        all_logits.append(cl.float().cpu().numpy())
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
    loss = total_loss / max(n_batches, 1)
    if return_logits:
        return (
            metrics,
            loss,
            np.concatenate(all_logits) if all_logits else np.zeros((0, 4)),
            np.concatenate(y_true) if y_true else np.array([]),
        )
    return metrics, loss


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
    balanced_sampling: bool = True,
    samples_per_epoch: int | None = None,
) -> TrainResult:
    """Train one model under a hard wall-clock budget, keeping the best checkpoint.

    ``samples_per_epoch`` fixes how many samples constitute an epoch, independently of how large
    the training set is. Without it, adding data silently changes what "an epoch" costs, and any
    A/B against a larger training set becomes a comparison of compute rather than of data: a
    measured case had the augmented arm complete **one** epoch against the baseline's eight under
    the same wall-clock cap, and collapse to predicting one class for 445 of 450 test tiles.
    """
    name = getattr(model, "name", model.__class__.__name__)
    result = TrainResult(model_name=name, n_params=getattr(model, "n_params", lambda: 0)())
    dev = torch.device(pick_device(device))
    t_start = time.time()
    budget = max_minutes * 60.0

    try:
        model.to(dev)
        if hasattr(model, "set_dropout"):
            model.set_dropout(config.dropout)

        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            drop_last=False,
            **_sampler_kwargs(train_ds, balanced=balanced_sampling,
                              samples_per_epoch=samples_per_epoch),
        )
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

        opt = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(config.epochs, 1))
        # See _sampler_kwargs: exposure and loss weighting must not both correct the imbalance.
        effective_weights = None if balanced_sampling else class_weights
        loss_fn = DualLoss(
            class_weights=effective_weights.to(dev) if effective_weights is not None else None,
            label_smoothing=config.label_smoothing,
        )

        best_state, epochs_without_gain = None, 0

        for epoch in range(1, config.epochs + 1):
            ep_t0 = time.time()
            model.train()
            losses, yt, yp = [], [], []

            for batch in train_loader:
                x, mask, y = batch[0].to(dev), batch[1].to(dev), batch[2].to(dev)
                frac = batch[3].to(dev) if len(batch) > 3 else None
                sl, cl = model(x)
                loss = loss_fn(sl, cl, mask, y, frac)
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

    from ..eval.metrics import generalisation_gap

    result.gap = generalisation_gap(result.train_curve, result.val_curve)
    result.seconds = time.time() - t_start
    return result
