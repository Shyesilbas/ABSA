from __future__ import annotations

from contextlib import nullcontext
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, f1_score

from core.progress import loader_total, track

try:
    from torch.amp import autocast as _amp_autocast

    def _amp_ctx(use_amp: bool):
        return _amp_autocast("cuda") if use_amp else nullcontext()
except ImportError:
    try:
        from torch.cuda.amp import autocast as _cuda_autocast

        def _amp_ctx(use_amp: bool):
            return _cuda_autocast() if use_amp else nullcontext()
    except ImportError:
        def _amp_ctx(use_amp: bool):
            return nullcontext()


def compute_class_weights_from_labels(
    labels: np.ndarray,
    num_classes: int,
    neutral_class_index: int,
    neutral_loss_boost: float,
) -> torch.Tensor:
    labels = np.asarray(labels, dtype=np.int64)
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    total = counts.sum()
    if total == 0:
        weights = np.ones(num_classes)
    else:
        weights = total / (num_classes * np.maximum(counts, 1.0))
        weights = weights / weights.mean()

    out = torch.tensor(weights, dtype=torch.float32)
    out[neutral_class_index] *= neutral_loss_boost
    out = out / out.mean()
    return out


def build_loss_fn(
    train_labels: np.ndarray,
    *,
    use_class_weights: bool,
    num_classes: int,
    neutral_class_index: int,
    neutral_loss_boost: float,
    device: torch.device,
) -> nn.Module:
    if not use_class_weights:
        return nn.CrossEntropyLoss()

    class_weights = compute_class_weights_from_labels(
        labels=train_labels,
        num_classes=num_classes,
        neutral_class_index=neutral_class_index,
        neutral_loss_boost=neutral_loss_boost,
    ).to(device)
    print(f"Class weights: {class_weights.cpu().numpy().round(4)}")
    return nn.CrossEntropyLoss(weight=class_weights)


def train_epoch(
    model,
    loader,
    optimizer,
    scheduler,
    device,
    use_amp,
    scaler,
    loss_fn,
    *,
    progress_desc: str = "Train",
):
    model.train()
    losses = []
    n_batches = loader_total(loader)
    for batch in track(loader, total=n_batches, desc=progress_desc, unit="batch"):
        optimizer.zero_grad(set_to_none=True)
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)

        with _amp_ctx(use_amp):
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        loss = loss_fn(logits.float(), labels)

        if use_amp and scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        scheduler.step()
        losses.append(loss.detach().item())
    return float(np.mean(losses))


@torch.no_grad()
def evaluate_epoch(
    model,
    loader,
    loss_fn,
    device,
    use_amp,
    *,
    progress_desc: str = "Val",
):
    model.eval()
    all_y, all_p = [], []
    losses = []
    n_batches = loader_total(loader)
    for batch in track(loader, total=n_batches, desc=progress_desc, unit="batch"):
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        with _amp_ctx(use_amp):
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        loss = loss_fn(logits.float(), labels)
        losses.append(loss.item())
        all_y.extend(labels.cpu().tolist())
        all_p.extend(logits.argmax(-1).cpu().tolist())
    macro_f1 = f1_score(all_y, all_p, average="macro", zero_division=0)
    return float(np.mean(losses)), macro_f1, all_y, all_p


def fit(
    model,
    *,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    device: torch.device,
    use_amp: bool,
    scaler: Any,
    loss_fn: nn.Module,
    class_names: Sequence[str],
    epochs: int,
    model_path: str,
    early_stopping: bool,
    early_stopping_patience: int,
    early_stopping_min_delta: float,
) -> tuple[float, list[dict[str, float | int | bool]]]:
    best_f1 = -1.0
    stale_epochs = 0
    history: list[dict[str, float | int | bool]] = []

    for epoch in range(epochs):
        ep = epoch + 1
        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            device,
            use_amp,
            scaler,
            loss_fn,
            progress_desc=f"Train {ep}/{epochs}",
        )
        val_loss, macro_f1, y_true, y_pred = evaluate_epoch(
            model,
            val_loader,
            loss_fn,
            device,
            use_amp,
            progress_desc=f"Val {ep}/{epochs}",
        )
        print(
            f"\nEpoch {epoch + 1}/{epochs} "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_macro_f1={macro_f1:.4f}"
        )
        print(
            classification_report(
                y_true,
                y_pred,
                target_names=class_names,
                digits=4,
                zero_division=0,
            )
        )

        improved = bool(macro_f1 > best_f1 + early_stopping_min_delta)
        history.append(
            {
                "epoch": ep,
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
                "val_macro_f1": float(macro_f1),
                "improved": improved,
            }
        )

        if improved:
            best_f1 = macro_f1
            stale_epochs = 0
            torch.save(model.state_dict(), model_path)
            print(f"Checkpoint saved: {model_path}")
            continue

        stale_epochs += 1
        if early_stopping and stale_epochs >= early_stopping_patience:
            print("Early stopping triggered.")
            break

    return best_f1, history
