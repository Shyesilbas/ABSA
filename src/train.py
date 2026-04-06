from __future__ import annotations

import os
import random
from contextlib import nullcontext
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, f1_score
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

try:
    from torch.cuda.amp import GradScaler, autocast as _cuda_autocast
except ImportError:
    GradScaler = None  # type: ignore[misc, assignment]

    def _cuda_autocast():
        return nullcontext()


def _amp_ctx(use_amp: bool):
    return _cuda_autocast() if use_amp else nullcontext()

from config import (
    MODEL_NAME,
    MAX_LEN,
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    TRAIN_DATA_PATH,
    VAL_DATA_PATH,
    RAW_DATA_PATH,
    MODEL_PATH,
    CLASS_NAMES,
    RANDOM_SEED,
    USE_CLASS_WEIGHTS,
    NEUTRAL_CLASS_INDEX,
    NEUTRAL_LOSS_BOOST,
    EARLY_STOPPING,
    EARLY_STOPPING_PATIENCE,
    EARLY_STOPPING_MIN_DELTA,
    USE_AMP,
    WARMUP_RATIO,
    DATALOADER_NUM_WORKERS,
    MERGE_RAW_ABSA_FOR_TRAIN,
    USE_HF_TRAIN_EXTRA,
    HF_DATASET_ID,
    HF_SAMPLE_SIZE,
    HF_SEED,
    HARD_EXAMPLES_PATH,
    MERGE_HARD_EXAMPLES,
)
from dataset_loader import SentenceClassificationDataset
from training_data import build_train_val_frames


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_class_weights_from_labels(y: np.ndarray, n: int) -> torch.Tensor:
    y = np.asarray(y, dtype=np.int64)
    counts = np.bincount(y, minlength=n).astype(np.float64)
    total = counts.sum()
    if total == 0:
        w = np.ones(n)
    else:
        w = total / (n * np.maximum(counts, 1.0))
        w = w / w.mean()
    t = torch.tensor(w, dtype=torch.float32)
    t[NEUTRAL_CLASS_INDEX] *= NEUTRAL_LOSS_BOOST
    t = t / t.mean()
    return t


def train_epoch(model, loader, optimizer, scheduler, device, use_amp, scaler, loss_fn):
    model.train()
    losses = []
    for batch in loader:
        optimizer.zero_grad(set_to_none=True)
        ids = batch["input_ids"].to(device, non_blocking=True)
        mask = batch["attention_mask"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)

        with _amp_ctx(use_amp):
            out = model(input_ids=ids, attention_mask=mask)
            logits = out.logits
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
        optimizer.zero_grad()
        losses.append(loss.detach().item())
    return float(np.mean(losses))


@torch.no_grad()
def evaluate(model, loader, loss_fn, device, use_amp):
    model.eval()
    all_y, all_p = [], []
    losses = []
    for batch in loader:
        ids = batch["input_ids"].to(device, non_blocking=True)
        mask = batch["attention_mask"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        with _amp_ctx(use_amp):
            logits = model(input_ids=ids, attention_mask=mask).logits
        loss = loss_fn(logits.float(), labels)
        losses.append(loss.item())
        all_y.extend(labels.cpu().tolist())
        all_p.extend(logits.argmax(-1).cpu().tolist())
    macro = f1_score(all_y, all_p, average="macro", zero_division=0)
    return float(np.mean(losses)), macro, all_y, all_p


def main():
    set_seed(RANDOM_SEED)
    cuda_ok = torch.cuda.is_available()
    device = torch.device(f"cuda:{torch.cuda.current_device()}" if cuda_ok else "cpu")
    use_amp = bool(USE_AMP and cuda_ok)

    print(f"Device: {device}  AMP: {use_amp}")

    train_df, val_df = build_train_val_frames(
        TRAIN_DATA_PATH,
        VAL_DATA_PATH,
        RAW_DATA_PATH,
        MERGE_RAW_ABSA_FOR_TRAIN,
        USE_HF_TRAIN_EXTRA,
        HF_DATASET_ID,
        HF_SAMPLE_SIZE,
        HF_SEED,
        hard_path=HARD_EXAMPLES_PATH,
        merge_hard=MERGE_HARD_EXAMPLES,
    )
    print(f"Eğitim cümle sayısı: {len(train_df)}  |  Doğrulama: {len(val_df)}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_ds = SentenceClassificationDataset(tokenizer, MAX_LEN, dataframe=train_df.reset_index(drop=True))
    val_ds = SentenceClassificationDataset(tokenizer, MAX_LEN, dataframe=val_df.reset_index(drop=True))

    nw = DATALOADER_NUM_WORKERS if cuda_ok else 0
    dl_kw: dict = {"num_workers": nw, "pin_memory": cuda_ok}
    if nw > 0:
        dl_kw["persistent_workers"] = True
        dl_kw["prefetch_factor"] = 2

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, **dl_kw)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, **dl_kw)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=len(CLASS_NAMES)
    )
    model.to(device)

    if USE_CLASS_WEIGHTS:
        y_tr = train_df["Polarity"].values.astype(int)
        cw = compute_class_weights_from_labels(y_tr, len(CLASS_NAMES)).to(device)
        print(f"Sınıf ağırlıkları: {cw.cpu().numpy().round(4)}")
        loss_fn = nn.CrossEntropyLoss(weight=cw)
    else:
        loss_fn = nn.CrossEntropyLoss()

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    total_steps = len(train_loader) * EPOCHS
    warmup = int(total_steps * WARMUP_RATIO)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup, num_training_steps=total_steps
    )

    scaler: Any = None
    if use_amp and cuda_ok and GradScaler is not None:
        scaler = GradScaler()

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    best_f1 = -1.0
    stale = 0

    for epoch in range(EPOCHS):
        tl = train_epoch(model, train_loader, optimizer, scheduler, device, use_amp, scaler, loss_fn)
        vl, macro_f1, y_true, y_pred = evaluate(model, val_loader, loss_fn, device, use_amp)
        print(f"\nEpoch {epoch + 1}/{EPOCHS}  train_loss={tl:.4f}  val_loss={vl:.4f}  val_macro_f1={macro_f1:.4f}")
        print(classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=4, zero_division=0))

        if macro_f1 > best_f1 + EARLY_STOPPING_MIN_DELTA:
            best_f1 = macro_f1
            stale = 0
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"=> Kaydedildi: {MODEL_PATH}")
        else:
            stale += 1
            if EARLY_STOPPING and stale >= EARLY_STOPPING_PATIENCE:
                print("Early stopping.")
                break

    if os.path.isfile(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    print(f"Bitti. En iyi val macro_f1: {best_f1:.4f}")


if __name__ == "__main__":
    main()
