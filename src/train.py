from __future__ import annotations

import os
import random
from typing import Any

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

try:
    from torch.cuda.amp import GradScaler
except ImportError:
    GradScaler = None  # type: ignore[misc, assignment]

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
from trainer import build_loss_fn, fit
from training_data import build_train_val_frames


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_dataloaders(train_df, val_df, tokenizer, cuda_ok: bool):
    train_ds = SentenceClassificationDataset(
        tokenizer,
        MAX_LEN,
        dataframe=train_df.reset_index(drop=True),
    )
    val_ds = SentenceClassificationDataset(
        tokenizer,
        MAX_LEN,
        dataframe=val_df.reset_index(drop=True),
    )

    num_workers = DATALOADER_NUM_WORKERS if cuda_ok else 0
    dl_kw: dict[str, Any] = {"num_workers": num_workers, "pin_memory": cuda_ok}
    if num_workers > 0:
        dl_kw["persistent_workers"] = True
        dl_kw["prefetch_factor"] = 2

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, **dl_kw)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, **dl_kw)
    return train_loader, val_loader


def main():
    set_seed(RANDOM_SEED)
    cuda_ok = torch.cuda.is_available()
    device = torch.device(f"cuda:{torch.cuda.current_device()}" if cuda_ok else "cpu")
    use_amp = bool(USE_AMP and cuda_ok)
    print(f"Device: {device} | AMP: {use_amp}")

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
    print(f"Training sentences: {len(train_df)} | Validation: {len(val_df)}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_loader, val_loader = build_dataloaders(train_df, val_df, tokenizer, cuda_ok=cuda_ok)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=len(CLASS_NAMES)
    )
    model.to(device)

    loss_fn = build_loss_fn(
        train_labels=train_df["Polarity"].values.astype(int),
        use_class_weights=USE_CLASS_WEIGHTS,
        num_classes=len(CLASS_NAMES),
        neutral_class_index=NEUTRAL_CLASS_INDEX,
        neutral_loss_boost=NEUTRAL_LOSS_BOOST,
        device=device,
    )

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    total_steps = len(train_loader) * EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    scaler: Any = None
    if use_amp and cuda_ok and GradScaler is not None:
        scaler = GradScaler()

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    best_f1 = fit(
        model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        use_amp=use_amp,
        scaler=scaler,
        loss_fn=loss_fn,
        class_names=CLASS_NAMES,
        epochs=EPOCHS,
        model_path=MODEL_PATH,
        early_stopping=EARLY_STOPPING,
        early_stopping_patience=EARLY_STOPPING_PATIENCE,
        early_stopping_min_delta=EARLY_STOPPING_MIN_DELTA,
    )

    if os.path.isfile(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))

    print(f"Training completed. Best val macro_f1: {best_f1:.4f}")


if __name__ == "__main__":
    main()
