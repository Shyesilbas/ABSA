from __future__ import annotations

import json
import os
import random
from datetime import datetime, timezone
from typing import Any

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from core.config import (
    BATCH_SIZE,
    CLASS_NAMES,
    CONFIDENCE_FALLBACK_ENABLED,
    CONFIDENCE_FALLBACK_LABEL,
    CONFIDENCE_THRESHOLD,
    DATALOADER_NUM_WORKERS,
    EARLY_STOPPING,
    EARLY_STOPPING_MIN_DELTA,
    EARLY_STOPPING_PATIENCE,
    EPOCHS,
    EXPERIMENT_ARTIFACT_PATH,
    HARD_EXAMPLES_PATH,
    HF_DATASET_ID,
    HF_SAMPLE_SIZE,
    HF_SEED,
    LEAKAGE_GUARD_ENABLED,
    LEARNING_RATE,
    MAX_LEN,
    MERGE_HARD_EXAMPLES,
    MERGE_RAW_ABSA_FOR_TRAIN,
    MODEL_NAME,
    MODEL_PATH,
    NEUTRAL_CLASS_INDEX,
    NEUTRAL_LOSS_BOOST,
    RANDOM_SEED,
    RAW_DATA_PATH,
    TRAIN_DATA_PATH,
    USE_AMP,
    USE_CLASS_WEIGHTS,
    USE_HF_TRAIN_EXTRA,
    VAL_DATA_PATH,
    WARMUP_RATIO,
)
from data.dataset_loader import SentenceClassificationDataset
from data.training_data import build_train_val_frames
from model.trainer import build_loss_fn, fit

try:
    from torch.amp import GradScaler
except ImportError:
    try:
        from torch.cuda.amp import GradScaler
    except ImportError:
        GradScaler = None  # type: ignore[misc, assignment]


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


def _save_experiment_artifact(
    *,
    path: str,
    best_f1: float,
    history: list[dict[str, float | int | bool]],
    train_size: int,
    val_size: int,
    device: str,
    use_amp: bool,
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_name": MODEL_NAME,
        "model_path": MODEL_PATH,
        "class_names": CLASS_NAMES,
        "dataset_sizes": {"train": train_size, "val": val_size},
        "runtime": {"device": device, "use_amp": bool(use_amp)},
        "config_snapshot": {
            "random_seed": RANDOM_SEED,
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "learning_rate": LEARNING_RATE,
            "max_len": MAX_LEN,
            "warmup_ratio": WARMUP_RATIO,
            "use_class_weights": USE_CLASS_WEIGHTS,
            "neutral_class_index": NEUTRAL_CLASS_INDEX,
            "neutral_loss_boost": NEUTRAL_LOSS_BOOST,
            "early_stopping": EARLY_STOPPING,
            "early_stopping_patience": EARLY_STOPPING_PATIENCE,
            "early_stopping_min_delta": EARLY_STOPPING_MIN_DELTA,
            "merge_raw_absa_for_train": MERGE_RAW_ABSA_FOR_TRAIN,
            "use_hf_train_extra": USE_HF_TRAIN_EXTRA,
            "hf_dataset_id": HF_DATASET_ID,
            "hf_sample_size": HF_SAMPLE_SIZE,
            "hf_seed": HF_SEED,
            "merge_hard_examples": MERGE_HARD_EXAMPLES,
            "leakage_guard_enabled": LEAKAGE_GUARD_ENABLED,
            "confidence_fallback_enabled": CONFIDENCE_FALLBACK_ENABLED,
            "confidence_threshold": CONFIDENCE_THRESHOLD,
            "confidence_fallback_label": CONFIDENCE_FALLBACK_LABEL,
        },
        "best_val_macro_f1": float(best_f1),
        "epochs_ran": len(history),
        "epoch_history": history,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=True, indent=2)
    print(f"Experiment artifact saved: {path}")


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
        leakage_guard=LEAKAGE_GUARD_ENABLED,
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
        scaler = GradScaler("cuda")

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    best_f1, history = fit(
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
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))

    _save_experiment_artifact(
        path=EXPERIMENT_ARTIFACT_PATH,
        best_f1=best_f1,
        history=history,
        train_size=len(train_df),
        val_size=len(val_df),
        device=str(device),
        use_amp=use_amp,
    )

    print(f"Training completed. Best val macro_f1: {best_f1:.4f}")


if __name__ == "__main__":
    main()
