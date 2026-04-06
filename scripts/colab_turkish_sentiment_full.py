#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Türkçe cümle düzeyi duygu analizi — tek script (Google Colab T4 + yerel uyumlu).

Kurulum (Colab ilk hücre):
  !pip install -q transformers datasets accelerate scikit-learn pandas torch

Veri klasörü: ortam değişkeni SENTIMENT_DATA_DIR veya aşağıdaki DATA_DIR sabiti.
  Colab örnek: os.environ["SENTIMENT_DATA_DIR"] = "/content/absa/data"
  Yerel: proje kökünden data/ (train.csv, val.csv, turkish_absa_train.csv)

Not: Hücreye yapıştırırken "import random" ile "from typing import Optional"
ayrı satırlarda kalmalı; birleşirse SyntaxError oluşur.
"""
from __future__ import annotations

import os
import random
from contextlib import nullcontext
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, f1_score
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset

try:
    from torch.cuda.amp import GradScaler, autocast as _cuda_autocast
except ImportError:
    GradScaler = None  # type: ignore[misc, assignment]

    def _cuda_autocast():
        return nullcontext()


def _amp_ctx(use_amp: bool):
    return _cuda_autocast() if use_amp else nullcontext()


from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

# ---------------------------------------------------------------------------
# Yapılandırma
# ---------------------------------------------------------------------------
MODEL_NAME = "dbmdz/bert-base-turkish-cased"  # BertForSequenceClassification ile aynı gövde
MAX_LEN = 160
BATCH_SIZE = int(os.environ["TRAIN_BATCH_SIZE"]) if os.environ.get("TRAIN_BATCH_SIZE") else 32
EPOCHS = 4
LEARNING_RATE = 2e-5
WARMUP_RATIO = 0.1
HF_SAMPLE_SIZE = int(os.environ["HF_SAMPLE_SIZE"]) if os.environ.get("HF_SAMPLE_SIZE") else 10_000
HF_SEED = 42

CLASS_NAMES = ["Negative", "Neutral", "Positive"]
EARLY_STOPPING_PATIENCE = 2
EARLY_STOPPING_MIN_DELTA = 1e-4
NEUTRAL_INDEX = 1
NEUTRAL_LOSS_BOOST = 1.5

# Yerel: repo kökü/data | Colab: SENTIMENT_DATA_DIR veya cwd/data
def _default_data_and_models() -> tuple[str, str]:
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        cwd = os.getcwd()
        return os.path.join(cwd, "data"), os.path.join(cwd, "models")
    root = os.path.dirname(script_dir)
    return os.path.join(root, "data"), os.path.join(root, "models")


_DEFAULT_DATA, _DEFAULT_MODELS = _default_data_and_models()
DATA_DIR = os.environ.get("SENTIMENT_DATA_DIR", _DEFAULT_DATA)
OUTPUT_DIR = os.environ.get("SENTIMENT_OUTPUT_DIR", _DEFAULT_MODELS)
# Yerel `src/config.py` ile aynı dosya adı (kopyala-yapıştır uyumu)
MODEL_SAVE_PATH = os.path.join(OUTPUT_DIR, "sentence_best_model.bin")
# `src/config.py` MERGE_HARD_EXAMPLES ile aynı anlam; False ise hard_examples.csv birleştirilmez
MERGE_HARD_EXAMPLES = os.environ.get("MERGE_HARD_EXAMPLES", "1").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)

RANDOM_SEED = 42


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _polarity_mode(s: pd.Series) -> int:
    vc = s.value_counts()
    return int(vc.idxmax())


def load_local_csvs(data_dir: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """train + ham ABSA birleşik eğitim havuzu; val ayrı doğrulama."""
    paths = {
        "train": os.path.join(data_dir, "train.csv"),
        "raw": os.path.join(data_dir, "turkish_absa_train.csv"),
        "val": os.path.join(data_dir, "val.csv"),
    }
    for k, p in paths.items():
        if not os.path.isfile(p):
            raise FileNotFoundError(f"Eksik dosya ({k}): {p}")

    def prep(path: str) -> pd.DataFrame:
        df = pd.read_csv(path)
        if "Aspect" in df.columns:
            df = df.drop(columns=["Aspect"])
        df = df.rename(columns={c: c.strip() for c in df.columns})
        if "Sentence" not in df.columns or "Polarity" not in df.columns:
            raise ValueError(f"Beklenen sütunlar yok: {df.columns.tolist()} — {path}")
        df["Sentence"] = df["Sentence"].astype(str).str.strip()
        df["Polarity"] = pd.to_numeric(df["Polarity"], errors="coerce").fillna(1).astype(int)
        df = df[df["Sentence"].str.len() > 0]
        df = df.groupby("Sentence", as_index=False)["Polarity"].agg(_polarity_mode)
        return df

    train_a = prep(paths["train"])
    train_b = prep(paths["raw"])
    train_pool = pd.concat([train_a, train_b], ignore_index=True)
    train_pool = train_pool.groupby("Sentence", as_index=False)["Polarity"].agg(_polarity_mode)

    val_df = prep(paths["val"])
    return train_pool, val_df


def merge_hard_examples_csv(train_df: pd.DataFrame, data_dir: str) -> pd.DataFrame:
    """data/hard_examples.csv varsa aynı cümlelerde bu etiketler baskın (train.py ile uyumlu)."""
    path = os.path.join(data_dir, "hard_examples.csv")
    if not os.path.isfile(path):
        return train_df
    df = pd.read_csv(path)
    if "Aspect" in df.columns:
        df = df.drop(columns=["Aspect"])
    df = df.rename(columns={c: c.strip() for c in df.columns})
    if "Sentence" not in df.columns or "Polarity" not in df.columns:
        print(f"Uyarı: hard_examples.csv atlandı (sütun yok): {df.columns.tolist()}")
        return train_df
    df["Sentence"] = df["Sentence"].astype(str).str.strip()
    df["Polarity"] = pd.to_numeric(df["Polarity"], errors="coerce").fillna(1).astype(int)
    df = df[df["Sentence"].str.len() > 0]
    df = df.groupby("Sentence", as_index=False)["Polarity"].agg(_polarity_mode)
    drop_s = set(df["Sentence"])
    out = train_df[~train_df["Sentence"].isin(drop_s)]
    print(f"hard_examples.csv: {len(df)} satır birleştirildi (çakışan cümlelerde baskın).")
    return pd.concat([out, df], ignore_index=True)


def map_hf_label(y) -> int:
    if y is None:
        return 1
    if isinstance(y, (int, np.integer)):
        v = int(y)
        if v in (0, 1, 2):
            return v
    s = str(y).strip().lower()
    if any(x in s for x in ("neg", "olumsuz")):
        return 0
    if any(x in s for x in ("notr", "nötr", "neutral", "objektif")):
        return 1
    if any(x in s for x in ("pos", "poz", "olumlu")):
        return 2
    try:
        v = int(s)
        if v in (0, 1, 2):
            return v
    except ValueError:
        pass
    return 1


def _hf_text_field(example: dict) -> str:
    for key in ("sentence", "text", "tweet", "Text", "Sentence", "comment"):
        if key in example and example[key] is not None:
            return str(example[key]).strip()
    raise KeyError(f"Metin alanı bulunamadı: {list(example.keys())}")


def load_hf_subset(n: int, seed: int) -> pd.DataFrame:
    print("Hugging Face veri seti indiriliyor: winvoker/turkish-sentiment-analysis-dataset …")
    ds = load_dataset("winvoker/turkish-sentiment-analysis-dataset", split="train")
    ds = ds.shuffle(seed=seed)
    ds = ds.select(range(min(n, len(ds))))
    rows = []
    label_key = None
    for c in ds.column_names:
        cl = c.lower()
        if cl in ("label", "labels", "sentiment", "polarity", "class"):
            label_key = c
            break
    if label_key is None:
        raise RuntimeError(f"Etiket sütunu yok: {ds.column_names}")

    for i in range(len(ds)):
        ex = ds[i]
        text = _hf_text_field(ex)
        lab = map_hf_label(ex.get(label_key))
        if len(text) < 1:
            continue
        rows.append({"Sentence": text, "Polarity": lab})
    return pd.DataFrame(rows)


class SentenceDataset(Dataset):
    def __init__(self, sentences, labels, tokenizer, max_len: int):
        self.sentences = list(sentences)
        self.labels = list(labels)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        s = self.sentences[idx]
        y = int(self.labels[idx])
        enc = self.tokenizer(
            s,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(y, dtype=torch.long),
        }


def compute_class_weights(labels: np.ndarray, n_classes: int, neutral_boost: float) -> torch.Tensor:
    counts = np.bincount(labels, minlength=n_classes).astype(np.float64)
    total = counts.sum()
    if total == 0:
        w = np.ones(n_classes)
    else:
        w = total / (n_classes * np.maximum(counts, 1.0))
        w = w / w.mean()
    w = torch.tensor(w, dtype=torch.float32)
    w[NEUTRAL_INDEX] *= neutral_boost
    w = w / w.mean()
    return w


@torch.no_grad()
def evaluate(model, loader, loss_fn, device, use_amp: bool):
    model.eval()
    all_y, all_p = [], []
    total_loss = 0.0
    for batch in loader:
        ids = batch["input_ids"].to(device, non_blocking=True)
        mask = batch["attention_mask"].to(device, non_blocking=True)
        y = batch["labels"].to(device, non_blocking=True)
        with _amp_ctx(use_amp):
            out = model(input_ids=ids, attention_mask=mask)
            logits = out.logits
        loss = loss_fn(logits.float(), y)
        total_loss += loss.item()
        pred = logits.argmax(dim=-1)
        all_y.extend(y.cpu().tolist())
        all_p.extend(pred.cpu().tolist())
    macro = f1_score(all_y, all_p, average="macro", zero_division=0)
    return total_loss / max(len(loader), 1), macro, all_y, all_p


def train_loop(
    model,
    train_loader,
    val_loader,
    device,
    use_amp: bool,
    class_weights: torch.Tensor,
):
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    total_steps = len(train_loader) * EPOCHS
    warmup = int(total_steps * WARMUP_RATIO)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup, num_training_steps=total_steps
    )
    loss_fn = nn.CrossEntropyLoss(weight=class_weights.to(device))
    scaler: Any = None
    if use_amp and device.type == "cuda" and GradScaler is not None:
        scaler = GradScaler()

    best_f1 = -1.0
    stale = 0
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for epoch in range(EPOCHS):
        model.train()
        losses = []
        for batch in train_loader:
            ids = batch["input_ids"].to(device, non_blocking=True)
            mask = batch["attention_mask"].to(device, non_blocking=True)
            y = batch["labels"].to(device, non_blocking=True)

            with _amp_ctx(use_amp):
                out = model(input_ids=ids, attention_mask=mask)
                logits = out.logits
            loss = loss_fn(logits.float(), y)

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

        vloss, macro_f1, y_true, y_pred = evaluate(model, val_loader, loss_fn, device, use_amp)
        print(f"\nEpoch {epoch + 1}/{EPOCHS}  train_loss={np.mean(losses):.4f}  val_loss={vloss:.4f}  val_macro_f1={macro_f1:.4f}")
        print(classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=4, zero_division=0))

        if macro_f1 > best_f1 + EARLY_STOPPING_MIN_DELTA:
            best_f1 = macro_f1
            stale = 0
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"=> En iyi model kaydedildi (macro_f1={macro_f1:.4f}) -> {MODEL_SAVE_PATH}")
        else:
            stale += 1
            if stale >= EARLY_STOPPING_PATIENCE:
                print(f"Early stopping ({EARLY_STOPPING_PATIENCE} epoch iyileşme yok).")
                break

    if os.path.isfile(MODEL_SAVE_PATH):
        model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
    return best_f1


@torch.no_grad()
def print_probs(model, tokenizer, device, sentence: str, use_amp: bool):
    model.eval()
    enc = tokenizer(
        sentence,
        max_length=MAX_LEN,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    ids = enc["input_ids"].to(device)
    mask = enc["attention_mask"].to(device)
    with _amp_ctx(use_amp):
        logits = model(input_ids=ids, attention_mask=mask).logits.float()
    prob = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
    pred = int(prob.argmax())
    print(f"\nCümle: {sentence!r}")
    print(f"  Tahmin: {CLASS_NAMES[pred]} (id={pred})")
    for i, name in enumerate(CLASS_NAMES):
        print(f"  P({name}): {100 * prob[i]:.2f}%")


def main():
    set_seed(RANDOM_SEED)
    cuda_ok = torch.cuda.is_available()
    if cuda_ok:
        idx = torch.cuda.current_device()
        device = torch.device(f"cuda:{idx}")
    else:
        device = torch.device("cpu")
    use_amp = cuda_ok

    print(f"Device: {device}  |  AMP: {use_amp}")
    print(f"DATA_DIR={DATA_DIR}")

    train_df, val_df = load_local_csvs(DATA_DIR)
    hf_df = load_hf_subset(HF_SAMPLE_SIZE, HF_SEED)
    train_df = pd.concat([train_df, hf_df], ignore_index=True)
    train_df = train_df.groupby("Sentence", as_index=False)["Polarity"].agg(_polarity_mode)
    if MERGE_HARD_EXAMPLES:
        train_df = merge_hard_examples_csv(train_df, DATA_DIR)
    else:
        print("hard_examples.csv birleştirilmedi (MERGE_HARD_EXAMPLES kapalı).")

    print(f"Eğitim cümle sayısı: {len(train_df)}  |  Doğrulama: {len(val_df)}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_ds = SentenceDataset(train_df["Sentence"].values, train_df["Polarity"].values, tokenizer, MAX_LEN)
    val_ds = SentenceDataset(val_df["Sentence"].values, val_df["Polarity"].values, tokenizer, MAX_LEN)

    nw = 2 if cuda_ok else 0
    dl_kw = {"num_workers": nw, "pin_memory": cuda_ok}
    if nw > 0:
        dl_kw["persistent_workers"] = True
        dl_kw["prefetch_factor"] = 2

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, **dl_kw)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, **dl_kw)

    y_train = train_df["Polarity"].values.astype(int)
    cw = compute_class_weights(y_train, 3, NEUTRAL_LOSS_BOOST)
    print(f"Sınıf ağırlıkları (Neg, Neu, Pos) [Neutral boost={NEUTRAL_LOSS_BOOST}]: {cw.numpy().round(4)}")

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3)
    model.to(device)

    best = train_loop(model, train_loader, val_loader, device, use_amp, cw)
    print(f"\nEğitim bitti. En iyi val macro-F1: {best:.4f}")

    # İroni / negasyon testleri
    tests = [
        ("Yemek mükemmeldi dersem yalan olur", 0, "Negatif"),
        ("Servis berbattı dersem yalan olur", 2, "Pozitif"),
        ("Fiyatlar ucuz sayılmaz", 0, "Negatif"),
    ]
    print("\n" + "=" * 60)
    print("İRONİ / NEGASYON — olasılıklar (beklenen etiket referans)")
    print("=" * 60)
    for sent, exp_id, exp_name in tests:
        print_probs(model, tokenizer, device, sent, use_amp)
        print(f"  (Referans beklenen: {exp_name} / id={exp_id})")


if __name__ == "__main__":
    main()
