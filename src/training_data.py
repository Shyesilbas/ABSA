"""Training pool builder from local CSV, raw ABSA, and optional HF subset."""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
from datasets import load_dataset

from data_contracts import (
    prepare_sentence_polarity_frame,
    deduplicate_by_sentence_majority,
)

def _prep_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    out = prepare_sentence_polarity_frame(df, fill_missing_label=1)
    return deduplicate_by_sentence_majority(out)


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
    raise KeyError(f"No text field found in example keys: {list(example.keys())}")


def load_hf_subset(dataset_id: str, n: int, seed: int) -> pd.DataFrame:
    print(f"Downloading Hugging Face dataset: {dataset_id} ...")
    ds = load_dataset(dataset_id, split="train")
    ds = ds.shuffle(seed=seed)
    ds = ds.select(range(min(n, len(ds))))
    label_key = None
    for c in ds.column_names:
        if c.lower() in ("label", "labels", "sentiment", "polarity", "class"):
            label_key = c
            break
    if label_key is None:
        raise RuntimeError(f"No label column found: {ds.column_names}")

    rows = []
    for i in range(len(ds)):
        ex = ds[i]
        text = _hf_text_field(ex)
        if len(text) < 1:
            continue
        rows.append({"Sentence": text, "Polarity": map_hf_label(ex.get(label_key))})
    return pd.DataFrame(rows)


def _merge_hard_overrides(train_pool: pd.DataFrame, hard_path: str) -> pd.DataFrame:
    """Hard example labels override duplicates from the training pool."""
    hard_df = _prep_csv(hard_path)
    drop_s = set(hard_df["Sentence"])
    out = train_pool[~train_pool["Sentence"].isin(drop_s)]
    out = pd.concat([out, hard_df], ignore_index=True)
    print(f"Merged hard_examples rows: {len(hard_df)} (hard labels win on duplicates).")
    return out


def build_train_val_frames(
    train_path: str,
    val_path: str,
    raw_path: str,
    merge_raw: bool,
    use_hf: bool,
    hf_dataset_id: str,
    hf_sample_size: int,
    hf_seed: int,
    hard_path: str | None = None,
    merge_hard: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_a = _prep_csv(train_path)
    parts = [train_a]
    if merge_raw and os.path.isfile(raw_path):
        parts.append(_prep_csv(raw_path))
    elif merge_raw:
        print(f"Warning: raw ABSA file not found, skipping: {raw_path}")

    train_pool = pd.concat(parts, ignore_index=True)
    train_pool = deduplicate_by_sentence_majority(train_pool)

    if use_hf:
        hf_df = load_hf_subset(hf_dataset_id, hf_sample_size, hf_seed)
        train_pool = pd.concat([train_pool, hf_df], ignore_index=True)
        train_pool = deduplicate_by_sentence_majority(train_pool)

    if merge_hard and hard_path and os.path.isfile(hard_path):
        train_pool = _merge_hard_overrides(train_pool, hard_path)

    val_df = _prep_csv(val_path)
    return train_pool, val_df
