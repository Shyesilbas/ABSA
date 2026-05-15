"""Training pool construction — merge, enrich, and guard against leakage.

Builds the final (train_df, val_df) pair by:
1. Loading the canonical train / val CSVs.
2. Optionally merging the raw ABSA-oriented source into training.
3. Optionally sampling from a Hugging Face dataset for extra coverage.
4. Optionally overriding labels from a curated hard-examples file.
5. Running a leakage guard (train ∩ val sentence overlap check).
"""
from __future__ import annotations

import json
import os
from typing import Optional

import pandas as pd

from core.config import LEAKAGE_REPORT_PATH
from core.progress import track
from data.contracts import (
    deduplicate_by_sentence_majority,
    prepare_sentence_polarity_frame,
)


# ── Helpers ─────────────────────────────────────────────────────────────────

def _prep_csv(path: str) -> pd.DataFrame:
    """Read a CSV and normalise to Sentence / Polarity."""
    df = pd.read_csv(path)
    return prepare_sentence_polarity_frame(df)


def load_hf_subset(
    dataset_id: str,
    sample_size: int,
    seed: int,
) -> pd.DataFrame:
    """Download a HuggingFace dataset split and return a sampled DataFrame."""
    from datasets import load_dataset  # lazy import — optional dependency

    ds = load_dataset(dataset_id, split="train")
    if len(ds) > sample_size:
        ds = ds.shuffle(seed=seed).select(range(sample_size))

    df = pd.DataFrame(ds)
    return prepare_sentence_polarity_frame(df)


def _merge_hard_overrides(
    train_df: pd.DataFrame,
    hard_path: str,
) -> pd.DataFrame:
    """Merge curated hard examples into training; hard labels override."""
    if not os.path.isfile(hard_path):
        print(f"Hard examples file not found, skipping: {hard_path}")
        return train_df

    hard = _prep_csv(hard_path)
    if hard.empty:
        return train_df

    # For sentences that appear in both, hard label wins
    existing = set(train_df["Sentence"].str.lower().str.strip())
    hard_new = hard[~hard["Sentence"].str.lower().str.strip().isin(existing)]

    # Override: drop train rows that match hard sentences, then append hard
    hard_sents_lower = set(hard["Sentence"].str.lower().str.strip())
    kept = train_df[~train_df["Sentence"].str.lower().str.strip().isin(hard_sents_lower)]
    merged = pd.concat([kept, hard], ignore_index=True)

    print(
        f"Hard-example merge: {len(hard)} hard rows "
        f"({len(hard) - len(hard_new)} overrides, {len(hard_new)} new). "
        f"Training pool: {len(train_df)} → {len(merged)}"
    )
    return merged


def _check_leakage(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
) -> dict:
    """Report sentence overlap between train and val pools."""
    train_sents = set(train_df["Sentence"].str.lower().str.strip())
    val_sents = set(val_df["Sentence"].str.lower().str.strip())
    overlap = train_sents & val_sents
    report = {
        "train_size": len(train_df),
        "val_size": len(val_df),
        "overlap_count": len(overlap),
        "overlap_ratio_of_val": round(len(overlap) / max(len(val_sents), 1), 4),
        "overlap_samples": sorted(list(overlap))[:20],
    }
    return report


# ── Public API ──────────────────────────────────────────────────────────────

def build_train_val_frames(
    train_path: str,
    val_path: str,
    raw_path: str,
    merge_raw_absa: bool,
    use_hf_extra: bool,
    hf_dataset_id: str,
    hf_sample_size: int,
    hf_seed: int,
    *,
    hard_path: Optional[str] = None,
    merge_hard: bool = False,
    leakage_guard: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the final training and validation DataFrames.

    Returns
    -------
    (train_df, val_df) : tuple[pd.DataFrame, pd.DataFrame]
        Both with ``Sentence`` (str) and ``Polarity`` (int) columns.
    """
    # 1. Load canonical splits
    train_df = _prep_csv(train_path)
    val_df = _prep_csv(val_path)
    print(f"Base splits loaded — train: {len(train_df)}, val: {len(val_df)}")

    # 2. Optionally merge raw ABSA source into training
    if merge_raw_absa and os.path.isfile(raw_path):
        raw_df = _prep_csv(raw_path)
        before = len(train_df)
        train_df = pd.concat([train_df, raw_df], ignore_index=True)
        train_df = deduplicate_by_sentence_majority(train_df)
        print(f"Raw ABSA merged: {before} → {len(train_df)} (after dedup)")

    # 3. Optionally add HuggingFace extra data
    if use_hf_extra:
        print(f"Loading HF dataset: {hf_dataset_id} (sample={hf_sample_size}) ...")
        hf_df = load_hf_subset(hf_dataset_id, hf_sample_size, hf_seed)
        before = len(train_df)
        train_df = pd.concat([train_df, hf_df], ignore_index=True)
        train_df = deduplicate_by_sentence_majority(train_df)
        print(f"HF data merged: {before} → {len(train_df)} (after dedup)")

    # 4. Optionally merge hard examples (overrides)
    if merge_hard and hard_path:
        train_df = _merge_hard_overrides(train_df, hard_path)

    # 5. Final dedup
    train_df = deduplicate_by_sentence_majority(train_df)

    # 6. Leakage guard
    if leakage_guard:
        report = _check_leakage(train_df, val_df)
        overlap_n = report["overlap_count"]
        if overlap_n > 0:
            print(
                f"⚠ Leakage warning: {overlap_n} sentences overlap "
                f"between train and val ({report['overlap_ratio_of_val']:.1%} of val)."
            )
            # Remove overlapping sentences from training
            val_sents = set(val_df["Sentence"].str.lower().str.strip())
            train_df = train_df[
                ~train_df["Sentence"].str.lower().str.strip().isin(val_sents)
            ].reset_index(drop=True)
            print(f"Removed {overlap_n} leaking sentences from train → {len(train_df)}")
        else:
            print("Leakage guard: no overlap detected ✓")

        # Save report
        os.makedirs(os.path.dirname(LEAKAGE_REPORT_PATH), exist_ok=True)
        with open(LEAKAGE_REPORT_PATH, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)
