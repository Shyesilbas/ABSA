import json
import os

import pandas as pd
from sklearn.model_selection import train_test_split

from core.config import (
    DATA_DIR,
    LEAKAGE_GUARD_ENABLED,
    LEAKAGE_REPORT_PATH,
    RANDOM_SEED,
    RAW_DATA_PATH,
    TEST_DATA_PATH,
    TRAIN_DATA_PATH,
    VAL_DATA_PATH,
)
from data.contracts import deduplicate_by_sentence_majority, prepare_sentence_polarity_frame


def _sentence_level(df: pd.DataFrame) -> pd.DataFrame:
    out = prepare_sentence_polarity_frame(df, fill_missing_label=None)
    return deduplicate_by_sentence_majority(out)


def _overlap_count(a: pd.DataFrame, b: pd.DataFrame) -> int:
    return int(len(set(a["Sentence"]).intersection(set(b["Sentence"]))))


def process_data():
    df = pd.read_csv(RAW_DATA_PATH)
    df = _sentence_level(df)

    print(f"Sentence-level row count: {len(df)}")
    print("Polarity distribution:\n", df["Polarity"].value_counts())

    train_df, temp_df = train_test_split(
        df, test_size=0.2, random_state=RANDOM_SEED, stratify=df["Polarity"]
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, random_state=RANDOM_SEED, stratify=temp_df["Polarity"]
    )

    overlap_report = {
        "train_val_overlap": _overlap_count(train_df, val_df),
        "train_test_overlap": _overlap_count(train_df, test_df),
        "val_test_overlap": _overlap_count(val_df, test_df),
    }
    if LEAKAGE_GUARD_ENABLED and any(v > 0 for v in overlap_report.values()):
        raise ValueError(f"Data leakage detected between splits: {overlap_report}")

    print("-" * 30)
    print(f"Train: {len(train_df)}")
    print(f"Val: {len(val_df)}")
    print(f"Test: {len(test_df)}")

    train_df.to_csv(TRAIN_DATA_PATH, index=False)
    val_df.to_csv(VAL_DATA_PATH, index=False)
    test_df.to_csv(TEST_DATA_PATH, index=False)

    print(f"Saved to: {DATA_DIR}")
    os.makedirs(os.path.dirname(LEAKAGE_REPORT_PATH), exist_ok=True)
    with open(LEAKAGE_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(overlap_report, f, ensure_ascii=True, indent=2)
    print(f"Leakage report saved: {LEAKAGE_REPORT_PATH}")


if __name__ == "__main__":
    process_data()
