import pandas as pd
from sklearn.model_selection import train_test_split

from config import (
    RAW_DATA_PATH,
    DATA_DIR,
    TRAIN_DATA_PATH,
    VAL_DATA_PATH,
    TEST_DATA_PATH,
    RANDOM_SEED,
)
from data_contracts import prepare_sentence_polarity_frame, deduplicate_by_sentence_majority


def _sentence_level(df: pd.DataFrame) -> pd.DataFrame:
    out = prepare_sentence_polarity_frame(df, fill_missing_label=None)
    return deduplicate_by_sentence_majority(out)


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

    print("-" * 30)
    print(f"Train: {len(train_df)}")
    print(f"Val: {len(val_df)}")
    print(f"Test: {len(test_df)}")

    train_df.to_csv(TRAIN_DATA_PATH, index=False)
    val_df.to_csv(VAL_DATA_PATH, index=False)
    test_df.to_csv(TEST_DATA_PATH, index=False)

    print(f"Saved to: {DATA_DIR}")


if __name__ == "__main__":
    process_data()
