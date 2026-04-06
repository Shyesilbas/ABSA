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


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    if "text" in df.columns and "Sentence" not in df.columns:
        df = df.rename(columns={"text": "Sentence"})
    if "label" in df.columns and "Polarity" not in df.columns:
        df = df.rename(columns={"label": "Polarity"})
    return df


def _sentence_level(df: pd.DataFrame) -> pd.DataFrame:
    """Aynı cümle için birden fazla aspect satırı varsa Polarity çoğunluk oyu."""
    df = _normalize_columns(df)
    if "Aspect" in df.columns:
        df = df.drop(columns=["Aspect"])
    df = df.dropna(subset=["Sentence", "Polarity"])
    df["Sentence"] = df["Sentence"].astype(str).str.strip()
    df["Polarity"] = df["Polarity"].astype(int)

    def majority_polarity(s: pd.Series) -> int:
        m = s.mode()
        return int(m.iloc[0])

    out = df.groupby("Sentence", as_index=False).agg(Polarity=("Polarity", majority_polarity))
    return out


def process_data():
    df = pd.read_csv(RAW_DATA_PATH)
    df = _sentence_level(df)

    print(f"Cümle düzeyi satır sayısı: {len(df)}")
    print("Polarity dağılımı:\n", df["Polarity"].value_counts())

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

    print(f"Kayıt: {DATA_DIR}")


if __name__ == "__main__":
    process_data()
