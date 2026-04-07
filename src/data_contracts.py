from __future__ import annotations

import pandas as pd

TEXT_COL = "Sentence"
LABEL_COL = "Polarity"


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map: dict[str, str] = {}
    for col in df.columns:
        c = str(col).strip()
        low = c.lower()
        if low == "text":
            rename_map[col] = TEXT_COL
        elif low == "label":
            rename_map[col] = LABEL_COL
        elif c != col:
            rename_map[col] = c
    return df.rename(columns=rename_map)


def prepare_sentence_polarity_frame(
    df: pd.DataFrame,
    *,
    fill_missing_label: int | None = None,
    drop_aspect: bool = True,
) -> pd.DataFrame:
    out = normalize_columns(df.copy())
    if drop_aspect and "Aspect" in out.columns:
        out = out.drop(columns=["Aspect"])

    if TEXT_COL not in out.columns or LABEL_COL not in out.columns:
        raise ValueError(f"Missing required columns: {out.columns.tolist()}")

    out[TEXT_COL] = out[TEXT_COL].astype(str).str.strip()
    out[LABEL_COL] = pd.to_numeric(out[LABEL_COL], errors="coerce")
    out = out[out[TEXT_COL].str.len() > 0]

    if fill_missing_label is None:
        out = out.dropna(subset=[LABEL_COL])
    else:
        out[LABEL_COL] = out[LABEL_COL].fillna(fill_missing_label)

    out[LABEL_COL] = out[LABEL_COL].astype(int)
    return out[[TEXT_COL, LABEL_COL]]


def majority_label(s: pd.Series) -> int:
    return int(s.value_counts().idxmax())


def deduplicate_by_sentence_majority(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby(TEXT_COL, as_index=False)[LABEL_COL].agg(majority_label)
