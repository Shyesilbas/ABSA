"""Data format contracts — column normalisation and deduplication.

Every raw source is mapped to a shared schema:
    Sentence  (str)   – cleaned text
    Polarity  (int)   – 0 = Negative, 1 = Neutral, 2 = Positive
"""
from __future__ import annotations

from typing import Optional

import pandas as pd

# ── Column aliases ──────────────────────────────────────────────────────────
_SENTENCE_ALIASES = [
    "sentence", "text", "tweet", "tweets", "content",
    "metin", "yorum", "review", "message",
]
_POLARITY_ALIASES = ["polarity", "label", "sentiment"]

_POLARITY_MAP = {
    "negative": 0, "neg": 0, "olumsuz": 0,
    "neutral": 1, "neu": 1, "notr": 1, "nötr": 1,
    "positive": 2, "pos": 2, "olumlu": 2,
}


def _resolve_column(df: pd.DataFrame, candidates: list[str], target: str) -> str:
    """Return the first matching column name (case-insensitive) or *target*."""
    low = {c.lower(): c for c in df.columns}
    for alias in candidates:
        if alias in low:
            return low[alias]
    if target.lower() in low:
        return low[target.lower()]
    raise KeyError(
        f"Cannot find '{target}' column. "
        f"Expected one of {candidates}, got {list(df.columns)}"
    )


def _normalise_polarity(
    series: pd.Series, fill_missing: Optional[int]
) -> pd.Series:
    """Map string / mixed polarity values to int (0 / 1 / 2)."""
    def _map(val):  # noqa: ANN001, ANN202
        if pd.isna(val):
            return fill_missing
        if isinstance(val, (int, float)):
            v = int(val)
            if v in (0, 1, 2):
                return v
        s = str(val).strip().lower()
        if s in _POLARITY_MAP:
            return _POLARITY_MAP[s]
        try:
            v = int(float(s))
            if v in (0, 1, 2):
                return v
        except (ValueError, TypeError):
            pass
        return fill_missing

    return series.map(_map)


def prepare_sentence_polarity_frame(
    df: pd.DataFrame,
    fill_missing_label: Optional[int] = 1,
) -> pd.DataFrame:
    """Normalise any compatible DataFrame to ``Sentence`` + ``Polarity`` schema.

    Parameters
    ----------
    df : pd.DataFrame
        Input with at least one text column and one label column.
    fill_missing_label : int | None
        Value used when a polarity cell cannot be parsed.
        ``None`` → drop the row instead.

    Returns
    -------
    pd.DataFrame
        Columns: ``Sentence`` (str), ``Polarity`` (int).
    """
    out = df.copy()
    out.columns = pd.Index([str(c).strip() for c in out.columns])  # type: ignore[assignment]

    sent_col = _resolve_column(out, _SENTENCE_ALIASES, "Sentence")
    pol_col = _resolve_column(out, _POLARITY_ALIASES, "Polarity")

    out = out.rename(columns={sent_col: "Sentence", pol_col: "Polarity"})
    out["Sentence"] = out["Sentence"].astype(str).str.strip()
    out["Polarity"] = _normalise_polarity(out["Polarity"], fill_missing_label)

    # Drop rows where sentence is empty or polarity is unresolvable
    out = out[out["Sentence"].str.len() > 0]
    if fill_missing_label is None:
        out = out.dropna(subset=["Polarity"])  # type: ignore[call-overload]
    out["Polarity"] = out["Polarity"].astype(int)

    return out[["Sentence", "Polarity"]].reset_index(drop=True)


def deduplicate_by_sentence_majority(df: pd.DataFrame) -> pd.DataFrame:
    """When the same sentence appears multiple times, keep the majority label.

    Ties are broken by keeping the first occurrence.
    """
    if df.empty:
        return df

    majority = (
        df.groupby("Sentence", sort=False)["Polarity"]
        .agg(lambda s: s.mode().iloc[0])
        .reset_index()
    )
    result = pd.DataFrame(majority[["Sentence", "Polarity"]]).reset_index(drop=True)
    return result
