"""Backward-compatible exports for data contracts."""

from data.contracts import (
    LABEL_COL,
    TEXT_COL,
    deduplicate_by_sentence_majority,
    majority_label,
    normalize_columns,
    prepare_sentence_polarity_frame,
)

__all__ = [
    "TEXT_COL",
    "LABEL_COL",
    "normalize_columns",
    "prepare_sentence_polarity_frame",
    "majority_label",
    "deduplicate_by_sentence_majority",
]
