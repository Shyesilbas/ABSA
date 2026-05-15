"""Unit tests for data.contracts — column normalisation, polarity mapping, and deduplication."""
from __future__ import annotations

import pandas as pd
import pytest

from data.contracts import (
    deduplicate_by_sentence_majority,
    prepare_sentence_polarity_frame,
)


# ── prepare_sentence_polarity_frame ─────────────────────────────────────────


class TestPrepareSentencePolarityFrame:
    """Tests for prepare_sentence_polarity_frame."""

    def test_standard_columns(self, sample_sentence_polarity_df: pd.DataFrame):
        result = prepare_sentence_polarity_frame(sample_sentence_polarity_df)
        assert list(result.columns) == ["Sentence", "Polarity"]
        assert len(result) == 5
        assert result["Polarity"].dtype == int

    def test_alias_columns(self, sample_raw_df_with_aliases: pd.DataFrame):
        """Column aliases (tweet, sentiment) should be resolved."""
        result = prepare_sentence_polarity_frame(sample_raw_df_with_aliases)
        assert "Sentence" in result.columns
        assert "Polarity" in result.columns
        assert len(result) == 3

    def test_string_polarity_mapping(self):
        df = pd.DataFrame(
            {
                "text": ["a", "b", "c", "d", "e", "f"],
                "label": ["positive", "negative", "neutral", "pos", "neg", "neu"],
            }
        )
        result = prepare_sentence_polarity_frame(df)
        assert list(result["Polarity"]) == [2, 0, 1, 2, 0, 1]

    def test_turkish_polarity_mapping(self):
        df = pd.DataFrame(
            {
                "metin": ["iyi", "kötü", "normal"],
                "polarity": ["olumlu", "olumsuz", "notr"],
            }
        )
        result = prepare_sentence_polarity_frame(df)
        assert list(result["Polarity"]) == [2, 0, 1]

    def test_numeric_polarity_passthrough(self):
        df = pd.DataFrame({"sentence": ["a", "b", "c"], "polarity": [0, 1, 2]})
        result = prepare_sentence_polarity_frame(df)
        assert list(result["Polarity"]) == [0, 1, 2]

    def test_empty_sentences_dropped(self):
        df = pd.DataFrame({"sentence": ["hello", "", "   "], "polarity": [1, 1, 1]})
        result = prepare_sentence_polarity_frame(df)
        assert len(result) == 1

    def test_missing_column_raises_error(self):
        df = pd.DataFrame({"unrelated_col": ["a"], "another_col": [1]})
        with pytest.raises(KeyError, match="Cannot find"):
            prepare_sentence_polarity_frame(df)

    def test_fill_missing_label_none_drops_rows(self):
        df = pd.DataFrame(
            {"sentence": ["a", "b"], "polarity": [1, "unknown_value"]}
        )
        result = prepare_sentence_polarity_frame(df, fill_missing_label=None)
        assert len(result) == 1

    def test_fill_missing_label_default(self):
        df = pd.DataFrame(
            {"sentence": ["a", "b"], "polarity": [0, "unknown_value"]}
        )
        result = prepare_sentence_polarity_frame(df, fill_missing_label=1)
        assert len(result) == 2
        assert result["Polarity"].iloc[1] == 1


# ── deduplicate_by_sentence_majority ────────────────────────────────────────


class TestDeduplicateBySentenceMajority:
    """Tests for deduplicate_by_sentence_majority."""

    def test_majority_vote(self):
        df = pd.DataFrame(
            {
                "Sentence": ["same", "same", "same"],
                "Polarity": [0, 1, 1],
            }
        )
        result = deduplicate_by_sentence_majority(df)
        assert len(result) == 1
        assert result["Polarity"].iloc[0] == 1  # majority wins

    def test_no_duplicates(self, sample_sentence_polarity_df: pd.DataFrame):
        result = deduplicate_by_sentence_majority(sample_sentence_polarity_df)
        assert len(result) == len(sample_sentence_polarity_df)

    def test_empty_dataframe(self):
        df = pd.DataFrame(columns=["Sentence", "Polarity"])
        result = deduplicate_by_sentence_majority(df)
        assert len(result) == 0

    def test_tie_keeps_first(self):
        """When counts are tied, first occurrence wins."""
        df = pd.DataFrame(
            {
                "Sentence": ["tie", "tie"],
                "Polarity": [0, 2],
            }
        )
        result = deduplicate_by_sentence_majority(df)
        assert len(result) == 1
        # mode() picks first in tied case
        assert result["Polarity"].iloc[0] in (0, 2)
