"""Unit tests for backend.schemas — Pydantic validation rules."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Ensure imports work
_ROOT = Path(__file__).resolve().parent.parent
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from backend.schemas import (
    MAX_BATCH_ITEMS,
    MAX_VISUALIZE_TEXTS,
    BatchItem,
    BatchPredictRequest,
    DistributionStatsResponse,
    HealthResponse,
    MetaResponse,
    PredictRequest,
    PredictResponse,
    SentimentRow,
    VisualizeDistributionRequest,
)


class TestPredictRequest:
    def test_valid_text(self):
        req = PredictRequest(text="Bu harika!")
        assert req.text == "Bu harika!"

    def test_empty_text_rejected(self):
        with pytest.raises(Exception):
            PredictRequest(text="")

    def test_max_length_enforced(self):
        req = PredictRequest(text="a" * 8000)
        assert len(req.text) == 8000
        with pytest.raises(Exception):
            PredictRequest(text="a" * 8001)


class TestBatchItem:
    def test_valid_item(self):
        item = BatchItem(id=1, text="Merhaba")
        assert item.id == 1
        assert item.text == "Merhaba"

    def test_optional_id(self):
        item = BatchItem(text="Test")
        assert item.id is None

    def test_empty_text_rejected(self):
        with pytest.raises(Exception):
            BatchItem(text="")


class TestBatchPredictRequest:
    def test_valid_request(self):
        items = [BatchItem(text="a"), BatchItem(text="b")]
        req = BatchPredictRequest(items=items)
        assert len(req.items) == 2

    def test_empty_items_rejected(self):
        with pytest.raises(Exception):
            BatchPredictRequest(items=[])

    def test_max_items_enforced(self):
        items = [BatchItem(text=f"item {i}") for i in range(MAX_BATCH_ITEMS + 1)]
        with pytest.raises(Exception):
            BatchPredictRequest(items=items)


class TestVisualizeDistributionRequest:
    def test_texts_only_valid(self):
        req = VisualizeDistributionRequest(texts=["hello", "world"])
        assert req.texts is not None
        assert req.rows is None

    def test_rows_only_valid(self):
        req = VisualizeDistributionRequest(
            rows=[SentimentRow(sentiment="Positive")]
        )
        assert req.rows is not None
        assert req.texts is None

    def test_both_fields_rejected(self):
        with pytest.raises(Exception, match="Exactly one"):
            VisualizeDistributionRequest(
                texts=["a"], rows=[SentimentRow(sentiment="Positive")]
            )

    def test_neither_field_rejected(self):
        with pytest.raises(Exception, match="Exactly one"):
            VisualizeDistributionRequest()

    def test_max_texts_enforced(self):
        with pytest.raises(Exception):
            VisualizeDistributionRequest(
                texts=["a"] * (MAX_VISUALIZE_TEXTS + 1)
            )

    def test_per_text_max_length(self):
        with pytest.raises(Exception, match="8000"):
            VisualizeDistributionRequest(texts=["a" * 8001])


class TestResponseModels:
    def test_health_response(self):
        r = HealthResponse(status="ok", model_loaded=True)
        assert r.status == "ok"
        assert r.detail is None

    def test_predict_response(self):
        r = PredictResponse(
            sentiment="Positive",
            raw_sentiment="Positive",
            confidence=0.95,
            fallback_applied=False,
            probabilities={"Negative": 0.02, "Neutral": 0.03, "Positive": 0.95},
        )
        assert r.sentiment == "Positive"

    def test_meta_response(self):
        r = MetaResponse(
            model_name="test",
            class_names=["Negative", "Neutral", "Positive"],
            confidence_fallback_enabled=True,
            confidence_threshold=0.7,
            confidence_fallback_label="Neutral",
        )
        assert len(r.class_names) == 3

    def test_distribution_stats_response(self):
        r = DistributionStatsResponse(
            counts={"Positive": 10, "Negative": 5},
            total=15,
            topic_title="Test",
            keywords_subtitle="kw",
            source="rows",
        )
        assert r.total == 15
