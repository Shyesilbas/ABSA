"""Integration tests for the FastAPI endpoints using TestClient.

These tests mock the model layer so they can run without a GPU or trained model.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

_ROOT = Path(__file__).resolve().parent.parent
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from fastapi.testclient import TestClient


def _make_mock_bundle():
    """Create a mock model/tokenizer/device triple."""
    model = MagicMock()
    logits = torch.tensor([[0.05, 0.10, 0.85]])
    model.return_value = MagicMock(logits=logits)
    model.eval = MagicMock()

    tokenizer = MagicMock()
    tokenizer.return_value = {
        "input_ids": torch.ones(1, 160, dtype=torch.long),
        "attention_mask": torch.ones(1, 160, dtype=torch.long),
    }

    device = torch.device("cpu")
    return model, tokenizer, device


@pytest.fixture()
def client():
    """Create a TestClient with mocked model loading."""
    mock_model, mock_tokenizer, mock_device = _make_mock_bundle()

    with (
        patch("backend.state.startup_load"),
        patch("backend.state.shutdown_clear"),
        patch("backend.state.model_ready", return_value=True),
        patch("backend.state.load_error", return_value=None),
        patch(
            "backend.state.get_model_bundle",
            return_value=(mock_model, mock_tokenizer, mock_device),
        ),
        patch(
            "backend.api.predict_sentence_with_meta",
            return_value=(
                "Positive",
                [0.05, 0.10, 0.85],
                {"confidence": 0.85, "raw_label": "Positive", "fallback_applied": False},
            ),
        ),
    ):
        from backend.main import app

        yield TestClient(app)


class TestHealthEndpoint:
    def test_health_returns_ok(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "ok"
        assert data["model_loaded"] is True

    def test_health_has_request_id(self, client):
        r = client.get("/health")
        assert "x-request-id" in r.headers


class TestMetaEndpoint:
    def test_meta_returns_model_info(self, client):
        r = client.get("/meta")
        assert r.status_code == 200
        data = r.json()
        assert "model_name" in data
        assert "class_names" in data
        assert isinstance(data["class_names"], list)
        assert "confidence_threshold" in data


class TestPredictEndpoint:
    def test_predict_single_text(self, client):
        r = client.post("/predict", json={"text": "Bu harika bir ürün!"})
        assert r.status_code == 200
        data = r.json()
        assert data["sentiment"] == "Positive"
        assert "confidence" in data
        assert "probabilities" in data
        assert isinstance(data["probabilities"], dict)

    def test_predict_empty_text_rejected(self, client):
        r = client.post("/predict", json={"text": ""})
        assert r.status_code == 422


class TestBatchPredictEndpoint:
    def test_batch_predict(self, client):
        items = [
            {"id": 0, "text": "Harika!"},
            {"id": 1, "text": "Kötü bir deneyim."},
        ]
        with patch(
            "backend.api.predict_batch_entries",
            return_value=[
                {
                    "id": 0,
                    "text": "Harika!",
                    "sentiment": "Positive",
                    "raw_sentiment": "Positive",
                    "fallback_applied": False,
                    "confidence": 0.95,
                },
                {
                    "id": 1,
                    "text": "Kötü bir deneyim.",
                    "sentiment": "Negative",
                    "raw_sentiment": "Negative",
                    "fallback_applied": False,
                    "confidence": 0.92,
                },
            ],
        ):
            r = client.post("/predict/batch", json={"items": items})
            assert r.status_code == 200
            data = r.json()
            assert "predictions" in data
            assert len(data["predictions"]) == 2

    def test_batch_empty_items_rejected(self, client):
        r = client.post("/predict/batch", json={"items": []})
        assert r.status_code == 422


class TestVisualizationEndpoints:
    def test_stats_with_rows(self, client):
        body = {"rows": [{"sentiment": "Positive"}, {"sentiment": "Negative"}]}
        r = client.post("/visualize/distribution/stats", json=body)
        assert r.status_code == 200
        data = r.json()
        assert data["total"] == 2
        assert "counts" in data

    def test_stats_both_fields_rejected(self, client):
        body = {"texts": ["a"], "rows": [{"sentiment": "Positive"}]}
        r = client.post("/visualize/distribution/stats", json=body)
        assert r.status_code == 422

    def test_stats_neither_field_rejected(self, client):
        r = client.post("/visualize/distribution/stats", json={})
        assert r.status_code == 422
