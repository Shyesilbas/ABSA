"""Unit tests for model.inference — predict helpers and fallback logic."""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import torch
import pytest

_ROOT = Path(__file__).resolve().parent.parent
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from model.inference import logits_to_final_class_ids


class TestLogitsToFinalClassIds:
    """Tests for the confidence fallback logic in logits_to_final_class_ids."""

    @patch("model.inference.CONFIDENCE_FALLBACK_ENABLED", False)
    def test_fallback_disabled_returns_argmax(self):
        logits = torch.tensor([[0.1, 0.2, 0.9]])  # Positive
        result = logits_to_final_class_ids(logits)
        assert result.item() == 2  # Positive index

    @patch("model.inference.CONFIDENCE_FALLBACK_ENABLED", True)
    @patch("model.inference.CONFIDENCE_THRESHOLD", 0.99)
    @patch("model.inference.CLASS_NAMES", ["Negative", "Neutral", "Positive"])
    @patch("model.inference.CONFIDENCE_FALLBACK_LABEL", "Neutral")
    def test_fallback_applied_low_confidence(self):
        # softmax of these won't exceed 0.99
        logits = torch.tensor([[0.3, 0.3, 0.4]])
        result = logits_to_final_class_ids(logits)
        assert result.item() == 1  # Falls back to Neutral

    @patch("model.inference.CONFIDENCE_FALLBACK_ENABLED", True)
    @patch("model.inference.CONFIDENCE_THRESHOLD", 0.5)
    @patch("model.inference.CLASS_NAMES", ["Negative", "Neutral", "Positive"])
    @patch("model.inference.CONFIDENCE_FALLBACK_LABEL", "Neutral")
    def test_fallback_not_applied_high_confidence(self):
        logits = torch.tensor([[0.0, 0.0, 5.0]])  # Very high Positive
        result = logits_to_final_class_ids(logits)
        assert result.item() == 2  # Stays Positive

    @patch("model.inference.CONFIDENCE_FALLBACK_ENABLED", True)
    @patch("model.inference.CONFIDENCE_THRESHOLD", 0.99)
    @patch("model.inference.CLASS_NAMES", ["Negative", "Neutral", "Positive"])
    @patch("model.inference.CONFIDENCE_FALLBACK_LABEL", "Neutral")
    def test_batch_fallback(self):
        logits = torch.tensor([
            [0.0, 0.0, 10.0],  # High confidence → stays
            [0.3, 0.3, 0.4],   # Low confidence → fallback
        ])
        result = logits_to_final_class_ids(logits)
        assert result[0].item() == 2  # Positive stays
        assert result[1].item() == 1  # Falls back to Neutral

    def test_single_sample_shape(self):
        logits = torch.tensor([[1.0, 2.0, 3.0]])
        result = logits_to_final_class_ids(logits)
        assert result.shape == (1,)

    def test_multi_sample_shape(self):
        logits = torch.randn(16, 3)
        result = logits_to_final_class_ids(logits)
        assert result.shape == (16,)
