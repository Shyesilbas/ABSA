"""Shared test fixtures for the Turkish Sentiment Analysis test suite."""
from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest

# Ensure src/ is importable in test context.
_ROOT = Path(__file__).resolve().parent.parent
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# Suppress noisy HuggingFace warnings in tests
os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


@pytest.fixture()
def sample_sentence_polarity_df() -> pd.DataFrame:
    """A small DataFrame with Sentence + Polarity columns."""
    return pd.DataFrame(
        {
            "Sentence": [
                "Bu ürün harika!",
                "Kötü bir deneyimdi.",
                "Normal bir hizmet.",
                "Çok memnun kaldım.",
                "Hiç beğenmedim.",
            ],
            "Polarity": [2, 0, 1, 2, 0],
        }
    )


@pytest.fixture()
def sample_raw_df_with_aliases() -> pd.DataFrame:
    """A DataFrame using non-standard column aliases."""
    return pd.DataFrame(
        {
            "tweet": [
                "Harika bir gün!",
                "Kargo çok geç geldi.",
                "İdare eder.",
            ],
            "sentiment": ["positive", "negative", "neutral"],
        }
    )


@pytest.fixture()
def sample_batch_entries() -> list[dict]:
    """Sample batch entries for batch prediction."""
    return [
        {"id": 0, "text": "Bu ürün harika!"},
        {"id": 1, "text": "Kötü bir deneyimdi."},
        {"id": 2, "text": "Normal bir hizmet."},
    ]


@pytest.fixture()
def mock_model():
    """A mock model that returns fake logits."""
    import torch

    model = MagicMock()
    # Return logits for 3 classes
    logits = torch.tensor([[0.1, 0.2, 0.9]])  # predicts Positive
    model.return_value = MagicMock(logits=logits)
    model.eval = MagicMock()
    model.to = MagicMock(return_value=model)
    model.config = MagicMock(hidden_size=768)
    return model


@pytest.fixture()
def mock_tokenizer():
    """A mock tokenizer that returns fake encodings."""
    import torch

    tokenizer = MagicMock()
    encoding = {
        "input_ids": torch.ones(1, 160, dtype=torch.long),
        "attention_mask": torch.ones(1, 160, dtype=torch.long),
    }
    tokenizer.return_value = encoding
    return tokenizer
