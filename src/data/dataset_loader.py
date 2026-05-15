"""PyTorch Dataset for sentence-level sentiment classification.

Tokenises ``Sentence`` text into BERT-compatible tensors and returns
``input_ids``, ``attention_mask``, ``labels``, and ``sentence`` (raw text).
"""
from __future__ import annotations

from typing import Optional

import pandas as pd
import torch
from torch.utils.data import Dataset


class SentenceClassificationDataset(Dataset):
    """Sentence → (input_ids, attention_mask, labels) dataset.

    Parameters
    ----------
    tokenizer
        A HuggingFace tokenizer instance.
    max_len : int
        Maximum token length (padding / truncation target).
    csv_path : str | None
        Path to a CSV with ``Sentence`` and ``Polarity`` columns.
    dataframe : pd.DataFrame | None
        Pre-loaded DataFrame (mutually exclusive with *csv_path*).
    """

    def __init__(
        self,
        tokenizer,
        max_len: int,
        csv_path: Optional[str] = None,
        dataframe: Optional[pd.DataFrame] = None,
    ) -> None:
        if csv_path is not None:
            df = pd.read_csv(csv_path)
        elif dataframe is not None:
            df = dataframe.copy()
        else:
            raise ValueError("Either csv_path or dataframe must be provided.")

        df.columns = pd.Index([str(c).strip() for c in df.columns])  # type: ignore[assignment]
        if "Sentence" not in df.columns:
            raise KeyError(f"Missing 'Sentence' column. Got: {list(df.columns)}")

        self.sentences = df["Sentence"].astype(str).values
        self.labels = (
            df["Polarity"].astype(int).values
            if "Polarity" in df.columns
            else None
        )
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.sentences)

    def __getitem__(self, index: int) -> dict:
        text = str(self.sentences[index]).strip()
        encoding = self.tokenizer(
            text,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        item = {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "sentence": text,
        }

        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[index], dtype=torch.long)

        return item
