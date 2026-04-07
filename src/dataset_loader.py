import torch
from torch.utils.data import Dataset
import pandas as pd

from data_contracts import prepare_sentence_polarity_frame


class SentenceClassificationDataset(Dataset):
    """Sentence-level dataset with Sentence and Polarity columns."""

    def __init__(self, tokenizer, max_len: int, csv_path=None, dataframe=None):
        if (csv_path is None) == (dataframe is None):
            raise ValueError("Provide exactly one of csv_path or dataframe.")
        if dataframe is not None:
            self.df = dataframe
        else:
            self.df = pd.read_csv(csv_path)
        self.df = prepare_sentence_polarity_frame(self.df, fill_missing_label=None)
        self.texts = self.df["Sentence"].values
        self.labels = self.df["Polarity"].astype(int).values
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        y = int(self.labels[idx])
        enc = self.tokenizer(
            text,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "sentence": text,
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(y, dtype=torch.long),
        }
