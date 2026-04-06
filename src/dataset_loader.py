import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import pandas as pd


class SentenceClassificationDataset(Dataset):
    """Cümle düzeyi: Sentence + Polarity (0,1,2)."""

    def __init__(self, tokenizer, max_len: int, csv_path=None, dataframe=None):
        if (csv_path is None) == (dataframe is None):
            raise ValueError("csv_path veya dataframe verin (yalnızca biri).")
        if dataframe is not None:
            self.df = dataframe
        else:
            self.df = pd.read_csv(csv_path)
        if "Aspect" in self.df.columns:
            self.df = self.df.drop(columns=["Aspect"])
        self.df = self.df.dropna(subset=["Sentence", "Polarity"])
        self.df["Sentence"] = self.df["Sentence"].astype(str).str.strip()
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
