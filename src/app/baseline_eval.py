from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from torch.utils.data import DataLoader

from core.config import BATCH_SIZE, CLASS_NAMES, MAX_LEN, OUTPUTS_DIR, TEST_DATA_PATH, TRAIN_DATA_PATH
from core.progress import loader_total, track
from data.contracts import prepare_sentence_polarity_frame
from data.dataset_loader import SentenceClassificationDataset
from model.inference import load_classifier


@dataclass
class EvalResult:
    model: str
    accuracy: float
    macro_f1: float
    report: dict[str, Any]


def _load_split(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return prepare_sentence_polarity_frame(df, fill_missing_label=None)


def _evaluate(y_true: np.ndarray, y_pred: np.ndarray, model_name: str) -> EvalResult:
    report = classification_report(
        y_true,
        y_pred,
        target_names=CLASS_NAMES,
        digits=4,
        zero_division=0,
        output_dict=True,
    )
    return EvalResult(
        model=model_name,
        accuracy=float(accuracy_score(y_true, y_pred)),
        macro_f1=float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        report=report,
    )


def run_majority_baseline(y_train: np.ndarray, y_test: np.ndarray) -> EvalResult:
    majority = int(pd.Series(y_train).value_counts().idxmax())
    pred = np.full(shape=len(y_test), fill_value=majority, dtype=np.int64)
    return _evaluate(y_test, pred, "majority_class")


def run_tfidf_logreg_baseline(
    x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, y_test: np.ndarray
) -> EvalResult:
    vec = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        min_df=2,
        max_features=80_000,
    )
    xtr = vec.fit_transform(x_train)
    xte = vec.transform(x_test)

    clf = LogisticRegression(
        max_iter=1200,
        solver="liblinear",
        class_weight="balanced",
        random_state=42,
    )
    clf.fit(xtr, y_train)
    pred = clf.predict(xte)
    return _evaluate(y_test, pred, "tfidf_logreg")


@torch.no_grad()
def run_bert_model(x_test: pd.DataFrame) -> np.ndarray:
    model, tokenizer, device = load_classifier()
    ds = SentenceClassificationDataset(tokenizer, max_len=MAX_LEN, dataframe=x_test.reset_index(drop=True))
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)
    preds: list[int] = []

    n_batches = loader_total(loader)
    for batch in track(loader, total=n_batches, desc="BERT eval", unit="batch"):
        ids = batch["input_ids"].to(device, non_blocking=True)
        mask = batch["attention_mask"].to(device, non_blocking=True)
        logits = model(input_ids=ids, attention_mask=mask).logits
        preds.extend(logits.argmax(dim=-1).cpu().tolist())
    return np.asarray(preds, dtype=np.int64)


def save_outputs(results: list[EvalResult]) -> None:
    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    summary = pd.DataFrame(
        [
            {
                "model": r.model,
                "accuracy": round(r.accuracy, 6),
                "macro_f1": round(r.macro_f1, 6),
            }
            for r in results
        ]
    ).sort_values(["macro_f1", "accuracy"], ascending=False)
    summary_path = os.path.join(OUTPUTS_DIR, "baseline_comparison.csv")
    summary.to_csv(summary_path, index=False)

    detail_rows = []
    for r in results:
        for cls in CLASS_NAMES:
            c = r.report.get(cls, {})
            detail_rows.append(
                {
                    "model": r.model,
                    "class": cls,
                    "precision": round(float(c.get("precision", 0.0)), 6),
                    "recall": round(float(c.get("recall", 0.0)), 6),
                    "f1_score": round(float(c.get("f1-score", 0.0)), 6),
                    "support": int(c.get("support", 0)),
                }
            )
    details = pd.DataFrame(detail_rows)
    details_path = os.path.join(OUTPUTS_DIR, "baseline_class_reports.csv")
    details.to_csv(details_path, index=False)

    print("\n=== Baseline comparison summary ===")
    print(summary.to_string(index=False))
    print(f"\nSaved summary: {summary_path}")
    print(f"Saved class reports: {details_path}")


def main() -> None:
    train_df = _load_split(TRAIN_DATA_PATH)
    test_df = _load_split(TEST_DATA_PATH)

    x_train = train_df["Sentence"].values.astype(str)
    y_train = train_df["Polarity"].values.astype(np.int64)
    y_test = test_df["Polarity"].values.astype(np.int64)

    print(f"Train rows: {len(train_df)} | Test rows: {len(test_df)}")

    majority = run_majority_baseline(y_train, y_test)
    tfidf_lr = run_tfidf_logreg_baseline(x_train, y_train, test_df["Sentence"].values.astype(str), y_test)

    bert_pred = run_bert_model(test_df)
    bert = _evaluate(y_test, bert_pred, "bert_finetuned")

    save_outputs([majority, tfidf_lr, bert])


if __name__ == "__main__":
    main()
