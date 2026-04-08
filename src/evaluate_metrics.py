import os
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader

from config import (
    MODEL_NAME,
    MAX_LEN,
    CLASS_NAMES,
    MODEL_PATH,
    TEST_DATA_PATH,
    OUTPUTS_DIR,
    MISCLASSIFIED_REPORT_PATH,
    CONFUSION_PAIRS_REPORT_PATH,
)
from core.progress import loader_total, track
from data.dataset_loader import SentenceClassificationDataset
from model.loader import load_finetuned_resources


@torch.no_grad()
def collect_predictions(model, loader, device, *, progress_desc: str = "Test eval"):
    model.eval()
    all_y, all_p, all_s = [], [], []
    n_batches = loader_total(loader)
    for batch in track(loader, total=n_batches, desc=progress_desc, unit="batch"):
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        logits = model(input_ids=ids, attention_mask=mask).logits
        pred = logits.argmax(dim=-1)
        all_y.extend(labels.cpu().tolist())
        all_p.extend(pred.cpu().tolist())
        all_s.extend(batch["sentence"])
    return np.array(all_y), np.array(all_p), all_s


def plot_cm(cm, path):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.ylabel("True")
    plt.xlabel("Predicted")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def export_error_analysis(y_true, y_pred, sentences) -> None:
    rows = []
    for i, (yt, yp, s) in enumerate(zip(y_true, y_pred, sentences)):
        if int(yt) != int(yp):
            rows.append(
                {
                    "row_id": i,
                    "sentence": s,
                    "true_label": CLASS_NAMES[int(yt)],
                    "pred_label": CLASS_NAMES[int(yp)],
                }
            )
    miss_df = pd.DataFrame(rows)
    miss_df.to_csv(MISCLASSIFIED_REPORT_PATH, index=False)

    if len(miss_df) > 0:
        pair_df = (
            miss_df.groupby(["true_label", "pred_label"], as_index=False)
            .size()
            .rename(columns={"size": "count"})
            .sort_values("count", ascending=False)
        )
    else:
        pair_df = pd.DataFrame(columns=["true_label", "pred_label", "count"])
    pair_df.to_csv(CONFUSION_PAIRS_REPORT_PATH, index=False)
    print(f"Misclassified rows saved: {MISCLASSIFIED_REPORT_PATH}")
    print(f"Confusion pairs saved: {CONFUSION_PAIRS_REPORT_PATH}")


def main():
    if not os.path.isfile(MODEL_PATH):
        raise FileNotFoundError(MODEL_PATH)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer, device = load_finetuned_resources(
        model_name=MODEL_NAME,
        model_path=MODEL_PATH,
        class_names=CLASS_NAMES,
        device=device,
    )
    test_ds = SentenceClassificationDataset(tokenizer, MAX_LEN, csv_path=TEST_DATA_PATH)
    loader = DataLoader(test_ds, batch_size=32, shuffle=False)

    y_true, y_pred, sentences = collect_predictions(model, loader, device)

    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=4, zero_division=0))

    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    cm_path = os.path.join(OUTPUTS_DIR, "confusion_matrix.png")
    plot_cm(confusion_matrix(y_true, y_pred), cm_path)
    export_error_analysis(y_true, y_pred, sentences)
    print(f"Confusion matrix saved: {cm_path}")


if __name__ == "__main__":
    main()
