import os
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from config import (
    MODEL_NAME,
    MAX_LEN,
    CLASS_NAMES,
    MODEL_PATH,
    TEST_DATA_PATH,
    OUTPUTS_DIR,
)
from dataset_loader import SentenceClassificationDataset


@torch.no_grad()
def collect_predictions(model, loader, device):
    model.eval()
    all_y, all_p = [], []
    for batch in loader:
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        logits = model(input_ids=ids, attention_mask=mask).logits
        pred = logits.argmax(dim=-1)
        all_y.extend(labels.cpu().tolist())
        all_p.extend(pred.cpu().tolist())
    return np.array(all_y), np.array(all_p)


def plot_cm(cm, path):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.ylabel("Gerçek")
    plt.xlabel("Tahmin")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def main():
    if not os.path.isfile(MODEL_PATH):
        raise FileNotFoundError(MODEL_PATH)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    test_ds = SentenceClassificationDataset(tokenizer, MAX_LEN, csv_path=TEST_DATA_PATH)
    loader = DataLoader(test_ds, batch_size=32, shuffle=False)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=len(CLASS_NAMES)
    )
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)

    y_true, y_pred = collect_predictions(model, loader, device)

    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=4, zero_division=0))

    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    cm_path = os.path.join(OUTPUTS_DIR, "confusion_matrix.png")
    plot_cm(confusion_matrix(y_true, y_pred), cm_path)
    print(f"CM: {cm_path}")


if __name__ == "__main__":
    main()
