import torch
from transformers import BertTokenizer
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from auto_predict import SentimentClassifier

MODEL_NAME = 'dbmdz/bert-base-turkish-cased'
MAX_LEN = 128
class_names = ['Negative', 'Neutral', 'Positive']

def get_predictions(model, data_loader, device):

    model = model.eval()
    review_texts = []
    predictions = []
    prediction_probs = []
    real_values = []

    with torch.no_grad():
        for d in data_loader:
            texts = d["review_text"]
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            targets = d["targets"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            _, preds = torch.max(outputs, dim=1)

            review_texts.extend(texts)
            predictions.extend(preds)
            prediction_probs.extend(outputs)
            real_values.extend(targets)

    predictions = torch.stack(predictions).cpu()
    prediction_probs = torch.stack(prediction_probs).cpu()
    real_values = torch.stack(real_values).cpu()

    return review_texts, predictions, prediction_probs, real_values

def show_confusion_matrix(confusion_matrix):
    plt.figure(figsize=(8, 6))
    hmap = sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap="Blues")

    hmap.yaxis.set_ticklabels(hmap.yaxis.get_ticklabels(), rotation=0, ha='right')
    hmap.xaxis.set_ticklabels(hmap.xaxis.get_ticklabels(), rotation=30, ha='right')

    plt.ylabel('Actual Class')
    plt.xlabel('Predicted Class')
    plt.title('Confusion Matrix')
    plt.show()

if __name__ == "__main__":
    from dataset_loader import ABSADataset
    from torch.utils.data import DataLoader

    device = torch.device("cpu")
    base_path = os.path.dirname(os.path.dirname(__file__))
    model_path = os.path.join(base_path, 'models', 'best_model_state.bin')
    data_path = os.path.join(base_path, 'data', 'test.csv')

    print("Loading model and test data...")
    model = SentimentClassifier(n_classes=3)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)

    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    test_dataset = ABSADataset(data_path, tokenizer, MAX_LEN)
    test_data_loader = DataLoader(test_dataset, batch_size=16)

    print("Analyzing test set. May take a while")

    y_review_texts, y_pred, y_pred_probs, y_test = get_predictions(
        model,
        test_data_loader,
        device
    )

    print("\n" + "=" * 60)
    print("CLASSIFICATION REPORT")
    print("=" * 60)
    print(classification_report(y_test, y_pred, target_names=class_names))

    print("Generating Confusion Matrix plot...")
    cm = confusion_matrix(y_test, y_pred)
    show_confusion_matrix(cm)
