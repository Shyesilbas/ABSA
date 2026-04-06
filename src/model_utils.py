import os
from typing import Optional, Tuple, List

import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from config import MODEL_NAME, MODEL_PATH, CLASS_NAMES, MAX_LEN


def load_classifier(device: Optional[torch.device] = None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.isfile(MODEL_PATH):
        raise FileNotFoundError(f"Model bulunamadı: {MODEL_PATH}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=len(CLASS_NAMES)
    )
    state = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model, tokenizer, device


def load_model_resources(device: Optional[torch.device] = None):
    """Backend uyumu: (model, tokenizer, device)."""
    return load_classifier(device)


@torch.no_grad()
def predict_sentence(model, tokenizer, device, text: str) -> Tuple[str, List[float]]:
    enc = tokenizer(
        str(text).strip(),
        max_length=MAX_LEN,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    enc = {k: v.to(device) for k, v in enc.items()}
    logits = model(**enc).logits.float()
    probs = F.softmax(logits, dim=-1).squeeze(0)
    pred = int(probs.argmax().item())
    return CLASS_NAMES[pred], probs.cpu().tolist()


def predict_sentiment_single(model, tokenizer, device, text: str) -> Tuple[str, float]:
    """Tek cümle: (etiket, güven = max olasılık)."""
    label, probs = predict_sentence(model, tokenizer, device, text)
    return label, float(max(probs))
