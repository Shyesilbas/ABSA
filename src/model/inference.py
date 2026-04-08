from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F

from config import (
    CLASS_NAMES,
    CONFIDENCE_FALLBACK_ENABLED,
    CONFIDENCE_FALLBACK_LABEL,
    CONFIDENCE_THRESHOLD,
    MAX_LEN,
    MODEL_NAME,
    MODEL_PATH,
)
from model.loader import load_finetuned_resources


def load_classifier(device: Optional[torch.device] = None):
    return load_finetuned_resources(
        model_name=MODEL_NAME,
        model_path=MODEL_PATH,
        class_names=CLASS_NAMES,
        device=device,
    )


@torch.no_grad()
def _predict_core(model, tokenizer, device, text: str) -> dict:
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
    raw_pred_idx = int(probs.argmax().item())
    confidence = float(probs.max().item())

    fallback_idx = raw_pred_idx
    if CONFIDENCE_FALLBACK_ENABLED and confidence < float(CONFIDENCE_THRESHOLD):
        if CONFIDENCE_FALLBACK_LABEL in CLASS_NAMES:
            fallback_idx = CLASS_NAMES.index(CONFIDENCE_FALLBACK_LABEL)
        elif "Neutral" in CLASS_NAMES:
            fallback_idx = CLASS_NAMES.index("Neutral")
        else:
            fallback_idx = raw_pred_idx

    final_pred_idx = fallback_idx
    return {
        "label": CLASS_NAMES[final_pred_idx],
        "probs": probs.cpu().tolist(),
        "confidence": confidence,
        "raw_label": CLASS_NAMES[raw_pred_idx],
        "fallback_applied": final_pred_idx != raw_pred_idx,
    }


@torch.no_grad()
def predict_sentence(model, tokenizer, device, text: str) -> Tuple[str, List[float]]:
    out = _predict_core(model, tokenizer, device, text)
    return out["label"], out["probs"]


@torch.no_grad()
def predict_sentence_with_meta(model, tokenizer, device, text: str):
    out = _predict_core(model, tokenizer, device, text)
    return out["label"], out["probs"], {
        "confidence": out["confidence"],
        "raw_label": out["raw_label"],
        "fallback_applied": out["fallback_applied"],
    }
