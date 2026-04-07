from typing import Optional, Tuple, List

import torch
import torch.nn.functional as F

from config import MODEL_NAME, MODEL_PATH, CLASS_NAMES, MAX_LEN
from model_loader import load_finetuned_resources


def load_classifier(device: Optional[torch.device] = None):
    return load_finetuned_resources(
        model_name=MODEL_NAME,
        model_path=MODEL_PATH,
        class_names=CLASS_NAMES,
        device=device,
    )


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
