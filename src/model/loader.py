from __future__ import annotations

import os
from typing import Sequence

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def _resolve_device(device: torch.device | None) -> torch.device:
    if device is not None:
        return device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _validate_state_dict(
    state: dict,
    *,
    hidden_size: int,
    num_labels: int,
) -> None:
    if "classifier.weight" not in state or "classifier.bias" not in state:
        raise ValueError("State dict must contain classifier.weight and classifier.bias.")

    weight_shape = tuple(state["classifier.weight"].shape)
    bias_shape = tuple(state["classifier.bias"].shape)
    expected_weight = (num_labels, hidden_size)
    expected_bias = (num_labels,)

    if weight_shape != expected_weight or bias_shape != expected_bias:
        raise ValueError(
            f"Classifier shape mismatch. Expected {expected_weight}/{expected_bias}, "
            f"got {weight_shape}/{bias_shape}."
        )


def load_finetuned_resources(
    model_name: str,
    model_path: str,
    class_names: Sequence[str],
    device: torch.device | None = None,
):
    runtime_device = _resolve_device(device)
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(class_names),
    )

    state = torch.load(model_path, map_location=runtime_device)
    if not isinstance(state, dict):
        raise ValueError(f"Invalid state dict type: {type(state)}")

    _validate_state_dict(
        state,
        hidden_size=model.config.hidden_size,
        num_labels=len(class_names),
    )

    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise ValueError(
            f"State dict was not loaded cleanly. missing={missing}, unexpected={unexpected}"
        )

    model.to(runtime_device)
    model.eval()
    return model, tokenizer, runtime_device
