"""Singleton model instance across the application lifetime (filled during lazy startup)."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

logger = logging.getLogger(__name__)


@dataclass
class _ModelBundle:
    """Typed container for the loaded model resources."""
    model: Optional[PreTrainedModel] = None
    tokenizer: Optional[PreTrainedTokenizerBase] = None
    device: Optional[torch.device] = None


_bundle = _ModelBundle()
_load_error: Optional[str] = None


def startup_load() -> None:
    global _load_error
    _load_error = None
    _bundle.model = None
    _bundle.tokenizer = None
    _bundle.device = None
    try:
        from model.inference import load_classifier

        model, tokenizer, device = load_classifier()
        _bundle.model = model
        _bundle.tokenizer = tokenizer
        _bundle.device = device
        logger.info("Model loaded successfully on %s", device)
    except Exception as exc:  # noqa: BLE001
        _load_error = str(exc)
        _bundle.model = None
        _bundle.tokenizer = None
        _bundle.device = None
        logger.error("Model loading failed: %s", exc)


def shutdown_clear() -> None:
    _bundle.model = None
    _bundle.tokenizer = None
    _bundle.device = None


def load_error() -> Optional[str]:
    return _load_error


def model_ready() -> bool:
    return _load_error is None and _bundle.model is not None


def get_model_bundle() -> tuple[PreTrainedModel, PreTrainedTokenizerBase, torch.device]:
    """Return the loaded model, tokenizer, and device.

    Raises
    ------
    RuntimeError
        If the model has not been loaded yet.
    """
    if _bundle.model is None or _bundle.tokenizer is None or _bundle.device is None:
        raise RuntimeError("Model bundle is not loaded.")
    return _bundle.model, _bundle.tokenizer, _bundle.device
