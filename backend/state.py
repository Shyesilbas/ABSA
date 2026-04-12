"""Uygulama ömrü boyunca tek model örneği (lazy startup’ta doldurulur)."""
from __future__ import annotations

from typing import Any, Optional

_runtime: dict[str, Any] = {}
_load_error: Optional[str] = None


def startup_load() -> None:
    global _load_error
    _load_error = None
    _runtime.clear()
    try:
        from model.inference import load_classifier

        model, tokenizer, device = load_classifier()
        _runtime["model"] = model
        _runtime["tokenizer"] = tokenizer
        _runtime["device"] = device
    except Exception as exc:  # noqa: BLE001
        _load_error = str(exc)
        _runtime.clear()


def shutdown_clear() -> None:
    _runtime.clear()


def load_error() -> Optional[str]:
    return _load_error


def model_ready() -> bool:
    return _load_error is None and _runtime.get("model") is not None


def get_model_bundle() -> tuple[Any, Any, Any]:
    return _runtime["model"], _runtime["tokenizer"], _runtime["device"]
