"""Backward-compatible exports for inference helpers."""

from model.inference import load_classifier, predict_sentence, predict_sentence_with_meta

__all__ = ["load_classifier", "predict_sentence", "predict_sentence_with_meta"]
