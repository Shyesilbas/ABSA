"""Backward-compatible exports for trainer functions."""

from model.trainer import (
    build_loss_fn,
    compute_class_weights_from_labels,
    evaluate_epoch,
    fit,
    train_epoch,
)

__all__ = [
    "compute_class_weights_from_labels",
    "build_loss_fn",
    "train_epoch",
    "evaluate_epoch",
    "fit",
]
