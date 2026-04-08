"""Backward-compatible exports for training data utilities."""

from data.training_data import (
    build_train_val_frames,
    load_hf_subset,
    map_hf_label,
)

__all__ = ["build_train_val_frames", "map_hf_label", "load_hf_subset"]
