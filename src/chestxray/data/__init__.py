"""Data layer.

Torch-free helpers (label building, patient-level splits, class weights,
transform factory) are imported eagerly. ``ChestXrayDataset`` needs torch, so it
is exposed lazily — importing this package does not force the heavy stack.
"""
from .prepare import LABELS_DEFAULT, build_label_frame, make_splits, positive_weights, sample_weights
from .transforms import build_transforms
from .features import (
    engineer_metadata_features, age_band, apply_clahe, METADATA_FEATURES,
)

__all__ = [
    "LABELS_DEFAULT",
    "build_label_frame",
    "make_splits",
    "positive_weights",
    "sample_weights",
    "build_transforms",
    "ChestXrayDataset",
    "engineer_metadata_features",
    "age_band",
    "apply_clahe",
    "METADATA_FEATURES",
]


def __getattr__(name):  # PEP 562 lazy import
    if name == "ChestXrayDataset":
        from .dataset import ChestXrayDataset

        return ChestXrayDataset
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
