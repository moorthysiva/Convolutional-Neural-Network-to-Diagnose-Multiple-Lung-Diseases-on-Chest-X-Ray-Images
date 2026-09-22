"""Training layer.

Metrics are torch-free and imported eagerly. Losses need torch and are exposed
lazily so the metrics/evaluation code stays importable without the heavy stack.
"""
from .metrics import (
    compute_metrics,
    expected_calibration_error,
    operating_point_metrics,
    tune_thresholds,
)

__all__ = [
    "compute_metrics",
    "expected_calibration_error",
    "operating_point_metrics",
    "tune_thresholds",
    "build_loss",
    "FocalLoss",
]


def __getattr__(name):  # PEP 562 lazy import
    if name in {"build_loss", "FocalLoss"}:
        from . import losses

        return getattr(losses, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
