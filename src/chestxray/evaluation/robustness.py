"""Robustness assessment.

A model that only works on pristine images is fragile in deployment, where
scans vary in noise, contrast and positioning. We apply a set of controlled
corruptions and measure how much macro AUC degrades — a proxy for robustness
and an early warning for the kinds of shift the drift monitor watches for in
production.

The corruption functions are torch-free and unit-tested; ``robustness_report``
takes a callable that maps a batch of images to scores, so it is model-agnostic.
"""
from __future__ import annotations

from typing import Callable, Sequence

import numpy as np

from ..training.metrics import compute_metrics


def _gaussian_noise(img: np.ndarray, sigma: float = 12.0) -> np.ndarray:
    noisy = img.astype(np.float32) + np.random.normal(0, sigma, img.shape)
    return np.clip(noisy, 0, 255).astype(np.uint8)


def _blur(img: np.ndarray, k: int = 3) -> np.ndarray:
    import cv2
    return cv2.GaussianBlur(img, (k, k), 0)


def _brightness(img: np.ndarray, factor: float = 1.3) -> np.ndarray:
    return np.clip(img.astype(np.float32) * factor, 0, 255).astype(np.uint8)


def _rotate(img: np.ndarray, degrees: float = 8.0) -> np.ndarray:
    import cv2
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), degrees, 1.0)
    return cv2.warpAffine(img, m, (w, h), borderMode=cv2.BORDER_REFLECT)


CORRUPTIONS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "gaussian_noise": _gaussian_noise,
    "blur": _blur,
    "brightness": _brightness,
    "rotate": _rotate,
}


def corrupt(image: np.ndarray, kind: str) -> np.ndarray:
    """Apply a named corruption to a uint8 image."""
    if kind not in CORRUPTIONS:
        raise ValueError(f"Unknown corruption '{kind}'. Options: {list(CORRUPTIONS)}")
    return CORRUPTIONS[kind](image)


def robustness_report(
    images: Sequence[np.ndarray],
    y_true: np.ndarray,
    labels: Sequence[str],
    predict_fn: Callable[[Sequence[np.ndarray]], np.ndarray],
) -> dict:
    """Compare clean macro AUC with macro AUC under each corruption.

    ``predict_fn`` maps a list of uint8 images to a (N, C) score array.
    Returns clean AUC, per-corruption AUC and the absolute drop.
    """
    clean = compute_metrics(y_true, predict_fn(list(images)), labels)["auc_macro"]
    report = {"clean_auc": clean, "corruptions": {}}
    for kind in CORRUPTIONS:
        corrupted = [corrupt(im, kind) for im in images]
        auc = compute_metrics(y_true, predict_fn(corrupted), labels)["auc_macro"]
        report["corruptions"][kind] = {"auc": auc, "drop": round(clean - auc, 4)}
    worst = max(report["corruptions"].values(), key=lambda d: d["drop"])["drop"]
    report["worst_drop"] = worst
    return report
