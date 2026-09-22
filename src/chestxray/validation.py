"""Experimental-design and validation methods.

* **Patient-grouped K-fold** — the correct cross-validation for this data, so no
  patient spans a fold boundary (naive K-fold would leak).
* **Bootstrap confidence intervals** for AUC — communicates uncertainty rather
  than a single point estimate.

Torch-free; used by the baseline comparison and available for any evaluation.
"""
from __future__ import annotations

from typing import Iterator

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def grouped_kfold(
    groups: pd.Series, n_splits: int = 5, seed: int = 42
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield (train_idx, test_idx) with whole groups kept on one side.

    A dependency-light GroupKFold that also shuffles the group assignment.
    """
    groups = pd.Series(np.asarray(groups)).reset_index(drop=True)
    unique = groups.unique()
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    folds = np.array_split(unique, n_splits)
    all_idx = np.arange(len(groups))
    for k in range(n_splits):
        test_groups = set(folds[k])
        test_mask = groups.isin(test_groups).to_numpy()
        yield all_idx[~test_mask], all_idx[test_mask]


def bootstrap_auc_ci(
    y_true: np.ndarray,
    y_score: np.ndarray,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Bootstrap (point, lower, upper) CI for a single-column ROC-AUC."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    rng = np.random.default_rng(seed)
    n = len(y_true)
    point = float(roc_auc_score(y_true, y_score)) if len(np.unique(y_true)) > 1 else float("nan")
    stats = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        stats.append(roc_auc_score(y_true[idx], y_score[idx]))
    if not stats:
        return point, float("nan"), float("nan")
    lo, hi = np.percentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return point, float(lo), float(hi)


def bootstrap_macro_auc_ci(
    y_true: np.ndarray,
    y_score: np.ndarray,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Bootstrap (point, lower, upper) CI for the macro-averaged multi-label AUC.

    Resamples images (rows) with replacement, recomputing the mean per-class AUC
    each time, so the interval reflects sampling uncertainty across the test set.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    n, k = y_true.shape
    rng = np.random.default_rng(seed)

    def macro(idx):
        aucs = []
        for i in range(k):
            yt = y_true[idx, i]
            if len(np.unique(yt)) > 1:
                aucs.append(roc_auc_score(yt, y_score[idx, i]))
        return float(np.mean(aucs)) if aucs else float("nan")

    point = macro(np.arange(n))
    stats = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        val = macro(idx)
        if not np.isnan(val):
            stats.append(val)
    if not stats:
        return point, float("nan"), float("nan")
    lo, hi = np.percentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return point, float(lo), float(hi)
