"""Evaluation metrics for imbalanced multi-label classification.

Deliberately torch-free (numpy + scikit-learn) so it can be unit-tested cheaply
and reused by the API and monitoring code.

Why these metrics:

* **ROC-AUC** (macro + per class) – threshold-independent, the standard for
  ChestX-ray14 and comparable to published work.
* **Average precision / PR-AUC** – more informative than ROC-AUC when positives
  are very rare.
* **Sensitivity (recall) & specificity** – the clinically meaningful pair;
  accuracy is intentionally *not* headlined because on this imbalance it is
  dominated by the negative class and misleads.
* **Expected Calibration Error** – are the predicted probabilities trustworthy,
  not just well-ranked?
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def _safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """ROC-AUC that returns NaN when a class has a single label present."""
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def compute_metrics(
    y_true: np.ndarray, y_score: np.ndarray, labels: Sequence[str]
) -> dict:
    """Per-class and aggregate threshold-independent metrics."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    per_class = {}
    aucs, aps = [], []
    for i, name in enumerate(labels):
        auc = _safe_auc(y_true[:, i], y_score[:, i])
        ap = float(average_precision_score(y_true[:, i], y_score[:, i])) \
            if y_true[:, i].sum() > 0 else float("nan")
        per_class[name] = {"auc": auc, "ap": ap, "n_pos": int(y_true[:, i].sum())}
        if not np.isnan(auc):
            aucs.append(auc)
        if not np.isnan(ap):
            aps.append(ap)
    return {
        "auc_macro": float(np.mean(aucs)) if aucs else float("nan"),
        "ap_macro": float(np.mean(aps)) if aps else float("nan"),
        "per_class": per_class,
    }


def sensitivity_specificity(
    y_true: np.ndarray, y_pred: np.ndarray
) -> tuple[float, float]:
    """Return (sensitivity, specificity) for one binary column."""
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    sens = tp / (tp + fn) if (tp + fn) else float("nan")
    spec = tn / (tn + fp) if (tn + fp) else float("nan")
    return sens, spec


def tune_thresholds(
    y_true: np.ndarray, y_score: np.ndarray, labels: Sequence[str]
) -> dict[str, float]:
    """Pick, per class, the probability threshold that maximises F1 on the
    provided (validation) set. Thresholds are then frozen and applied to test —
    never tuned on the test set itself."""
    thresholds = {}
    grid = np.linspace(0.05, 0.95, 19)
    for i, name in enumerate(labels):
        if y_true[:, i].sum() == 0:
            thresholds[name] = 0.5
            continue
        best_t, best_f1 = 0.5, -1.0
        for t in grid:
            f1 = f1_score(y_true[:, i], (y_score[:, i] >= t).astype(int),
                          zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, float(t)
        thresholds[name] = best_t
    return thresholds


def operating_point_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    labels: Sequence[str],
    thresholds: dict[str, float],
) -> dict:
    """Precision / recall / F1 / sensitivity / specificity at fixed thresholds."""
    out = {}
    for i, name in enumerate(labels):
        t = thresholds.get(name, 0.5)
        y_pred = (y_score[:, i] >= t).astype(int)
        sens, spec = sensitivity_specificity(y_true[:, i], y_pred)
        out[name] = {
            "threshold": t,
            "precision": float(precision_score(y_true[:, i], y_pred, zero_division=0)),
            "recall": float(recall_score(y_true[:, i], y_pred, zero_division=0)),
            "f1": float(f1_score(y_true[:, i], y_pred, zero_division=0)),
            "sensitivity": sens,
            "specificity": spec,
        }
    return out


def expected_calibration_error(
    y_true: np.ndarray, y_score: np.ndarray, n_bins: int = 10
) -> float:
    """ECE over a flattened multi-label problem: mean |confidence - accuracy|
    across equal-width probability bins, weighted by bin population."""
    y_true = np.asarray(y_true).ravel()
    y_score = np.asarray(y_score).ravel()
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece, n = 0.0, len(y_score)
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (y_score >= lo) & (y_score < hi if hi < 1.0 else y_score <= hi)
        if not mask.any():
            continue
        conf = y_score[mask].mean()
        acc = y_true[mask].mean()
        ece += (mask.sum() / n) * abs(conf - acc)
    return float(ece)
