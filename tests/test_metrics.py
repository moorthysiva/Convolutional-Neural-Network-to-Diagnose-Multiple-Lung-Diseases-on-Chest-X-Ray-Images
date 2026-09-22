"""Tests for the evaluation metrics."""
import numpy as np

from src.chestxray.training.metrics import (
    compute_metrics,
    expected_calibration_error,
    operating_point_metrics,
    sensitivity_specificity,
    tune_thresholds,
)

LABELS = ["a", "b"]


def _perfect():
    y_true = np.array([[1, 0], [0, 1], [1, 1], [0, 0]], float)
    y_score = np.array([[0.9, 0.1], [0.1, 0.9], [0.8, 0.7], [0.2, 0.3]], float)
    return y_true, y_score


def test_auc_is_one_for_perfect_ranking():
    y_true, y_score = _perfect()
    m = compute_metrics(y_true, y_score, LABELS)
    assert m["auc_macro"] == 1.0
    assert m["per_class"]["a"]["auc"] == 1.0


def test_sensitivity_specificity_bounds():
    y_true = np.array([1, 1, 0, 0])
    y_pred = np.array([1, 0, 0, 0])
    sens, spec = sensitivity_specificity(y_true, y_pred)
    assert sens == 0.5 and spec == 1.0


def test_threshold_tuning_and_operating_point():
    y_true, y_score = _perfect()
    thr = tune_thresholds(y_true, y_score, LABELS)
    op = operating_point_metrics(y_true, y_score, LABELS, thr)
    assert 0.0 <= op["a"]["f1"] <= 1.0
    assert set(op["a"]) >= {"precision", "recall", "f1", "sensitivity", "specificity"}


def test_ece_zero_for_calibrated_extremes():
    # Perfectly confident and correct -> zero calibration error.
    y_true = np.array([1, 1, 0, 0])
    y_score = np.array([1.0, 1.0, 0.0, 0.0])
    assert expected_calibration_error(y_true, y_score) == 0.0


def test_auc_nan_when_single_class_present():
    y_true = np.array([[1, 1]], float)     # only positives for both columns
    y_score = np.array([[0.6, 0.4]], float)
    m = compute_metrics(y_true, y_score, LABELS)
    assert np.isnan(m["per_class"]["a"]["auc"])
