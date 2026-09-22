"""Tests for the evaluation package (bias, robustness, overfitting)."""
import numpy as np
import pandas as pd

from src.chestxray.evaluation.fairness import subgroup_auc, fairness_report
from src.chestxray.evaluation.robustness import corrupt
from src.chestxray.evaluation.diagnostics import overfitting_report

LABELS = ["a", "b"]


def _data(n=400, seed=0):
    rng = np.random.default_rng(seed)
    y_true = (rng.random((n, 2)) < 0.3).astype(int)
    # scores correlated with truth -> AUC > 0.5
    y_score = np.clip(0.2 + 0.5 * y_true + rng.normal(0, 0.2, (n, 2)), 0, 1)
    groups = pd.Series(rng.choice(["M", "F"], size=n))
    return y_true, y_score, groups


def test_subgroup_auc_returns_row_per_group():
    y_true, y_score, groups = _data()
    table = subgroup_auc(y_true, y_score, LABELS, groups)
    assert set(table.index) == {"M", "F"}
    assert (table["macro_auc"] > 0.5).all()


def test_fairness_report_flags_and_reports_gap():
    y_true, y_score, groups = _data()
    rep = fairness_report(y_true, y_score, LABELS, {"sex": groups})
    assert "sex" in rep and "auc_gap" in rep["sex"]
    assert isinstance(rep["any_disparity"], bool)


def test_corrupt_numpy_ops_change_image_but_keep_shape():
    img = (np.ones((32, 32), dtype=np.uint8) * 128)
    for kind in ("gaussian_noise", "brightness"):
        out = corrupt(img, kind)
        assert out.shape == img.shape
        assert out.dtype == np.uint8
    # brightness up should not decrease the mean
    assert corrupt(img, "brightness").mean() >= img.mean()


def test_overfitting_report_detects_rising_val_loss():
    history = [{"epoch": e, "train_loss": 1.0 - 0.08 * e,
                "val_loss": 0.6 - 0.05 * e if e < 4 else 0.4 + 0.05 * (e - 4),
                "val_auc_macro": 0.7} for e in range(10)]
    rep = overfitting_report(history)
    assert rep["overfitting_suspected"] is True
    assert "generalisation_gap" in rep


def test_overfitting_report_clean_run():
    history = [{"epoch": e, "train_loss": 1.0 - 0.08 * e,
                "val_loss": 0.9 - 0.07 * e, "val_auc_macro": 0.7} for e in range(8)]
    rep = overfitting_report(history)
    assert rep["overfitting_suspected"] is False
