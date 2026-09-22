"""Tests for EDA, validation methods (grouped CV, bootstrap) and baselines."""
import numpy as np
import pandas as pd

from src.chestxray import eda, baselines
from src.chestxray.validation import grouped_kfold, bootstrap_auc_ci


def _toy(n_patients=120, per=4, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    labs = ["Effusion", "Cardiomegaly", "Nodule"]
    for pid in range(n_patients):
        age = int(rng.integers(20, 90))
        sex = rng.choice(["M", "F"])
        view = rng.choice(["PA", "AP"])
        for k in range(per):
            picks = rng.choice(labs, size=rng.integers(0, 3), replace=False)
            rows.append({
                "Image Index": f"{pid:03d}_{k}.png",
                "Finding Labels": "|".join(picks) if len(picks) else "No Finding",
                "Patient ID": pid, "Patient Age": f"{age:03d}Y",
                "Patient Gender": sex, "View Position": view, "Follow-up #": k,
            })
    df = pd.DataFrame(rows)
    for l in labs:
        df[l] = df["Finding Labels"].apply(lambda s: 1.0 if l in s.split("|") else 0.0)
    return df, labs


def test_eda_profile_core_fields():
    df, labs = _toy()
    prof = eda.profile(df, labs)
    assert prof["n_images"] == len(df)
    assert prof["n_patients"] == 120
    assert 0 <= prof["no_finding_fraction"] <= 1
    assert prof["images_per_patient"]["mean"] > 0


def test_grouped_kfold_is_patient_disjoint():
    df, _ = _toy()
    groups = df["Patient ID"]
    for tr, te in grouped_kfold(groups, n_splits=5, seed=1):
        assert not (set(groups.iloc[tr]) & set(groups.iloc[te]))


def test_bootstrap_ci_brackets_point():
    rng = np.random.default_rng(0)
    y = (rng.random(500) < 0.4).astype(int)
    s = np.clip(0.2 + 0.5 * y + rng.normal(0, 0.2, 500), 0, 1)
    point, lo, hi = bootstrap_auc_ci(y, s, n_boot=200)
    assert lo <= point <= hi
    assert 0.5 < point <= 1.0


def test_baseline_comparison_orders_sensibly():
    df, labs = _toy()
    res = baselines.compare_baselines(df, labs, n_splits=3, seed=0)
    assert set(res) == {"prevalence", "logistic_regression", "random_forest"}
    # prevalence is a chance-level floor
    assert abs(res["prevalence"]["macro_auc_mean"] - 0.5) < 0.05


def test_bootstrap_macro_auc_ci_brackets_point():
    from src.chestxray.validation import bootstrap_macro_auc_ci
    rng = np.random.default_rng(0)
    y = (rng.random((400, 3)) < 0.3).astype(int)
    s = np.clip(0.2 + 0.5 * y + rng.normal(0, 0.2, (400, 3)), 0, 1)
    point, lo, hi = bootstrap_macro_auc_ci(y, s, n_boot=200)
    assert lo <= point <= hi
    assert 0.5 < point <= 1.0


def test_compare_baselines_return_oof_shapes():
    df, labs = _toy()
    out = baselines.compare_baselines(df, labs, n_splits=3, seed=0, return_oof=True)
    assert set(out) == {"summary", "oof", "y_true"}
    for name, mat in out["oof"].items():
        assert mat.shape == out["y_true"].shape
