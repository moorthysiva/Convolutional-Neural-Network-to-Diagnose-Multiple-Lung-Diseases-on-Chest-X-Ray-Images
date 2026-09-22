"""Sound baselines and a method comparison.

Before trusting a deep model you need to know what trivial and simple methods
achieve on the same, correctly-validated protocol. We establish three baselines
that use only the cheap patient metadata (age, sex, view, follow-up):

* **Prevalence** — always predict the class base rate (ROC-AUC = 0.5 by
  construction; the floor any real model must beat).
* **Logistic regression** on engineered metadata features.
* **Random forest** on the same features.

All are evaluated with patient-grouped cross-validation (no leakage) and macro
ROC-AUC. If a metadata-only model already scores well above 0.5 on a pathology,
that signals confounding to watch for in the image model. Torch-free and fully
runnable on the metadata CSV alone.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from .data.features import engineer_metadata_features
from .validation import grouped_kfold


def _macro_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    aucs = []
    for i in range(y_true.shape[1]):
        if len(np.unique(y_true[:, i])) > 1:
            aucs.append(roc_auc_score(y_true[:, i], y_score[:, i]))
    return float(np.mean(aucs)) if aucs else float("nan")


def _fit_predict(model_ctor, X_tr, Y_tr, X_te):
    """Fit one binary model per label; return (n_te, n_labels) score matrix."""
    scores = np.zeros((X_te.shape[0], Y_tr.shape[1]))
    for i in range(Y_tr.shape[1]):
        if len(np.unique(Y_tr[:, i])) < 2:      # a fold with no positives
            scores[:, i] = Y_tr[:, i].mean()
            continue
        m = model_ctor()
        m.fit(X_tr, Y_tr[:, i])
        scores[:, i] = m.predict_proba(X_te)[:, 1]
    return scores


def compare_baselines(
    df: pd.DataFrame,
    labels: list[str],
    n_splits: int = 5,
    seed: int = 42,
    return_oof: bool = False,
) -> dict:
    """Run prevalence / logistic-regression / random-forest baselines with
    patient-grouped CV and return macro AUC (mean ± std across folds).

    With ``return_oof=True`` also returns out-of-fold score matrices per method
    (each image scored exactly once, when it fell in a test fold), which enable a
    bootstrap confidence interval on the macro AUC.
    """
    X, _ = engineer_metadata_features(df)
    Y = df[labels].to_numpy(dtype=float)
    groups = df["Patient ID"]

    methods = {
        "prevalence": None,
        "logistic_regression": lambda: LogisticRegression(max_iter=200, class_weight="balanced"),
        "random_forest": lambda: RandomForestClassifier(
            n_estimators=100, max_depth=8, class_weight="balanced", n_jobs=-1, random_state=seed),
    }

    results: dict[str, list[float]] = {m: [] for m in methods}
    oof = {m: np.zeros_like(Y) for m in methods}
    for tr, te in grouped_kfold(groups, n_splits=n_splits, seed=seed):
        for name, ctor in methods.items():
            if name == "prevalence":
                base = Y[tr].mean(axis=0, keepdims=True).repeat(len(te), axis=0)
                oof[name][te] = base
                results[name].append(_macro_auc(Y[te], base))
            else:
                s = _fit_predict(ctor, X[tr], Y[tr], X[te])
                oof[name][te] = s
                results[name].append(_macro_auc(Y[te], s))

    summary = {
        name: {"macro_auc_mean": round(float(np.nanmean(v)), 4),
               "macro_auc_std": round(float(np.nanstd(v)), 4),
               "folds": [round(x, 4) for x in v]}
        for name, v in results.items()
    }
    if return_oof:
        return {"summary": summary, "oof": oof, "y_true": Y}
    return summary
