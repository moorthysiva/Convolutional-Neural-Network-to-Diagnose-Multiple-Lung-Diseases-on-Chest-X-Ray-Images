"""Bias / fairness assessment.

A model can post a strong overall AUC while performing much worse for a
subgroup (e.g. a sex, age band or view position). ChestX-ray14 is known to
carry such biases, so we report AUC per subgroup and flag disparities. This is
torch-free: it works on saved scores/targets plus the subgroup labels from the
metadata, so it can run on any evaluation output.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from ..training.metrics import _safe_auc


def subgroup_auc(
    y_true: np.ndarray,
    y_score: np.ndarray,
    labels: Sequence[str],
    groups: pd.Series,
) -> pd.DataFrame:
    """Macro AUC within each subgroup value.

    ``groups`` is a per-row categorical Series (e.g. Patient Gender). Returns a
    dataframe indexed by subgroup with columns [n, macro_auc].
    """
    groups = pd.Series(np.asarray(groups)).reset_index(drop=True)
    rows = {}
    for value in groups.dropna().unique():
        mask = (groups == value).to_numpy()
        if mask.sum() < 10:                     # too few to estimate reliably
            continue
        aucs = [
            _safe_auc(y_true[mask, i], y_score[mask, i])
            for i in range(len(labels))
        ]
        aucs = [a for a in aucs if not np.isnan(a)]
        rows[str(value)] = {"n": int(mask.sum()),
                            "macro_auc": float(np.mean(aucs)) if aucs else float("nan")}
    return pd.DataFrame.from_dict(rows, orient="index").sort_index()


def fairness_report(
    y_true: np.ndarray,
    y_score: np.ndarray,
    labels: Sequence[str],
    subgroups: dict[str, pd.Series],
    disparity_threshold: float = 0.05,
) -> dict:
    """Run ``subgroup_auc`` over several attributes and flag gaps.

    ``subgroups`` maps an attribute name (e.g. "sex", "view", "age_band") to its
    per-row values. A gap between the best and worst subgroup AUC exceeding
    ``disparity_threshold`` is flagged.
    """
    report = {}
    for attr, values in subgroups.items():
        table = subgroup_auc(y_true, y_score, labels, values)
        if table.empty:
            continue
        gap = float(table["macro_auc"].max() - table["macro_auc"].min())
        report[attr] = {
            "per_group": table["macro_auc"].round(4).to_dict(),
            "counts": table["n"].to_dict(),
            "auc_gap": round(gap, 4),
            "flagged": gap > disparity_threshold,
        }
    report["any_disparity"] = any(v.get("flagged") for v in report.values() if isinstance(v, dict))
    return report
