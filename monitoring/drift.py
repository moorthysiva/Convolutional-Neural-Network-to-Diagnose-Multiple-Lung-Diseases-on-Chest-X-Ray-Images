"""Production monitoring: data / concept drift detection.

Once deployed, the input distribution shifts (new scanner, new hospital,
different demographics) and silently degrades a model — the "concept drift"
requirement. We monitor two cheap, effective signals:

* **PSI (Population Stability Index)** on a summary statistic of incoming images
  (e.g. mean pixel intensity) vs. the training distribution.
* **Prediction drift**: the distribution of output probabilities vs. a training
  baseline, which flags concept drift even when inputs look similar.

PSI rule of thumb:  < 0.1 stable · 0.1–0.25 moderate shift · > 0.25 significant.
"""
from __future__ import annotations

import numpy as np


def population_stability_index(
    expected: np.ndarray, actual: np.ndarray, n_bins: int = 10
) -> float:
    """PSI between a reference (expected) and a live (actual) 1-D sample."""
    expected, actual = np.asarray(expected, float), np.asarray(actual, float)
    quantiles = np.linspace(0, 100, n_bins + 1)
    edges = np.percentile(expected, quantiles)
    edges[0], edges[-1] = -np.inf, np.inf

    e_perc = np.histogram(expected, bins=edges)[0] / len(expected)
    a_perc = np.histogram(actual, bins=edges)[0] / len(actual)
    eps = 1e-6
    e_perc = np.clip(e_perc, eps, None)
    a_perc = np.clip(a_perc, eps, None)
    return float(np.sum((a_perc - e_perc) * np.log(a_perc / e_perc)))


def classify_psi(psi: float) -> str:
    if psi < 0.1:
        return "stable"
    if psi < 0.25:
        return "moderate_shift"
    return "significant_shift"


def drift_report(
    reference_stat: np.ndarray,
    live_stat: np.ndarray,
    reference_scores: np.ndarray | None = None,
    live_scores: np.ndarray | None = None,
) -> dict:
    """Combine input-feature drift and (optionally) prediction drift."""
    report = {}
    psi_in = population_stability_index(reference_stat, live_stat)
    report["input_psi"] = psi_in
    report["input_status"] = classify_psi(psi_in)

    if reference_scores is not None and live_scores is not None:
        psi_pred = population_stability_index(
            np.asarray(reference_scores).ravel(), np.asarray(live_scores).ravel()
        )
        report["prediction_psi"] = psi_pred
        report["prediction_status"] = classify_psi(psi_pred)

    report["alert"] = any(
        v == "significant_shift" for k, v in report.items() if k.endswith("status")
    )
    return report
