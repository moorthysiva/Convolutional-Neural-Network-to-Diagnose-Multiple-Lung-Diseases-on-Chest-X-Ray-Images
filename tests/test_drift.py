"""Tests for drift monitoring."""
import numpy as np

from monitoring.drift import population_stability_index, classify_psi, drift_report


def test_psi_near_zero_for_same_distribution():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, 5000)
    b = rng.normal(0, 1, 5000)
    assert population_stability_index(a, b) < 0.1


def test_psi_flags_large_shift():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, 5000)
    b = rng.normal(3, 1, 5000)  # big mean shift
    psi = population_stability_index(a, b)
    assert psi > 0.25
    assert classify_psi(psi) == "significant_shift"


def test_drift_report_alerts():
    rng = np.random.default_rng(1)
    ref = rng.normal(0, 1, 2000)
    live = rng.normal(4, 1, 2000)
    report = drift_report(ref, live)
    assert report["alert"] is True
