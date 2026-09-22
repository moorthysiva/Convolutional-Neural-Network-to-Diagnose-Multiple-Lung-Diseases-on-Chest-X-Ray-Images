"""Exploratory data analysis for ChestX-ray14.

Profiles the dataset the way a sound study should before modelling: label
prevalence and co-occurrence, missing values, outliers (implausible ages),
demographic distributions, and the multiple-images-per-patient structure that
is the source of both sampling bias and the leakage risk. Torch-free.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def profile(df: pd.DataFrame, labels: list[str]) -> dict:
    """Return a structured EDA summary of the metadata frame."""
    age_raw = pd.to_numeric(
        df.get("Patient Age", pd.Series(dtype=float)).astype(str).str.extract(r"(\d+)")[0],
        errors="coerce",
    )
    imgs_per_patient = df.groupby("Patient ID").size()
    n_findings = df[labels].sum(axis=1) if set(labels).issubset(df.columns) else None

    summary = {
        "n_images": int(len(df)),
        "n_patients": int(df["Patient ID"].nunique()),
        "images_per_patient": {
            "mean": round(float(imgs_per_patient.mean()), 2),
            "max": int(imgs_per_patient.max()),
            "median": float(imgs_per_patient.median()),
        },
        "missing_values": _missing(df),
        "age_outliers_gt_100": int((age_raw > 100).sum()),
        "age_stats": {
            "min": float(age_raw.min()), "max": float(age_raw.max()),
            "median": float(age_raw.median()),
        },
        "sex_distribution": df.get("Patient Gender", pd.Series(dtype=object)).value_counts().to_dict(),
        "view_distribution": df.get("View Position", pd.Series(dtype=object)).value_counts().to_dict(),
    }
    if n_findings is not None:
        summary["label_prevalence"] = {l: int(df[l].sum()) for l in labels}
        summary["no_finding_fraction"] = round(float((n_findings == 0).mean()), 4)
        summary["multi_label_fraction"] = round(float((n_findings > 1).mean()), 4)
        summary["max_imbalance_ratio"] = round(
            float(df[labels].sum().max() / max(df[labels].sum().min(), 1)), 1)
    return summary


def _missing(df: pd.DataFrame) -> dict:
    """Report genuinely-missing values, including all-blank 'phantom' columns
    (the NIH CSV ships an unnamed empty column) and zeroed pixel spacing."""
    miss = {}
    for col in df.columns:
        n_na = int(df[col].isna().sum())
        if n_na:
            miss[col] = n_na
    # zeroed pixel spacing is missing-in-disguise
    for col in df.columns:
        if "PixelSpacing" in col:
            z = int((pd.to_numeric(df[col], errors="coerce") == 0).sum())
            if z:
                miss[f"{col}==0"] = z
    return miss


def co_occurrence(df: pd.DataFrame, labels: list[str]) -> pd.DataFrame:
    """Label-by-label co-occurrence counts (how often two findings appear on
    the same film) — motivates multi-label modelling."""
    Y = df[labels].to_numpy()
    return pd.DataFrame(Y.T @ Y, index=labels, columns=labels)
