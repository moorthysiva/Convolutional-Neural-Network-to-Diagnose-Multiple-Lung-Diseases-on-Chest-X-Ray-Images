"""Feature engineering.

Two kinds of engineered features are supported:

1. **Image features** — CLAHE (Contrast-Limited Adaptive Histogram Equalisation)
   enhances local contrast in the lung fields, a standard radiographic
   pre-processing step for chest radiographs. Exposed here as a
   reusable function and, in ``transforms.py``, as an augmentation-pipeline flag.

2. **Patient-metadata features** — ChestX-ray14 ships structured metadata
   (age, sex, view position, follow-up number). These are engineered into a
   numeric feature matrix that can be fused with the CNN's image features or
   used to build a tabular baseline, and are useful for the bias/fairness audit
   in ``evaluation/fairness.py``.

The metadata functions are torch-free and unit-tested.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd


# --- image feature engineering --------------------------------------------
def apply_clahe(image: np.ndarray, clip_limit: float = 2.0,
                tile_grid: int = 8) -> np.ndarray:
    """Apply CLAHE to a single-channel uint8 image and return uint8."""
    import cv2

    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid, tile_grid))
    return clahe.apply(image.astype(np.uint8))


# --- metadata feature engineering -----------------------------------------
METADATA_FEATURES = [
    "age_norm", "sex_M", "sex_F", "view_PA", "view_AP", "followup_norm",
]


def engineer_metadata_features(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Turn raw NIH metadata columns into a clean numeric feature matrix.

    Handles the messy real-world details: ages recorded with a trailing unit
    ('058Y'), occasional out-of-range ages, and inconsistent view/sex strings.
    Returns (features, feature_names).
    """
    out = pd.DataFrame(index=df.index)

    # Age: strip non-digits, clip to a plausible range, normalise to [0, 1].
    age = (
        df.get("Patient Age", pd.Series(index=df.index, dtype=object))
        .astype(str).str.extract(r"(\d+)")[0].astype(float)
    )
    age = age.clip(lower=0, upper=100).fillna(age.median())
    out["age_norm"] = age / 100.0

    sex = df.get("Patient Gender", pd.Series(index=df.index, dtype=object)).astype(str).str.upper()
    out["sex_M"] = (sex.str.startswith("M")).astype(float)
    out["sex_F"] = (sex.str.startswith("F")).astype(float)

    view = df.get("View Position", pd.Series(index=df.index, dtype=object)).astype(str).str.upper()
    out["view_PA"] = (view == "PA").astype(float)
    out["view_AP"] = (view == "AP").astype(float)

    fu = pd.to_numeric(df.get("Follow-up #", pd.Series(index=df.index)), errors="coerce").fillna(0)
    # log1p then min-max, so a few very large follow-up numbers don't dominate.
    fu = np.log1p(fu.clip(lower=0))
    rng = (fu.max() - fu.min()) or 1.0
    out["followup_norm"] = (fu - fu.min()) / rng

    return out[METADATA_FEATURES].to_numpy(dtype=np.float32), list(METADATA_FEATURES)


def age_band(df: pd.DataFrame) -> pd.Series:
    """Coarse age bands for subgroup/fairness reporting."""
    age = (
        df.get("Patient Age", pd.Series(index=df.index, dtype=object))
        .astype(str).str.extract(r"(\d+)")[0].astype(float).clip(0, 100)
    )
    return pd.cut(age, bins=[0, 20, 40, 60, 80, 100],
                 labels=["0-20", "21-40", "41-60", "61-80", "81-100"])
