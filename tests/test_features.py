"""Tests for feature engineering (metadata; CLAHE image feature)."""
import numpy as np
import pandas as pd

from src.chestxray.data.features import (
    engineer_metadata_features,
    age_band,
    METADATA_FEATURES,
)


def _meta_df():
    return pd.DataFrame({
        "Patient Age": ["058Y", "002Y", "150", np.nan, "045"],
        "Patient Gender": ["M", "F", "m", "F", "X"],
        "View Position": ["PA", "AP", "PA", "AP", "PA"],
        "Follow-up #": [0, 3, 100, 1, np.nan],
    })


def test_metadata_features_shape_and_range():
    feats, names = engineer_metadata_features(_meta_df())
    assert names == METADATA_FEATURES
    assert feats.shape == (5, len(METADATA_FEATURES))
    # all engineered features normalised into [0, 1]
    assert feats.min() >= 0.0 and feats.max() <= 1.0


def test_age_parsing_strips_units_and_clips():
    feats, names = engineer_metadata_features(_meta_df())
    age = feats[:, names.index("age_norm")]
    assert abs(age[0] - 0.58) < 1e-6      # '058Y' -> 0.58
    assert age[2] <= 1.0                  # '150' clipped to 100 -> 1.0


def test_sex_one_hot_mutually_exclusive_when_known():
    feats, names = engineer_metadata_features(_meta_df())
    m = feats[:, names.index("sex_M")]
    f = feats[:, names.index("sex_F")]
    assert m[0] == 1.0 and f[0] == 0.0
    assert f[1] == 1.0 and m[1] == 0.0


def test_age_band_labels():
    bands = age_band(_meta_df())
    assert str(bands.iloc[0]) == "41-60"   # age 58
