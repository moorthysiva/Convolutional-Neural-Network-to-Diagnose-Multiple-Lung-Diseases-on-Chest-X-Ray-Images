"""Tests for label construction and — critically — leakage-free splits."""
import numpy as np
import pandas as pd
import pytest

from src.chestxray.data.prepare import (
    build_label_frame,
    make_splits,
    positive_weights,
    sample_weights,
    assert_no_patient_leakage,
)


def _toy_csv(tmp_path, n_patients=200, imgs_per_patient=5):
    rng = np.random.default_rng(0)
    rows = []
    diseases = ["Effusion", "Cardiomegaly", "Nodule"]
    for pid in range(n_patients):
        for k in range(imgs_per_patient):
            picks = rng.choice(diseases, size=rng.integers(0, 3), replace=False)
            label = "|".join(picks) if len(picks) else "No Finding"
            rows.append({"Image Index": f"{pid:04d}_{k}.png",
                         "Finding Labels": label, "Patient ID": pid})
    df = pd.DataFrame(rows)
    path = tmp_path / "Data_Entry_2017.csv"
    df.to_csv(path, index=False)
    return path


def test_build_label_frame_creates_binary_columns(tmp_path):
    csv = _toy_csv(tmp_path)
    df, labels = build_label_frame(csv, min_cases=1)
    assert set(labels).issubset({"Effusion", "Cardiomegaly", "Nodule"})
    for lab in labels:
        assert set(df[lab].unique()).issubset({0.0, 1.0})
    # "No Finding" must NOT become a label column.
    assert "No Finding" not in labels


def test_no_finding_is_all_zeros(tmp_path):
    csv = _toy_csv(tmp_path)
    df, labels = build_label_frame(csv, min_cases=1)
    nf = df[df["Finding Labels"] == "No Finding"]
    if len(nf):
        assert nf[labels].to_numpy().sum() == 0


def test_split_has_no_patient_leakage(tmp_path):
    csv = _toy_csv(tmp_path)
    df, _ = build_label_frame(csv, min_cases=1)
    train_df, val_df, test_df = make_splits(
        df, tmp_path, val_fraction=0.2, test_fraction=0.2, seed=1)
    # The guard raises if any patient crosses a boundary.
    assert_no_patient_leakage(train_df, val_df, test_df)
    total = len(train_df) + len(val_df) + len(test_df)
    assert total == len(df)


def test_positive_weights_upweight_rare_classes(tmp_path):
    csv = _toy_csv(tmp_path)
    df, labels = build_label_frame(csv, min_cases=1)
    train_df, _, _ = make_splits(df, tmp_path, 0.2, 0.2, seed=1)
    w = positive_weights(train_df, labels)
    assert w.shape == (len(labels),)
    assert (w >= 1.0).all()  # weights never below 1


def test_sample_weights_upweight_rare_class_images(tmp_path):
    csv = _toy_csv(tmp_path)
    df, labels = build_label_frame(csv, min_cases=1)
    train_df, _, _ = make_splits(df, tmp_path, 0.2, 0.2, seed=1)
    w = sample_weights(train_df, labels)
    assert w.shape == (len(train_df),)
    assert (w > 0).all()
    # An image carrying the rarest label should weigh at least as much as the
    # mean sample weight (it is drawn at least as often).
    counts = train_df[labels].sum(axis=0)
    rarest = counts.idxmin()
    rare_rows = train_df[rarest].to_numpy() == 1
    if rare_rows.any():
        assert w[rare_rows].mean() >= w.mean()
