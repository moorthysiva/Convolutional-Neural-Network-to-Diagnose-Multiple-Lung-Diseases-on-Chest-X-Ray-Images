"""Turn the raw NIH ``Data_Entry_2017.csv`` into clean, multi-hot label frames
and leakage-free train / validation / test splits.

Two properties matter here:

1. **No patient leakage.** A plain ``train_test_split`` over *images* would place
   the same patient (who contributes many scans) in both train and validation,
   inflating every score. Splits are made on *Patient ID* — using the official
   NIH lists when available, otherwise a grouped split — so no patient crosses a
   split boundary.

2. **Honest multi-label targets.** Each image gets a 14-length {0,1} vector.
   "No Finding" becomes an all-zeros vector rather than a 15th competing class.
"""
from __future__ import annotations

from itertools import chain
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

# The 14 pathology labels in ChestX-ray14 (excluding "No Finding").
LABELS_DEFAULT: list[str] = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Effusion",
    "Emphysema", "Fibrosis", "Hernia", "Infiltration", "Mass",
    "Nodule", "Pleural_Thickening", "Pneumonia", "Pneumothorax",
]


def build_label_frame(
    entry_csv: str | Path,
    min_cases: int = 1000,
) -> tuple[pd.DataFrame, list[str]]:
    """Read the metadata CSV and attach one binary column per pathology.

    Returns the enriched dataframe and the list of retained labels (those with
    at least ``min_cases`` positive examples).
    """
    df = pd.read_csv(entry_csv)
    findings = df["Finding Labels"].str.replace("No Finding", "", regex=False)

    all_labels = sorted(
        {lab for lab in chain.from_iterable(findings.str.split("|")) if lab}
    )
    for label in all_labels:
        df[label] = findings.apply(lambda s: 1.0 if label in s.split("|") else 0.0)

    kept = [lab for lab in all_labels if df[lab].sum() >= min_cases]
    return df, kept


def _split_by_official_lists(
    df: pd.DataFrame,
    data_dir: Path,
    train_val_list: str,
    test_list: str,
    val_fraction: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame] | None:
    """Use NIH's official patient-disjoint lists if both files exist."""
    tv_path, te_path = data_dir / train_val_list, data_dir / test_list
    if not (tv_path.exists() and te_path.exists()):
        return None

    tv_names = set(pd.read_csv(tv_path, header=None)[0])
    te_names = set(pd.read_csv(te_path, header=None)[0])

    test_df = df[df["Image Index"].isin(te_names)].copy()
    trainval_df = df[df["Image Index"].isin(tv_names)].copy()

    # Carve a validation set out of train_val *by patient* so early-stopping
    # never sees a patient that also appears in training.
    train_df, val_df = _group_holdout(
        trainval_df, group_col="Patient ID", holdout_fraction=val_fraction, seed=seed
    )
    return train_df, val_df, test_df


def _group_holdout(
    df: pd.DataFrame, group_col: str, holdout_fraction: float, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split rows so that whole groups (patients) stay on one side."""
    rng = np.random.default_rng(seed)
    groups = df[group_col].unique()
    rng.shuffle(groups)
    n_holdout = int(round(len(groups) * holdout_fraction))
    holdout_groups = set(groups[:n_holdout])
    holdout_mask = df[group_col].isin(holdout_groups)
    return df[~holdout_mask].copy(), df[holdout_mask].copy()


def make_splits(
    df: pd.DataFrame,
    data_dir: str | Path,
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
    train_val_list: str = "train_val_list.txt",
    test_list: str = "test_list.txt",
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return leakage-free (train, val, test) frames.

    Prefers the official NIH lists; otherwise performs a three-way grouped
    split on Patient ID.
    """
    data_dir = Path(data_dir)
    official = _split_by_official_lists(
        df, data_dir, train_val_list, test_list, val_fraction, seed
    )
    if official is not None:
        return official

    # Fallback: three-way grouped split (test first, then val out of the rest).
    trainval_df, test_df = _group_holdout(df, "Patient ID", test_fraction, seed)
    # val_fraction is expressed w.r.t. the whole set, so rescale for the remainder.
    remaining_fraction = val_fraction / (1.0 - test_fraction)
    train_df, val_df = _group_holdout(
        trainval_df, "Patient ID", remaining_fraction, seed + 1
    )
    return train_df, val_df, test_df


def assert_no_patient_leakage(
    train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame
) -> None:
    """Guard-rail used in tests and at the top of training."""
    tr, va, te = (set(d["Patient ID"]) for d in (train_df, val_df, test_df))
    assert not (tr & va), "Patient leakage between train and val"
    assert not (tr & te), "Patient leakage between train and test"
    assert not (va & te), "Patient leakage between val and test"


def positive_weights(train_df: pd.DataFrame, labels: Sequence[str]) -> np.ndarray:
    """Per-class ``pos_weight`` for BCE = (#neg / #pos), clipped for stability.

    This is the principled way to handle class imbalance for multi-label BCE —
    it up-weights the loss on rare positive findings without discarding data the
    way undersampling does.
    """
    pos = train_df[list(labels)].sum(axis=0).to_numpy()
    neg = len(train_df) - pos
    with np.errstate(divide="ignore", invalid="ignore"):
        w = np.where(pos > 0, neg / np.maximum(pos, 1.0), 1.0)
    return np.clip(w, 1.0, 100.0).astype(np.float32)


def sample_weights(train_df: pd.DataFrame, labels: Sequence[str]) -> np.ndarray:
    """Per-sample weights for a ``WeightedRandomSampler`` (multi-label case).

    Resampling is the second lever for class imbalance, complementary to the
    weighted loss: instead of (or as well as) re-weighting the loss, we draw
    rare-class images more often during training.

    For a multi-label image there is no single class, so each image is weighted
    by the rarest positive label it carries (inverse class frequency). Images
    with no positive label ("No Finding") receive the median class weight, so
    they are neither over- nor under-sampled.

    Note: combining aggressive resampling with a strongly class-weighted loss
    can over-correct and hurt calibration. Typically use resampling with a plain
    BCE loss (``loss: bce``), or a mild version of both.
    """
    labels = list(labels)
    counts = train_df[labels].sum(axis=0).to_numpy()
    class_w = len(train_df) / np.maximum(counts, 1.0)   # inverse frequency
    Y = train_df[labels].to_numpy()
    per_sample = (Y * class_w).max(axis=1)              # rarest positive label
    per_sample = np.where(per_sample > 0, per_sample, np.median(class_w))
    return per_sample.astype(np.float32)


def attach_paths(df: pd.DataFrame, data_dir: str | Path) -> pd.DataFrame:
    """Map each ``Image Index`` to its on-disk path under ``images*/``."""
    from glob import glob
    import os

    data_dir = Path(data_dir)
    index = {
        os.path.basename(p): p
        for p in glob(str(data_dir / "images*" / "**" / "*.png"), recursive=True)
    }
    df = df.copy()
    df["path"] = df["Image Index"].map(index.get)
    missing = df["path"].isna().sum()
    if missing:
        df = df.dropna(subset=["path"]).reset_index(drop=True)
    return df
