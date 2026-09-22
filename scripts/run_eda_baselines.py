"""Run EDA and the metadata-only baseline comparison on the NIH metadata CSV.

Reproduces the exploratory analysis and the sound-baseline table reported in the
paper — runnable on the metadata alone (no images/GPU required).

    python scripts/run_eda_baselines.py --csv data/Data_Entry_2017.csv --out artifacts/eda
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from chestxray.data import LABELS_DEFAULT
from chestxray.data.prepare import build_label_frame
from chestxray import eda, baselines
from chestxray.utils import get_logger

log = get_logger("eda")


def main() -> None:
    p = argparse.ArgumentParser(description="EDA + metadata baselines.")
    p.add_argument("--csv", required=True)
    p.add_argument("--out", default="artifacts/eda")
    p.add_argument("--folds", type=int, default=5)
    a = p.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    df, kept = build_label_frame(a.csv, min_cases=1000)
    labels = [l for l in LABELS_DEFAULT if l in kept] or kept

    profile = eda.profile(df, labels)
    (out / "eda_profile.json").write_text(json.dumps(profile, indent=2, default=str))
    log.info("EDA: %d images / %d patients | no-finding %.1f%% | multi-label %.1f%%",
             profile["n_images"], profile["n_patients"],
             100 * profile["no_finding_fraction"], 100 * profile["multi_label_fraction"])

    result = baselines.compare_baselines(df, labels, n_splits=a.folds)
    (out / "baseline_comparison.json").write_text(json.dumps(result, indent=2))
    for name, v in result.items():
        log.info("baseline %-20s macro AUC = %.4f ± %.4f",
                 name, v["macro_auc_mean"], v["macro_auc_std"])
    log.info("Wrote EDA + baseline reports to %s/", out)


if __name__ == "__main__":
    main()
