"""Evaluate a trained checkpoint on the held-out test set.

Thresholds are tuned on the *validation* split and frozen, then applied to the
test split — so no operating-point information leaks from test into the reported
precision/recall/F1. Writes a JSON report and a ROC figure.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .config import load_config
from .data import ChestXrayDataset, make_splits, build_label_frame
from .data.prepare import attach_paths
from .models import build_model
from .training.metrics import (
    compute_metrics,
    expected_calibration_error,
    operating_point_metrics,
    tune_thresholds,
)
from .utils import get_logger, seed_everything

log = get_logger(__name__)


def _infer(model, frame, labels, image_size, device, batch_size=64):
    ds = ChestXrayDataset(frame, labels, image_size, train=False)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4)
    scores, targets = [], []
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            scores.append(torch.sigmoid(model(x.to(device))).cpu().numpy())
            targets.append(y.numpy())
    return np.concatenate(scores), np.concatenate(targets)


def evaluate(checkpoint: str, config: str, out: str | None = None) -> dict:
    cfg = load_config(config)
    seed_everything(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(checkpoint, map_location=device)
    labels = ckpt["labels"]

    model = build_model(cfg.model.backbone, len(labels), pretrained=False).to(device)
    model.load_state_dict(ckpt["model_state"])

    df, _ = build_label_frame(Path(cfg.data.data_dir) / cfg.data.entry_csv,
                              min_cases=cfg.data.min_cases)
    df = attach_paths(df, cfg.data.data_dir)
    _, val_df, test_df = make_splits(
        df, cfg.data.data_dir, cfg.data.val_fraction, cfg.data.test_fraction,
        cfg.data.official_split_train_val, cfg.data.official_split_test, cfg.seed)

    val_scores, val_targets = _infer(model, val_df, labels, cfg.data.image_size, device)
    thresholds = tune_thresholds(val_targets, val_scores, labels)

    test_scores, test_targets = _infer(model, test_df, labels, cfg.data.image_size, device)

    # Persist raw scores/targets so ROC curves and any later analysis can be
    # regenerated without re-running inference. The manifest is written in the
    # SAME row order as the predictions, so subgroup/fairness analysis can join
    # metadata to predictions safely.
    np.savez(Path(checkpoint).parent / "test_predictions.npz",
             scores=test_scores, targets=test_targets, labels=np.array(labels))
    test_df.reset_index(drop=True).to_csv(
        Path(checkpoint).parent / "test_manifest.csv", index=False)

    report = {
        "threshold_free": compute_metrics(test_targets, test_scores, labels),
        "operating_point": operating_point_metrics(test_targets, test_scores, labels, thresholds),
        "calibration_ece": expected_calibration_error(test_targets, test_scores),
        "n_test_images": int(len(test_df)),
    }

    out = out or str(Path(checkpoint).parent / "test_report.json")
    Path(out).write_text(json.dumps(report, indent=2))
    log.info("Test macro AUC %.4f | ECE %.4f | report -> %s",
             report["threshold_free"]["auc_macro"], report["calibration_ece"], out)
    return report


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate on the held-out test set.")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--out", default=None)
    a = p.parse_args()
    evaluate(a.checkpoint, a.config, a.out)


if __name__ == "__main__":
    main()
