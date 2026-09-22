"""Reproducible training pipeline.

Design choices, all in one place:
  * the whole training set is seen each epoch;
  * BCE-with-logits loss on a sigmoid-free model (correct for multi-label);
  * checkpoint on macro validation AUC, early-stop on validation loss;
  * mixed precision, LR scheduling, deterministic seeding;
  * the exact config and per-epoch history are written next to the weights.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..config import Config
from ..data import (
    ChestXrayDataset,
    LABELS_DEFAULT,
    build_label_frame,
    make_splits,
    positive_weights,
)
from ..data.prepare import assert_no_patient_leakage, attach_paths
from ..models import build_model
from ..utils import get_logger, seed_everything
from .losses import build_loss
from .metrics import compute_metrics

log = get_logger(__name__)


def _evaluate(model, loader, device) -> tuple[np.ndarray, np.ndarray, float, torch.nn.Module]:
    model.eval()
    loss_fn = torch.nn.BCEWithLogitsLoss()
    scores, targets, losses = [], [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            losses.append(loss_fn(logits, y).item())
            scores.append(torch.sigmoid(logits).cpu().numpy())
            targets.append(y.cpu().numpy())
    return (
        np.concatenate(scores),
        np.concatenate(targets),
        float(np.mean(losses)),
        loss_fn,
    )


def train(cfg: Config) -> Path:
    seed_everything(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(cfg.output_dir) / cfg.experiment_name
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg.save(out_dir / "config.yaml")
    log.info("Device: %s | output: %s", device, out_dir)

    # --- data ---------------------------------------------------------------
    entry_csv = Path(cfg.data.data_dir) / cfg.data.entry_csv
    df, labels = build_label_frame(entry_csv, min_cases=cfg.data.min_cases)
    labels = [lab for lab in LABELS_DEFAULT if lab in labels] or labels
    df = attach_paths(df, cfg.data.data_dir)

    train_df, val_df, test_df = make_splits(
        df, cfg.data.data_dir,
        val_fraction=cfg.data.val_fraction,
        test_fraction=cfg.data.test_fraction,
        train_val_list=cfg.data.official_split_train_val,
        test_list=cfg.data.official_split_test,
        seed=cfg.seed,
    )
    assert_no_patient_leakage(train_df, val_df, test_df)
    log.info("Split (images): train=%d val=%d test=%d | labels=%d",
             len(train_df), len(val_df), len(test_df), len(labels))
    test_df.to_csv(out_dir / "test_manifest.csv", index=False)

    train_ds = ChestXrayDataset(train_df, labels, cfg.data.image_size, train=True, clahe=cfg.data.clahe)
    val_ds = ChestXrayDataset(val_df, labels, cfg.data.image_size, train=False, clahe=cfg.data.clahe)

    # Optional imbalance resampling: draw rare-class images more often.
    train_sampler = None
    if cfg.train.sampler == "weighted":
        from torch.utils.data import WeightedRandomSampler
        from ..data import sample_weights

        sw = sample_weights(train_df, labels)
        train_sampler = WeightedRandomSampler(
            weights=torch.as_tensor(sw, dtype=torch.double),
            num_samples=len(sw), replacement=True,
        )
        log.info("Using WeightedRandomSampler for imbalance resampling.")

    train_loader = DataLoader(train_ds, batch_size=cfg.train.batch_size,
                              shuffle=(train_sampler is None), sampler=train_sampler,
                              num_workers=cfg.data.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.train.batch_size, shuffle=False,
                            num_workers=cfg.data.num_workers, pin_memory=True)

    # --- model / loss / optim ----------------------------------------------
    model = build_model(cfg.model.backbone, len(labels),
                        cfg.model.pretrained, cfg.model.dropout).to(device)
    pos_w = positive_weights(train_df, labels)
    loss_fn = build_loss(cfg.train.loss, pos_weight=pos_w,
                         focal_gamma=cfg.train.focal_gamma, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr,
                                  weight_decay=cfg.train.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=cfg.train.lr_patience)
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.train.amp and device == "cuda")

    # --- loop ---------------------------------------------------------------
    best_metric, best_epoch, history = -np.inf, -1, []
    labels_path = out_dir / "labels.json"
    labels_path.write_text(json.dumps(labels))

    for epoch in range(cfg.train.epochs):
        model.train()
        t0, running = time.time(), 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=cfg.train.amp and device == "cuda"):
                loss = loss_fn(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running += loss.item()

        scores, targets, val_loss, _ = _evaluate(model, val_loader, device)
        m = compute_metrics(targets, scores, labels)
        scheduler.step(val_loss)
        rec = {"epoch": epoch, "train_loss": running / max(len(train_loader), 1),
               "val_loss": val_loss, "val_auc_macro": m["auc_macro"],
               "val_ap_macro": m["ap_macro"], "seconds": round(time.time() - t0, 1)}
        history.append(rec)
        log.info("epoch %02d | train_loss %.4f | val_loss %.4f | val_AUC %.4f",
                 epoch, rec["train_loss"], val_loss, m["auc_macro"])

        monitored = m["auc_macro"]
        if monitored > best_metric:
            best_metric, best_epoch = monitored, epoch
            torch.save({"model_state": model.state_dict(), "labels": labels,
                        "config": cfg.to_dict()}, out_dir / "best_model.pt")

        if epoch - best_epoch >= cfg.train.patience:
            log.info("Early stopping (no val AUC improvement for %d epochs).",
                     cfg.train.patience)
            break

    (out_dir / "history.json").write_text(json.dumps(history, indent=2))
    log.info("Best val AUC %.4f at epoch %d. Weights: %s",
             best_metric, best_epoch, out_dir / "best_model.pt")
    return out_dir / "best_model.pt"


def main() -> None:
    import argparse
    from ..config import load_config

    parser = argparse.ArgumentParser(description="Train the chest X-ray model.")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    train(load_config(args.config))


if __name__ == "__main__":
    main()
