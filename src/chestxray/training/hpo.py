"""Hyperparameter optimisation and model selection.

Uses Optuna (if installed) to search over backbone, learning rate, weight decay,
dropout and loss; falls back to random search otherwise. Each trial trains a
model and is scored by the best macro validation AUC that run achieved — so
model selection and HPO are the same search, and the winner is chosen on
validation data only (never test).

Run:
    python -m chestxray.training.hpo --config configs/default.yaml --trials 20
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from ..config import Config, load_config
from ..utils import get_logger

log = get_logger("hpo")

# The model-selection candidates and hyperparameter ranges.
BACKBONES = ["densenet121", "resnet50", "mobilenetv3_small_100", "efficientnet_b0"]
LOSSES = ["bce", "weighted_bce", "focal"]


def _apply_trial(base: Config, params: dict) -> Config:
    cfg = copy.deepcopy(base)
    cfg.model.backbone = params["backbone"]
    cfg.model.dropout = params["dropout"]
    cfg.train.lr = params["lr"]
    cfg.train.weight_decay = params["weight_decay"]
    cfg.train.loss = params["loss"]
    cfg.experiment_name = f"hpo_{params['backbone']}_{params['loss']}_lr{params['lr']:.0e}"
    return cfg


def _objective_value(cfg: Config) -> float:
    """Train and return the best validation macro AUC of the run."""
    from .train import train

    out_dir = train(cfg)                       # writes history.json beside weights
    history = json.loads((out_dir.parent / "history.json").read_text())
    return max(h["val_auc_macro"] for h in history)


def run_optuna(base: Config, n_trials: int) -> dict:
    import optuna

    def objective(trial):
        params = {
            "backbone": trial.suggest_categorical("backbone", BACKBONES),
            "loss": trial.suggest_categorical("loss", LOSSES),
            "lr": trial.suggest_float("lr", 1e-5, 1e-3, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
            "dropout": trial.suggest_float("dropout", 0.0, 0.5),
        }
        return _objective_value(_apply_trial(base, params))

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    log.info("Best val AUC %.4f with %s", study.best_value, study.best_params)
    return {"best_value": study.best_value, "best_params": study.best_params}


def run_random(base: Config, n_trials: int, seed: int = 0) -> dict:
    """Dependency-free fallback when Optuna is not installed."""
    import numpy as np

    rng = np.random.default_rng(seed)
    best = {"best_value": -1.0, "best_params": None}
    for _ in range(n_trials):
        params = {
            "backbone": str(rng.choice(BACKBONES)),
            "loss": str(rng.choice(LOSSES)),
            "lr": float(10 ** rng.uniform(-5, -3)),
            "weight_decay": float(10 ** rng.uniform(-6, -3)),
            "dropout": float(rng.uniform(0.0, 0.5)),
        }
        value = _objective_value(_apply_trial(base, params))
        if value > best["best_value"]:
            best = {"best_value": value, "best_params": params}
    log.info("Best val AUC %.4f with %s", best["best_value"], best["best_params"])
    return best


def main() -> None:
    p = argparse.ArgumentParser(description="Hyperparameter optimisation / model selection.")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--trials", type=int, default=20)
    p.add_argument("--out", default="artifacts/hpo_result.json")
    a = p.parse_args()
    base = load_config(a.config)
    try:
        import optuna  # noqa: F401
        result = run_optuna(base, a.trials)
    except ImportError:
        log.info("Optuna not installed; using random search fallback.")
        result = run_random(base, a.trials)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(result, indent=2))
    log.info("Wrote %s", a.out)


if __name__ == "__main__":
    main()
