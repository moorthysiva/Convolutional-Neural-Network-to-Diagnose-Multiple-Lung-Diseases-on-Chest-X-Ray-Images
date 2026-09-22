"""Typed, YAML-backed configuration.

A single source of truth for every run. Persisting the exact config alongside
the model weights is what makes an experiment reproducible: given the config +
the code commit, anyone can recreate the run.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DataConfig:
    data_dir: str = "data"                 # root holding images*/ folders + CSVs
    entry_csv: str = "Data_Entry_2017.csv"  # official NIH metadata
    image_size: int = 224                  # subtle findings need enough resolution
    min_cases: int = 1000                  # drop ultra-rare labels below this count
    val_fraction: float = 0.15
    test_fraction: float = 0.15
    # Use the official NIH patient-level lists when present; otherwise fall back
    # to a GROUP split on Patient ID (never a plain per-image random split).
    official_split_train_val: str = "train_val_list.txt"
    official_split_test: str = "test_list.txt"
    clahe: bool = False                    # engineered image feature (contrast)
    num_workers: int = 4


@dataclass
class ModelConfig:
    backbone: str = "densenet121"          # CheXNet-style default; any timm name works
    pretrained: bool = True
    dropout: float = 0.2


@dataclass
class TrainConfig:
    epochs: int = 30
    batch_size: int = 32
    lr: float = 1e-4
    weight_decay: float = 1e-5
    loss: str = "bce"                      # "bce" | "weighted_bce" | "focal"
    sampler: str = "none"                  # "none" | "weighted" (WeightedRandomSampler)
    focal_gamma: float = 2.0
    patience: int = 6                      # early-stopping patience on val loss
    lr_patience: int = 3                   # ReduceLROnPlateau patience
    amp: bool = True                       # mixed precision when CUDA is available
    monitor: str = "val_auc_macro"         # checkpoint selection metric (higher is better)


@dataclass
class Config:
    seed: int = 42
    output_dir: str = "artifacts"
    experiment_name: str = "densenet121_baseline"
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    # --- (de)serialisation -------------------------------------------------
    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        with open(path, "r") as fh:
            raw: dict[str, Any] = yaml.safe_load(fh) or {}
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Config":
        raw = dict(raw)
        data = DataConfig(**(raw.pop("data", {}) or {}))
        model = ModelConfig(**(raw.pop("model", {}) or {}))
        train = TrainConfig(**(raw.pop("train", {}) or {}))
        return cls(data=data, model=model, train=train, **raw)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as fh:
            yaml.safe_dump(self.to_dict(), fh, sort_keys=False)


def load_config(path: str | Path | None) -> Config:
    """Load a config from YAML, or return sensible defaults when path is None."""
    if path is None:
        return Config()
    return Config.from_yaml(path)
