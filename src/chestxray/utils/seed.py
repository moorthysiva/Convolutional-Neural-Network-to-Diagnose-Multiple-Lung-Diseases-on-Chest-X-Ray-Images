"""Reproducibility helpers.

Seeds every RNG the pipeline touches (Python, NumPy, PyTorch, cuDNN) and exposes
a deterministic-mode switch, so runs are reproducible from their config.
"""
from __future__ import annotations

import os
import random

import numpy as np


def seed_everything(seed: int = 42, deterministic: bool = True) -> None:
    """Seed Python, NumPy and (if installed) PyTorch RNGs."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            # Trades a little speed for run-to-run reproducibility.
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:  # torch is optional for the data/metrics layers
        pass
