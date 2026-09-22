"""Losses for multi-label classification under heavy class imbalance.

* ``bce``          – plain ``BCEWithLogitsLoss`` (correct baseline).
* ``weighted_bce`` – BCE with per-class ``pos_weight`` = #neg/#pos, so rare
  findings are not drowned out by the common negative class.
* ``focal``        – focal loss, which down-weights easy negatives further.

All operate on raw logits, so no softmax/sigmoid is applied in the model.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, pos_weight: torch.Tensor | None = None) -> None:
        super().__init__()
        self.gamma = gamma
        self.register_buffer(
            "pos_weight", pos_weight if pos_weight is not None else None, persistent=False
        )

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, reduction="none", pos_weight=self.pos_weight
        )
        p_t = torch.exp(-bce)  # probability assigned to the true class
        loss = ((1 - p_t) ** self.gamma) * bce
        return loss.mean()


def build_loss(
    kind: str,
    pos_weight: np.ndarray | None = None,
    focal_gamma: float = 2.0,
    device: str = "cpu",
) -> nn.Module:
    pw = None
    if pos_weight is not None and kind in {"weighted_bce", "focal"}:
        pw = torch.as_tensor(pos_weight, dtype=torch.float32, device=device)

    if kind == "bce":
        return nn.BCEWithLogitsLoss()
    if kind == "weighted_bce":
        return nn.BCEWithLogitsLoss(pos_weight=pw)
    if kind == "focal":
        return FocalLoss(gamma=focal_gamma, pos_weight=pw)
    raise ValueError(f"Unknown loss '{kind}'. Choose bce | weighted_bce | focal.")
