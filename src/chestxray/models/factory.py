"""Model factory.

Any timm backbone plus a linear head that outputs **one logit per pathology**.
We return raw logits (no activation baked in) and pair them with
``BCEWithLogitsLoss`` during training — the numerically stable, correct choice
for multi-label problems: unlike a softmax/categorical head, sigmoid + BCE lets
the diseases be predicted independently, which is correct when an image can show
several at once.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class MultiLabelClassifier(nn.Module):
    def __init__(self, backbone: str, num_classes: int, pretrained: bool = True,
                 dropout: float = 0.2) -> None:
        super().__init__()
        import timm

        # num_classes=0 -> timm returns pooled features; we add our own head so
        # the architecture is explicit and easy to swap.
        self.backbone = timm.create_model(
            backbone, pretrained=pretrained, num_classes=0, in_chans=3
        )
        feat_dim = self.backbone.num_features
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feat_dim, num_classes),
        )
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))  # logits, shape (B, num_classes)

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward(x))


def build_model(backbone: str, num_classes: int, pretrained: bool = True,
                dropout: float = 0.2) -> MultiLabelClassifier:
    return MultiLabelClassifier(backbone, num_classes, pretrained, dropout)
