"""A torch ``Dataset`` over a prepared label frame.

Reads a grayscale X-ray, converts to 3 channels (so ImageNet backbones apply),
runs the transform pipeline, and returns (image_tensor, label_vector).

This module imports torch at load time (a dataset needs it); the data-prep and
metrics modules deliberately do not, so they stay importable in lightweight
environments and CI unit tests.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .transforms import build_transforms


class ChestXrayDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        labels: Sequence[str],
        image_size: int = 224,
        train: bool = False,
        clahe: bool = False,
    ) -> None:
        super().__init__()
        self.frame = frame.reset_index(drop=True)
        self.labels = list(labels)
        self.paths = self.frame["path"].to_numpy()
        self.targets = self.frame[self.labels].to_numpy(dtype=np.float32)
        self.transform = build_transforms(image_size, train=train, clahe=clahe)

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, idx: int):
        import cv2

        img = cv2.imread(self.paths[idx], cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Could not read image: {self.paths[idx]}")
        img = np.stack([img] * 3, axis=-1)  # HxW -> HxWx3 for pretrained nets
        img = self.transform(image=img)["image"]
        target = torch.from_numpy(self.targets[idx])
        return img, target
