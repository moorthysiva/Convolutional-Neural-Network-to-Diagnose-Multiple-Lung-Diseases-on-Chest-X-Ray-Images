"""Single-image inference wrapper used by the API and CLI.

Loads a checkpoint once, then turns a raw image (bytes or array) into a
per-pathology probability dict. Keeping this separate from the web layer means
the same tested code path serves the API, batch jobs and notebooks.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

from .data.transforms import build_transforms
from .models import build_model


class Predictor:
    def __init__(self, checkpoint: str, image_size: int = 224, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(checkpoint, map_location=self.device)
        self.labels: list[str] = ckpt["labels"]
        backbone = ckpt["config"]["model"]["backbone"]
        self.model = build_model(backbone, len(self.labels), pretrained=False).to(self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()
        self.transform = build_transforms(image_size, train=False)
        self.image_size = image_size

    def _to_tensor(self, image: np.ndarray) -> torch.Tensor:
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)
        elif image.shape[-1] == 1:
            image = np.repeat(image, 3, axis=-1)
        x = self.transform(image=image)["image"]
        return x.unsqueeze(0).to(self.device)

    @torch.no_grad()
    def predict(self, image: np.ndarray) -> dict[str, float]:
        probs = torch.sigmoid(self.model(self._to_tensor(image)))[0].cpu().numpy()
        return {label: float(p) for label, p in zip(self.labels, probs)}

    def predict_from_bytes(self, raw: bytes) -> dict[str, float]:
        import cv2

        arr = np.frombuffer(raw, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError("Could not decode image bytes.")
        return self.predict(img)


@lru_cache(maxsize=1)
def get_predictor(checkpoint: str, image_size: int = 224) -> Predictor:
    """Process-wide cached predictor so the API loads weights only once."""
    return Predictor(checkpoint, image_size)
