"""Grad-CAM explainability.

Produces a heat-map over the input showing which regions drove a given
pathology's prediction. This addresses the "explainability" requirement and is
essential for clinical trust: a radiologist can check the model is attending to
the lung field, not to text markers or image borders (a classic shortcut-learning
failure on chest X-ray datasets).
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import torch


class GradCAM:
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module) -> None:
        self.model = model.eval()
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, _m, _i, output) -> None:
        self.activations = output.detach()

    def _save_gradient(self, _m, _gi, grad_output) -> None:
        self.gradients = grad_output[0].detach()

    def __call__(self, x: torch.Tensor, class_idx: int) -> np.ndarray:
        """Return a HxW heat-map in [0, 1] for one image (batch size 1)."""
        logits = self.model(x)
        self.model.zero_grad(set_to_none=True)
        logits[0, class_idx].backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)   # GAP over spatial
        cam = torch.relu((weights * self.activations).sum(dim=1)).squeeze(0)
        cam -= cam.min()
        cam /= (cam.max() + 1e-8)
        return cam.cpu().numpy()


def default_target_layer(model: torch.nn.Module) -> torch.nn.Module:
    """Best-effort pick of the last convolutional feature block for common
    timm backbones. Override explicitly for exotic architectures."""
    last_conv = None
    for module in model.modules():
        if isinstance(module, torch.nn.Conv2d):
            last_conv = module
    if last_conv is None:
        raise ValueError("No Conv2d layer found for Grad-CAM.")
    return last_conv


def overlay(heatmap: np.ndarray, image: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    """Blend a [0,1] heat-map onto a HxW or HxWx3 uint8 image."""
    import cv2

    hm = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    hm = cv2.applyColorMap((hm * 255).astype(np.uint8), cv2.COLORMAP_JET)
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return cv2.addWeighted(hm, alpha, image, 1 - alpha, 0)
