"""Minimal Grad-CAM for timm CNNs (used only for offline error analysis)."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from PIL import Image


def default_target_layer(model: nn.Module) -> nn.Module:
    # Last stage of the feature extractor (still spatial for EfficientNet and MobileNetV3;
    # MobileNetV3's ``conv_head`` runs *after* global pooling so it is not usable for CAM).
    if hasattr(model, "blocks"):
        return model.blocks[-1]
    raise ValueError("cannot determine Grad-CAM target layer; pass one explicitly")


class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module | None = None):
        self.model = model.eval()
        layer = target_layer or default_target_layer(model)
        self._acts: torch.Tensor | None = None
        self._grads: torch.Tensor | None = None
        layer.register_forward_hook(self._save_act)

    def _save_act(self, _m, _inp, out):
        self._acts = out
        out.register_hook(self._save_grad)

    def _save_grad(self, grad):
        self._grads = grad

    def __call__(self, x: torch.Tensor, class_idx: int = 1) -> np.ndarray:
        """x: (1, 3, H, W). Returns a heat-map in [0, 1] of shape (H, W) for ``class_idx``."""
        self.model.zero_grad(set_to_none=True)
        x = x.clone().requires_grad_(True)
        logits = self.model(x)
        logits[0, class_idx].backward()
        w = self._grads.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((w * self._acts).sum(1))[0].detach().cpu().numpy()
        cam = cam / (cam.max() + 1e-8)
        h, w_ = x.shape[-2:]
        return np.asarray(Image.fromarray((cam * 255).astype(np.uint8)).resize((w_, h), Image.Resampling.BILINEAR),
                          dtype=np.float32) / 255.0


def overlay(gray: Image.Image, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    import matplotlib

    base = np.asarray(gray.convert("L").resize(cam.shape[::-1]), dtype=np.float32) / 255.0
    heat = matplotlib.colormaps["jet"](cam)[..., :3]
    return np.clip((1 - alpha) * base[..., None] + alpha * heat, 0, 1)
