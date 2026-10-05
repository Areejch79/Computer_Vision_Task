"""Model factory (timm backbones with ImageNet weights + 2-class head)."""

from __future__ import annotations

import logging
from pathlib import Path

import timm
import torch.nn as nn

log = logging.getLogger(__name__)


def build_model(name: str, num_classes: int = 2, pretrained: bool = True, weights_path: str | None = None,
                drop_rate: float = 0.2) -> nn.Module:
    """Create a timm model.

    ``weights_path`` lets us load ImageNet weights from a local file (downloaded by
    ``scripts/download_weights.sh``) so training works in air-gapped / proxied environments;
    otherwise timm fetches them from the Hugging Face hub.
    """
    kwargs: dict = {"num_classes": num_classes, "drop_rate": drop_rate}
    if pretrained and weights_path:
        if not Path(weights_path).is_file():
            raise FileNotFoundError(f"weights file not found: {weights_path} (run scripts/download_weights.sh)")
        kwargs["pretrained_cfg_overlay"] = {"file": str(weights_path)}
    model = timm.create_model(name, pretrained=pretrained, **kwargs)
    # timm's EfficientNet/MobileNet init scales the classifier by 1/sqrt(fan_out); with only 2 outputs
    # that gives huge initial logits (CE ~3 instead of ~0.69) and noisy early gradients.
    # A small-std init lets the head start from an unbiased, low-confidence state.
    head = model.get_classifier()
    if isinstance(head, nn.Linear):
        nn.init.normal_(head.weight, std=0.01)
        nn.init.zeros_(head.bias)
    n_params = sum(p.numel() for p in model.parameters())
    log.info("Built %s (pretrained=%s, %.2fM params)", name, pretrained, n_params / 1e6)
    return model


def set_backbone_trainable(model: nn.Module, trainable: bool) -> None:
    head_params = {id(p) for p in model.get_classifier().parameters()}
    for p in model.parameters():
        if id(p) not in head_params:
            p.requires_grad = trainable


def param_groups(model: nn.Module, lr: float, backbone_lr_mult: float, weight_decay: float) -> list[dict]:
    head_ids = {id(p) for p in model.get_classifier().parameters()}
    head = [p for p in model.parameters() if id(p) in head_ids]
    backbone = [p for p in model.parameters() if id(p) not in head_ids]
    return [
        {"params": backbone, "lr": lr * backbone_lr_mult, "weight_decay": weight_decay},
        {"params": head, "lr": lr, "weight_decay": weight_decay},
    ]
