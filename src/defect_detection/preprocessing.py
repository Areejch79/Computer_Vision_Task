"""Deterministic image preprocessing shared by evaluation, ONNX export and the inference API.

This module deliberately depends only on NumPy + Pillow so that the production image does not
need PyTorch. Training-time *augmentation* lives in ``defect_detection.data.transforms``; it ends
with exactly the same resize + normalisation implemented here so that train/serve skew is avoided
(a unit test asserts the two paths agree).
"""

from __future__ import annotations

import io
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageOps

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class PreprocessConfig:
    image_size: int = 256
    mean: tuple[float, float, float] = IMAGENET_MEAN
    std: tuple[float, float, float] = IMAGENET_STD


def to_model_mode(img: Image.Image) -> Image.Image:
    """Normalise arbitrary input images (RGBA, palette, 16-bit, CMYK, EXIF-rotated...) to grayscale.

    The line camera produces grayscale images (the dataset JPEGs are RGB with identical channels),
    so colour carries no information. Converting everything to ``L`` and replicating to 3 channels
    makes the model robust to clients that send RGB, grayscale or RGBA versions of the same frame.
    """
    img = ImageOps.exif_transpose(img)
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        # Composite transparent images on white instead of silently using black.
        rgba = img.convert("RGBA")
        background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        img = Image.alpha_composite(background, rgba)
    if img.mode in ("I;16", "I;16B", "I;16L", "I"):
        arr = np.asarray(img, dtype=np.float32)
        lo, hi = float(arr.min()), float(arr.max())
        arr = (arr - lo) / (hi - lo + 1e-6) * 255.0
        return Image.fromarray(arr.astype(np.uint8), mode="L")
    return img.convert("L")


def resize(img: Image.Image, size: int) -> Image.Image:
    return img.resize((size, size), resample=Image.Resampling.BILINEAR)


def normalize(gray_resized: Image.Image, cfg: PreprocessConfig) -> np.ndarray:
    """Grayscale PIL image (already resized) -> float32 CHW array normalised with ImageNet stats."""
    arr = np.asarray(gray_resized, dtype=np.float32) / 255.0  # H, W
    chw = np.repeat(arr[None, :, :], 3, axis=0)
    mean = np.asarray(cfg.mean, dtype=np.float32)[:, None, None]
    std = np.asarray(cfg.std, dtype=np.float32)[:, None, None]
    return ((chw - mean) / std).astype(np.float32)


def preprocess_pil(img: Image.Image, cfg: PreprocessConfig) -> np.ndarray:
    """PIL image -> float32 array of shape (3, S, S)."""
    return normalize(resize(to_model_mode(img), cfg.image_size), cfg)


def load_image_bytes(data: bytes) -> Image.Image:
    """Decode bytes into a fully-loaded PIL image (raises on corrupt / truncated data)."""
    with Image.open(io.BytesIO(data)) as probe:
        probe.verify()  # cheap structural check, invalidates the handle
    img = Image.open(io.BytesIO(data))
    img.load()  # force full decode so truncated files fail here, not later
    return img


def softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)
