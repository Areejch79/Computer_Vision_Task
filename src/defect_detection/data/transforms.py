"""Training-time augmentation.

Design choices (see README "Preprocessing & augmentation"):

* **Geometric** – the parts are rotationally symmetric impellers photographed top-down, so all
  8 dihedral transforms (90-degree rotations + flips) are label-preserving and introduce no
  padding artefacts. A mild random-resized-crop (>= 85 % area) simulates small framing / zoom
  changes without cropping away defects on the rim.
* **Photometric** – strong brightness/contrast jitter. The EDA showed that defective parts were
  photographed against a darker background (background brightness alone gives AUC ~0.79), so
  jitter is used to make that cue unreliable and push the network towards the actual defects.
* **Sensor** – light Gaussian blur (focus drift) and Gaussian noise (sensor noise).
* No colour augmentation: the camera is grayscale.

The pipeline ends with :func:`defect_detection.preprocessing.resize` + ``normalize``, i.e. the very
same code used at inference time, which removes a classic source of train/serve skew.
"""

from __future__ import annotations

import random

import numpy as np
import torch
from PIL import Image
from torchvision import transforms as T

from defect_detection.preprocessing import PreprocessConfig, normalize, preprocess_pil, resize


class RandomDihedral:
    """Uniformly sample one of the 8 symmetries of the square."""

    def __call__(self, img: Image.Image) -> Image.Image:
        k = random.randint(0, 3)
        if k:
            img = img.rotate(90 * k, expand=False)
        if random.random() < 0.5:
            img = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        return img


class TrainTransform:
    def __init__(self, cfg: PreprocessConfig, aug: dict | None = None):
        aug = aug or {}
        self.cfg = cfg
        self.noise_std = float(aug.get("noise_std", 0.03))
        self.noise_p = float(aug.get("noise_p", 0.2))
        self.pil_ops = T.Compose([
            RandomDihedral(),
            T.RandomResizedCrop(
                size=cfg.image_size,
                scale=(float(aug.get("crop_scale_min", 0.85)), 1.0),
                ratio=(0.95, 1.05),
                interpolation=T.InterpolationMode.BILINEAR,
            ),
            T.ColorJitter(brightness=float(aug.get("brightness", 0.35)), contrast=float(aug.get("contrast", 0.35))),
            T.RandomApply([T.GaussianBlur(kernel_size=5, sigma=(0.1, 1.2))], p=float(aug.get("blur_p", 0.2))),
        ])

    def __call__(self, img: Image.Image) -> torch.Tensor:
        img = self.pil_ops(img)
        img = resize(img, self.cfg.image_size)  # no-op size-wise, kept for exact parity with serving
        if random.random() < self.noise_p:
            arr = np.asarray(img, dtype=np.float32)
            arr = arr + np.random.normal(0.0, self.noise_std * 255.0, arr.shape)
            img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), mode="L")
        return torch.from_numpy(normalize(img, self.cfg))


class EvalTransform:
    def __init__(self, cfg: PreprocessConfig):
        self.cfg = cfg

    def __call__(self, img: Image.Image) -> torch.Tensor:
        return torch.from_numpy(preprocess_pil(img, self.cfg))
