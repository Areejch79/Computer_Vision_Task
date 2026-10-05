"""Dataset backed by ``data/splits.csv``."""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, WeightedRandomSampler

from defect_detection.preprocessing import to_model_mode

log = logging.getLogger(__name__)


def load_split(splits_csv: Path, split: str, root: Path = Path(".")) -> pd.DataFrame:
    df = pd.read_csv(splits_csv)
    df = df[df.split == split].reset_index(drop=True)
    if df.empty:
        raise ValueError(f"split '{split}' is empty in {splits_csv}")
    df["abs_path"] = [str(root / p) for p in df.path]
    return df


def simulate_prevalence(df: pd.DataFrame, defect_fraction: float, seed: int) -> pd.DataFrame:
    """Down-sample defective images so they make up ``defect_fraction`` of the set.

    Used for the imbalance ablation: on a real line defects are rare (often <5 %), whereas this
    dataset is 60 % defective. Normals are kept, defects are sub-sampled.
    """
    normals = df[df.label == 0]
    n_def = int(round(defect_fraction * len(normals) / (1 - defect_fraction)))
    defects = df[df.label == 1].sample(n=min(n_def, int((df.label == 1).sum())), random_state=seed)
    out = pd.concat([normals, defects]).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    log.info("Simulated prevalence %.2f: %d normal / %d defective", defect_fraction, len(normals), len(defects))
    return out


class DefectDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform: Callable[[Image.Image], torch.Tensor], cache: bool = True):
        self.paths = df.abs_path.tolist()
        self.labels = df.label.astype(int).to_numpy()
        self.transform = transform
        self._cache: dict[int, Image.Image] | None = {} if cache else None

    def __len__(self) -> int:
        return len(self.paths)

    def _load(self, i: int) -> Image.Image:
        if self._cache is not None and i in self._cache:
            return self._cache[i]
        with Image.open(self.paths[i]) as im:
            img = to_model_mode(im)
            img.load()
        if self._cache is not None:
            self._cache[i] = img
        return img

    def __getitem__(self, i: int) -> tuple[torch.Tensor, int]:
        return self.transform(self._load(i)), int(self.labels[i])

    def class_counts(self) -> np.ndarray:
        return np.bincount(self.labels, minlength=2)


def inverse_frequency_weights(counts: np.ndarray) -> torch.Tensor:
    """w_c = N / (K * n_c): 1.0 for perfectly balanced data, >1 for minority classes."""
    counts = np.maximum(counts, 1)
    return torch.tensor(counts.sum() / (len(counts) * counts), dtype=torch.float32)


def balanced_sampler(labels: np.ndarray, seed: int) -> WeightedRandomSampler:
    counts = np.bincount(labels, minlength=2)
    sample_w = (1.0 / counts)[labels]
    g = torch.Generator().manual_seed(seed)
    return WeightedRandomSampler(torch.as_tensor(sample_w, dtype=torch.double), len(labels), replacement=True,
                                 generator=g)
