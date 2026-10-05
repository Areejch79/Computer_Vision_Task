"""Framework-light inference engine (ONNX Runtime + NumPy). Used by the API, evaluation and CLI.

A deployed model is a directory with two files::

    model.onnx      # graph: float32[N, 3, S, S] -> logits float32[N, 2]
    metadata.json   # class names, preprocessing, decision threshold, provenance

Keeping the threshold and preprocessing next to the weights means the API can never be
deployed with a threshold that was tuned for a different model.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from defect_detection.preprocessing import PreprocessConfig, preprocess_pil, softmax

log = logging.getLogger(__name__)


@dataclass
class Prediction:
    predicted_class: str
    confidence: float  # probability of the predicted class
    probabilities: dict[str, float]
    defect_probability: float
    threshold: float
    requires_review: bool  # confidence inside the uncertainty band -> route to a human

    def to_dict(self) -> dict:
        return asdict(self)


class ModelLoadError(RuntimeError):
    pass


class DefectClassifier:
    def __init__(self, model_dir: str | Path, threshold: float | None = None, review_band: float = 0.15,
                 intra_op_threads: int | None = None, onnx_filename: str = "model.onnx"):
        self.model_dir = Path(model_dir)
        onnx_path = self.model_dir / onnx_filename
        meta_path = self.model_dir / "metadata.json"
        if not onnx_path.is_file() or not meta_path.is_file():
            raise ModelLoadError(f"expected {onnx_path} and {meta_path}")
        try:
            self.metadata: dict = json.loads(meta_path.read_text())
            self.class_names: list[str] = list(self.metadata["class_names"])
            self.pp = PreprocessConfig(
                image_size=int(self.metadata["image_size"]),
                mean=tuple(self.metadata["mean"]),
                std=tuple(self.metadata["std"]),
            )
            self.threshold = float(threshold if threshold is not None else self.metadata["threshold"])
        except (KeyError, ValueError, TypeError) as exc:
            raise ModelLoadError(f"invalid metadata.json: {exc!r}") from exc
        if not 0.0 < self.threshold < 1.0:
            raise ModelLoadError(f"threshold must be in (0, 1), got {self.threshold}")
        self.review_band = float(review_band)
        self.positive_index = self.class_names.index("defective")

        import onnxruntime as ort

        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        so.intra_op_num_threads = intra_op_threads or min(4, os.cpu_count() or 1)
        so.inter_op_num_threads = 1
        try:
            self.session = ort.InferenceSession(str(onnx_path), sess_options=so, providers=["CPUExecutionProvider"])
        except Exception as exc:
            raise ModelLoadError(f"failed to load ONNX model {onnx_path}: {exc!r}") from exc
        self.input_name = self.session.get_inputs()[0].name
        self.model_path = onnx_path
        self._warmup()
        log.info("Loaded model %s (version %s, threshold %.4f)", onnx_path, self.version, self.threshold)

    @property
    def version(self) -> str:
        return str(self.metadata.get("model_version", "unknown"))

    def _warmup(self) -> None:
        dummy = np.zeros((1, 3, self.pp.image_size, self.pp.image_size), dtype=np.float32)
        self.session.run(None, {self.input_name: dummy})

    def preprocess(self, images: list[Image.Image]) -> np.ndarray:
        return np.stack([preprocess_pil(im, self.pp) for im in images])

    def predict_proba(self, batch: np.ndarray) -> np.ndarray:
        logits = self.session.run(None, {self.input_name: batch.astype(np.float32, copy=False)})[0]
        return softmax(logits.astype(np.float64))

    def _decide(self, probs: np.ndarray) -> Prediction:
        p_def = float(probs[self.positive_index])
        is_def = p_def >= self.threshold
        cls = "defective" if is_def else "normal"
        confidence = float(probs[self.class_names.index(cls)])
        # Uncertainty band around the operating threshold (in probability space).
        requires_review = abs(p_def - self.threshold) < self.review_band
        return Prediction(
            predicted_class=cls,
            confidence=round(confidence, 6),
            probabilities={c: round(float(p), 6) for c, p in zip(self.class_names, probs)},
            defect_probability=round(p_def, 6),
            threshold=round(self.threshold, 6),
            requires_review=bool(requires_review),
        )

    def predict(self, images: list[Image.Image]) -> tuple[list[Prediction], float]:
        """Classify a list of PIL images. Returns predictions and model latency (ms, pre+infer)."""
        t0 = time.perf_counter()
        probs = self.predict_proba(self.preprocess(images))
        latency_ms = (time.perf_counter() - t0) * 1000
        return [self._decide(p) for p in probs], latency_ms
