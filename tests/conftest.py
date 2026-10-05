"""Shared fixtures. Tests use a tiny hand-built ONNX model so they run in seconds without trained weights."""

from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

REPO = Path(__file__).resolve().parents[1]


def _tiny_onnx(path: Path, image_size: int) -> None:
    """logits = GAP(x) @ W + b with W chosen so that *dark* images are 'defective'."""
    onnx = pytest.importorskip("onnx")
    from onnx import TensorProto, helper, numpy_helper

    a = 4.0
    W = np.array([[a / 3, -a / 3]] * 3, dtype=np.float32)  # (3, 2)
    b = np.zeros(2, dtype=np.float32)
    graph = helper.make_graph(
        [
            helper.make_node("GlobalAveragePool", ["input"], ["gap"]),
            helper.make_node("Flatten", ["gap"], ["flat"], axis=1),
            helper.make_node("Gemm", ["flat", "W", "b"], ["logits"]),
        ],
        "tiny",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, ["batch", 3, image_size, image_size])],
        [helper.make_tensor_value_info("logits", TensorProto.FLOAT, ["batch", 2])],
        initializer=[numpy_helper.from_array(W, "W"), numpy_helper.from_array(b, "b")],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.save(model, str(path))


@pytest.fixture(scope="session")
def tiny_model_dir(tmp_path_factory) -> Path:
    d = tmp_path_factory.mktemp("tiny_model")
    _tiny_onnx(d / "model.onnx", 64)
    (d / "metadata.json").write_text(json.dumps({
        "model_name": "tiny", "model_version": "tiny-test", "class_names": ["normal", "defective"],
        "image_size": 64, "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225], "threshold": 0.5,
    }))
    return d


def encode(img: Image.Image, fmt: str = "PNG", **kw) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format=fmt, **kw)
    return buf.getvalue()


@pytest.fixture
def bright_png() -> bytes:
    return encode(Image.new("L", (128, 128), 230))


@pytest.fixture
def dark_jpeg() -> bytes:
    return encode(Image.new("RGB", (128, 128), (20, 20, 20)), "JPEG")
