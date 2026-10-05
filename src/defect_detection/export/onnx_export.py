"""Export a trained checkpoint to a deployable ONNX model directory.

Produces ``<out_dir>/model.onnx`` + ``<out_dir>/metadata.json`` and verifies numerical parity
between PyTorch and ONNX Runtime on real validation images. Optionally produces a statically
quantised INT8 model (``model.int8.onnx``) calibrated on training images.

Usage::

    python -m defect_detection.export.onnx_export --checkpoint runs/efficientnet_b0/best.pt \
        --out-dir models/efficientnet_b0 [--int8]
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import logging
from pathlib import Path

import numpy as np
import torch

from defect_detection import __version__
from defect_detection.data.dataset import load_split
from defect_detection.models import build_model
from defect_detection.preprocessing import PreprocessConfig, preprocess_pil, softmax

log = logging.getLogger("export")
OPSET = 18


def load_checkpoint_model(ckpt_path: Path) -> tuple[torch.nn.Module, dict]:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    mcfg = ckpt["config"]["model"]
    model = build_model(mcfg["name"], num_classes=len(ckpt["class_names"]), pretrained=False,
                        drop_rate=float(mcfg.get("drop_rate", 0.2)))
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt


def sample_inputs(splits_csv: Path, split: str, pp: PreprocessConfig, n: int, seed: int = 0) -> np.ndarray:
    from PIL import Image

    df = load_split(splits_csv, split)
    df = df.sample(n=min(n, len(df)), random_state=seed)
    return np.stack([preprocess_pil(Image.open(p), pp) for p in df.abs_path])


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def export(ckpt_path: Path, out_dir: Path, splits_csv: Path, int8: bool = False) -> Path:
    import onnx
    import onnxruntime as ort

    model, ckpt = load_checkpoint_model(ckpt_path)
    pp = PreprocessConfig(image_size=int(ckpt["image_size"]), mean=tuple(ckpt["mean"]), std=tuple(ckpt["std"]))
    out_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = out_dir / "model.onnx"

    dummy = torch.zeros(1, 3, pp.image_size, pp.image_size)
    torch.onnx.export(
        model, (dummy,), str(onnx_path),
        input_names=["input"], output_names=["logits"],
        dynamic_shapes={"x": {0: torch.export.Dim("batch", min=1, max=256)}},
        opset_version=OPSET, dynamo=True, external_data=False,
    )
    onnx.checker.check_model(str(onnx_path))

    # Parity check on real validation images (dynamic batch dimension is exercised too).
    x = sample_inputs(splits_csv, "val", pp, 16)
    with torch.no_grad():
        ref = torch.softmax(model(torch.from_numpy(x)), 1).numpy()
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    got = softmax(sess.run(None, {"input": x})[0])
    max_diff = float(np.abs(ref - got).max())
    log.info("PyTorch vs ONNX Runtime max |dp| = %.2e", max_diff)
    if max_diff > 1e-4:
        raise RuntimeError(f"ONNX parity check failed: max prob diff {max_diff}")

    created = dt.datetime.now(dt.timezone.utc)
    metadata = {
        "model_name": ckpt["config"]["model"]["name"],
        "model_version": f"{ckpt['config']['run_name']}-{created:%Y%m%d}-{sha256(onnx_path)[:8]}",
        "package_version": __version__,
        "created_at": created.isoformat(timespec="seconds"),
        "class_names": ckpt["class_names"],
        "positive_class": "defective",
        "image_size": pp.image_size,
        "mean": list(pp.mean),
        "std": list(pp.std),
        "input": {"name": "input", "shape": ["batch", 3, pp.image_size, pp.image_size], "dtype": "float32",
                  "color": "grayscale replicated to 3 channels"},
        "output": {"name": "logits", "shape": ["batch", len(ckpt["class_names"])]},
        "threshold": ckpt["threshold"],
        "threshold_policy": ckpt["threshold_policy"],
        "best_epoch": ckpt["best_epoch"],
        "validation_metrics": {k: ckpt["val_metrics"][k] for k in
                               ("precision", "recall", "f1", "roc_auc", "pr_auc", "confusion_matrix")},
        "train_class_counts": dict(zip(ckpt["class_names"], ckpt["train_class_counts"])),
        "onnx_opset": OPSET,
        "onnx_sha256": sha256(onnx_path),
        "parity_max_abs_prob_diff": max_diff,
    }

    if int8:
        metadata["int8"] = quantize_int8(onnx_path, out_dir / "model.int8.onnx", splits_csv, pp)
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    log.info("Exported %s (%.1f MB)", onnx_path, onnx_path.stat().st_size / 1e6)
    return onnx_path


def quantize_int8(fp32_path: Path, int8_path: Path, splits_csv: Path, pp: PreprocessConfig) -> dict:
    """Static QDQ INT8 quantisation, calibrated on 128 training images."""
    from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static
    from onnxruntime.quantization.shape_inference import quant_pre_process

    calib = sample_inputs(splits_csv, "train", pp, 128, seed=1)

    class Reader(CalibrationDataReader):
        def __init__(self):
            self.it = iter(calib[i: i + 1] for i in range(len(calib)))

        def get_next(self):
            x = next(self.it, None)
            return None if x is None else {"input": x}

    pre = int8_path.with_suffix(".pre.onnx")
    root_logger = logging.getLogger()
    prev_level = root_logger.level
    root_logger.setLevel(logging.WARNING)  # ORT quantiser logs one INFO line per tensor on the root logger
    try:
        quant_pre_process(str(fp32_path), str(pre))
        quantize_static(str(pre), str(int8_path), Reader(), quant_format=QuantFormat.QDQ,
                        activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8, per_channel=True)
    finally:
        root_logger.setLevel(prev_level)
    pre.unlink(missing_ok=True)
    log.info("INT8 model %s (%.1f MB)", int8_path, int8_path.stat().st_size / 1e6)
    return {"file": int8_path.name, "sha256": sha256(int8_path), "calibration_images": int(len(calib))}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--splits-csv", type=Path, default=Path("data/splits.csv"))
    ap.add_argument("--int8", action="store_true", help="also write a statically quantised INT8 model")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    export(args.checkpoint, args.out_dir, args.splits_csv, args.int8)


if __name__ == "__main__":
    main()
