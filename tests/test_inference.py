import json

import pytest
from PIL import Image

pytest.importorskip("onnxruntime")

from defect_detection.inference import DefectClassifier, ModelLoadError  # noqa: E402


def test_predicts_dark_as_defective_and_bright_as_normal(tiny_model_dir):
    clf = DefectClassifier(tiny_model_dir)
    preds, ms = clf.predict([Image.new("L", (100, 100), 10), Image.new("L", (100, 100), 245)])
    assert [p.predicted_class for p in preds] == ["defective", "normal"]
    for p in preds:
        assert 0.5 <= p.confidence <= 1
        assert abs(sum(p.probabilities.values()) - 1) < 1e-5
    assert ms > 0


def test_threshold_override_and_review_band(tiny_model_dir):
    gray = Image.new("L", (100, 100), 120)  # P(def) ~ 0.33 with the tiny model
    clf = DefectClassifier(tiny_model_dir, threshold=0.3, review_band=0.1)
    (p,), _ = clf.predict([gray])
    assert p.predicted_class == "defective" and p.requires_review
    assert p.confidence == pytest.approx(p.defect_probability)
    clf = DefectClassifier(tiny_model_dir, threshold=0.6, review_band=0.1)
    (p,), _ = clf.predict([gray])
    assert p.predicted_class == "normal" and not p.requires_review
    assert p.confidence == pytest.approx(p.probabilities["normal"])


def test_missing_model_raises(tmp_path):
    with pytest.raises(ModelLoadError):
        DefectClassifier(tmp_path)


def test_invalid_metadata_raises(tiny_model_dir, tmp_path):
    (tmp_path / "model.onnx").write_bytes((tiny_model_dir / "model.onnx").read_bytes())
    (tmp_path / "metadata.json").write_text(json.dumps({"class_names": ["normal", "defective"]}))
    with pytest.raises(ModelLoadError):
        DefectClassifier(tmp_path)


def test_quality_gate_flags_blurry_and_badly_exposed_images(tiny_model_dir, tmp_path):
    import numpy as np

    meta = json.loads((tiny_model_dir / "metadata.json").read_text())
    meta["quality_gate"] = {"min_sharpness": 50.0, "min_mean_brightness": 60.0, "max_mean_brightness": 200.0,
                           "max_noise_sigma": 100.0}
    (tmp_path / "model.onnx").write_bytes((tiny_model_dir / "model.onnx").read_bytes())
    (tmp_path / "metadata.json").write_text(json.dumps(meta))
    clf = DefectClassifier(tmp_path, review_band=0.0)

    rng = np.random.default_rng(0)
    sharp = Image.fromarray(rng.integers(60, 200, (128, 128)).astype("uint8"))  # high-frequency texture
    flat = Image.new("L", (128, 128), 128)  # zero Laplacian variance == maximally "blurry"
    dark = Image.fromarray(rng.integers(0, 40, (128, 128)).astype("uint8"))
    (p_sharp, p_flat, p_dark), _ = clf.predict([sharp, flat, dark])
    assert p_sharp.quality_warnings == [] and not p_sharp.requires_review
    assert any(w.startswith("image_blurry") for w in p_flat.quality_warnings) and p_flat.requires_review
    assert any(w.startswith("exposure_out_of_range") for w in p_dark.quality_warnings) and p_dark.requires_review
    clf.quality_gate["max_noise_sigma"] = 1.0
    (p_noisy,), _ = clf.predict([sharp])
    assert any(w.startswith("image_noisy") for w in p_noisy.quality_warnings)
