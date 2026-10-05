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
