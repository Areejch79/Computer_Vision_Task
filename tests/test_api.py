import pytest
from PIL import Image

pytest.importorskip("onnxruntime")
from fastapi.testclient import TestClient  # noqa: E402

from defect_detection.api.main import create_app  # noqa: E402
from defect_detection.api.settings import Settings  # noqa: E402

from .conftest import encode  # noqa: E402


@pytest.fixture
def client(tiny_model_dir):
    settings = Settings(model_dir=tiny_model_dir, log_json=False, max_upload_mb=0.5, max_batch_size=3,
                        min_image_side=32)
    with TestClient(create_app(settings), raise_server_exceptions=False) as c:
        yield c


def _post(client, data: bytes, name="part.png", ctype="image/png"):
    return client.post("/predict", files={"file": (name, data, ctype)})


def test_health_and_ready(client):
    assert client.get("/health").json() == {"status": "ok"}
    r = client.get("/ready")
    assert r.status_code == 200 and r.json()["model_loaded"] is True


def test_model_info(client):
    info = client.get("/model").json()
    assert info["class_names"] == ["normal", "defective"] and info["active_threshold"] == 0.5


def test_predict_returns_class_and_confidence(client, dark_jpeg, bright_png):
    r = _post(client, dark_jpeg, "dark.jpg", "image/jpeg")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["predicted_class"] == "defective"
    assert 0.5 <= body["confidence"] <= 1.0
    assert set(body["probabilities"]) == {"normal", "defective"}
    assert body["model_version"] == "tiny-test"
    assert r.headers["x-request-id"] == body["request_id"]
    assert _post(client, bright_png).json()["predicted_class"] == "normal"


def test_request_id_is_propagated(client, bright_png):
    r = client.post("/predict", files={"file": ("a.png", bright_png, "image/png")}, headers={"x-request-id": "abc123"})
    assert r.headers["x-request-id"] == "abc123" and r.json()["request_id"] == "abc123"


def test_rejects_non_image_content_type(client):
    r = _post(client, b"hello", "a.txt", "text/plain")
    assert r.status_code == 415 and r.json()["error"]["code"] == "unsupported_media_type"


def test_rejects_non_image_bytes(client):
    r = _post(client, b"definitely not an image" * 10, "a.png", "image/png")
    assert r.status_code == 415


def test_rejects_corrupt_image(client):
    data = encode(Image.new("RGB", (300, 300), (10, 200, 30)), "JPEG")
    r = _post(client, data[: len(data) // 3], "broken.jpg", "image/jpeg")
    assert r.status_code == 422 and r.json()["error"]["code"] == "invalid_image"


def test_rejects_empty_file(client):
    assert _post(client, b"").status_code == 422


def test_rejects_too_large_file(client):
    import numpy as np

    noise = (np.random.default_rng(0).random((800, 800)) * 255).astype("uint8")
    data = encode(Image.fromarray(noise), "PNG")  # incompressible -> > 0.5 MB
    r = _post(client, data)
    assert r.status_code == 413 and r.json()["error"]["code"] == "payload_too_large"


def test_rejects_too_small_image(client):
    assert _post(client, encode(Image.new("L", (16, 16), 100))).status_code == 422


def test_rejects_disallowed_format(client):
    assert _post(client, encode(Image.new("RGB", (64, 64)), "GIF"), "a.gif", "image/gif").status_code == 415


def test_missing_file_field_is_422(client):
    r = client.post("/predict")
    assert r.status_code == 422 and r.json()["error"]["code"] == "validation_error"


def test_batch_partial_failure(client, dark_jpeg, bright_png):
    files = [("files", ("a.jpg", dark_jpeg, "image/jpeg")), ("files", ("b.txt", b"nope", "text/plain")),
             ("files", ("c.png", bright_png, "image/png"))]
    r = client.post("/predict/batch", files=files)
    assert r.status_code == 200
    body = r.json()
    assert body["n_images"] == 3 and body["n_failed"] == 1
    assert [it["ok"] for it in body["items"]] == [True, False, True]
    assert body["items"][0]["result"]["predicted_class"] == "defective"
    assert body["items"][1]["error"]["code"] == "unsupported_media_type"


def test_batch_too_many_files(client, bright_png):
    files = [("files", (f"{i}.png", bright_png, "image/png")) for i in range(4)]
    assert client.post("/predict/batch", files=files).status_code == 413


def test_metrics_endpoint(client, bright_png):
    _post(client, bright_png)
    text = client.get("/metrics").text
    assert "dd_predictions_total" in text and "dd_http_requests_total" in text


def test_not_ready_when_model_missing(tmp_path, bright_png):
    app = create_app(Settings(model_dir=tmp_path, log_json=False))
    with TestClient(app, raise_server_exceptions=False) as c:
        assert c.get("/health").status_code == 200
        assert c.get("/ready").status_code == 503
        r = c.post("/predict", files={"file": ("a.png", bright_png, "image/png")})
        assert r.status_code == 503 and r.json()["error"]["code"] == "model_not_ready"
