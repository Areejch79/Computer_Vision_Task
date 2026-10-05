"""FastAPI inference service.

Endpoints
---------
GET  /health          liveness probe (process is up)
GET  /ready           readiness probe (model loaded and warmed up)
GET  /model           model metadata (version, threshold, validation metrics, ...)
POST /predict         single image (multipart field ``file``) -> class + confidence
POST /predict/batch   up to ``DD_MAX_BATCH_SIZE`` images (multipart field ``files``)
GET  /metrics         Prometheus metrics

Run locally::

    uvicorn defect_detection.api.main:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from starlette.responses import Response

from defect_detection import __version__
from defect_detection.api.errors import (
    APIError,
    InvalidImageError,
    ModelNotReadyError,
    PayloadTooLargeError,
    UnsupportedMediaTypeError,
    register_exception_handlers,
)
from defect_detection.api.logging_config import configure_logging, request_id_var
from defect_detection.api.schemas import (
    BatchItem,
    BatchResponse,
    ErrorResponse,
    HealthResponse,
    PredictionResponse,
    ReadyResponse,
)
from defect_detection.api.settings import Settings, get_settings
from defect_detection.inference import DefectClassifier, ModelLoadError
from defect_detection.preprocessing import load_image_bytes

log = logging.getLogger("api")

REQUESTS = Counter("dd_http_requests_total", "HTTP requests", ["method", "path", "status"])
LATENCY = Histogram("dd_http_request_seconds", "End-to-end request latency", ["path"],
                    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5))
PREDICTIONS = Counter("dd_predictions_total", "Predictions by class", ["predicted_class", "requires_review"])
INFER_LATENCY = Histogram("dd_model_inference_seconds", "Preprocess + model latency per call",
                          buckets=(0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2))
DEFECT_PROB = Histogram("dd_defect_probability", "Distribution of P(defective) (drift monitoring)",
                        buckets=tuple(i / 10 for i in range(1, 11)))

ERROR_RESPONSES = {
    413: {"model": ErrorResponse, "description": "file too large / too many files"},
    415: {"model": ErrorResponse, "description": "not an image / unsupported format"},
    422: {"model": ErrorResponse, "description": "corrupt image or invalid request"},
    503: {"model": ErrorResponse, "description": "model not loaded"},
}


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(settings.log_level, settings.log_json)
    Image.MAX_IMAGE_PIXELS = settings.max_image_pixels

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.classifier = None
        try:
            app.state.classifier = DefectClassifier(
                settings.model_dir, threshold=settings.threshold, review_band=settings.review_band,
                intra_op_threads=settings.intra_op_threads, onnx_filename=settings.onnx_filename)
            log.info("model ready", extra={"model_version": app.state.classifier.version,
                                           "threshold": app.state.classifier.threshold})
        except ModelLoadError:
            # Keep the process alive so /health works and /ready reports the problem;
            # the orchestrator will not route traffic to a not-ready pod.
            log.exception("model failed to load", extra={"model_dir": str(settings.model_dir)})
        yield

    app = FastAPI(
        title="Visual Defect Detection API",
        version=__version__,
        description="Classifies product images as **normal** or **defective**.",
        lifespan=lifespan,
    )
    app.state.settings = settings
    register_exception_handlers(app)
    if settings.cors_origins:
        app.add_middleware(CORSMiddleware, allow_origins=settings.cors_origins, allow_methods=["GET", "POST"],
                           allow_headers=["*"])

    @app.middleware("http")
    async def request_context(request: Request, call_next):
        rid = request.headers.get("x-request-id") or uuid.uuid4().hex[:16]
        token = request_id_var.set(rid)
        t0 = time.perf_counter()
        status = 500
        try:
            response = await call_next(request)
            status = response.status_code
            response.headers["x-request-id"] = rid
            return response
        finally:
            elapsed = time.perf_counter() - t0
            route = request.scope.get("route")
            path = getattr(route, "path", "unmatched")
            REQUESTS.labels(request.method, path, str(status)).inc()
            LATENCY.labels(path).observe(elapsed)
            if path not in ("/health", "/metrics"):
                log.info("request", extra={"method": request.method, "path": request.url.path, "status_code": status,
                                           "duration_ms": round(elapsed * 1000, 2),
                                           "client": request.client.host if request.client else None})
            request_id_var.reset(token)

    def classifier(request: Request) -> DefectClassifier:
        clf = request.app.state.classifier
        if clf is None:
            raise ModelNotReadyError()
        return clf

    async def read_and_validate(upload: UploadFile) -> Image.Image:
        """Validate an uploaded file and return a decoded PIL image."""
        ctype = (upload.content_type or "").lower()
        if ctype and not (ctype.startswith("image/") or ctype == "application/octet-stream"):
            raise UnsupportedMediaTypeError(f"content-type '{ctype}' is not an image")
        data = await upload.read(settings.max_upload_bytes + 1)
        if len(data) > settings.max_upload_bytes:
            raise PayloadTooLargeError(f"file exceeds {settings.max_upload_mb:g} MB")
        if not data:
            raise InvalidImageError("empty file")
        try:
            img = load_image_bytes(data)
        except Image.DecompressionBombError as exc:
            raise PayloadTooLargeError(f"image has too many pixels: {exc}") from exc
        except UnidentifiedImageError as exc:
            raise UnsupportedMediaTypeError("file is not a recognised image format") from exc
        except Exception as exc:  # truncated / corrupt data
            raise InvalidImageError(f"could not decode image: {exc}") from exc
        if img.format not in settings.allowed_formats:
            raise UnsupportedMediaTypeError(f"format {img.format} not allowed; use one of {settings.allowed_formats}")
        if min(img.size) < settings.min_image_side:
            raise InvalidImageError(f"image is {img.width}x{img.height}; minimum side is {settings.min_image_side}px")
        return img

    def to_response(pred, rid: str, filename: str | None, version: str, ms: float) -> PredictionResponse:
        PREDICTIONS.labels(pred.predicted_class, str(pred.requires_review).lower()).inc()
        DEFECT_PROB.observe(pred.defect_probability)
        return PredictionResponse(request_id=rid, filename=filename, model_version=version,
                                  inference_ms=round(ms, 2), **pred.to_dict())

    @app.get("/health", response_model=HealthResponse, tags=["ops"])
    async def health():
        return {"status": "ok"}

    @app.get("/ready", response_model=ReadyResponse, tags=["ops"], responses={503: {"model": ReadyResponse}})
    async def ready(request: Request):
        clf = request.app.state.classifier
        body = {"status": "ready" if clf else "not_ready", "model_loaded": clf is not None,
                "model_version": clf.version if clf else None}
        return body if clf else Response(content=ReadyResponse(**body).model_dump_json(), status_code=503,
                                         media_type="application/json")

    @app.get("/model", tags=["model"], responses={503: ERROR_RESPONSES[503]})
    async def model_info(request: Request):
        clf = classifier(request)
        return {**clf.metadata, "active_threshold": clf.threshold, "review_band": clf.review_band}

    @app.post("/predict", response_model=PredictionResponse, tags=["inference"], responses=ERROR_RESPONSES)
    async def predict(request: Request, file: UploadFile = File(..., description="product image (jpeg/png/bmp/tiff)")):
        clf = classifier(request)
        img = await read_and_validate(file)
        t0 = time.perf_counter()
        preds, _ = await run_in_threadpool(clf.predict, [img])
        ms = (time.perf_counter() - t0) * 1000
        INFER_LATENCY.observe(ms / 1000)
        resp = to_response(preds[0], request_id_var.get(), file.filename, clf.version, ms)
        log.info("prediction", extra={"upload_filename": file.filename, "predicted_class": resp.predicted_class,
                                      "confidence": resp.confidence, "defect_probability": resp.defect_probability,
                                      "requires_review": resp.requires_review, "inference_ms": resp.inference_ms})
        return resp

    @app.post("/predict/batch", response_model=BatchResponse, tags=["inference"], responses=ERROR_RESPONSES)
    async def predict_batch(request: Request, files: list[UploadFile] = File(...)):
        clf = classifier(request)
        if len(files) > settings.max_batch_size:
            raise PayloadTooLargeError(f"at most {settings.max_batch_size} files per request, got {len(files)}")
        rid = request_id_var.get()
        items: list[BatchItem | None] = [None] * len(files)
        valid: list[tuple[int, Image.Image]] = []
        for i, f in enumerate(files):  # a bad file fails only its own item, not the whole batch
            try:
                valid.append((i, await read_and_validate(f)))
            except APIError as exc:
                items[i] = BatchItem(filename=f.filename, ok=False, error=exc.body())
        if valid:
            t0 = time.perf_counter()
            preds, _ = await run_in_threadpool(clf.predict, [im for _, im in valid])
            ms = (time.perf_counter() - t0) * 1000
            INFER_LATENCY.observe(ms / 1000)
            for (i, _), pred in zip(valid, preds):
                items[i] = BatchItem(filename=files[i].filename, ok=True,
                                     result=to_response(pred, rid, files[i].filename, clf.version, ms / len(valid)))
        n_failed = sum(1 for it in items if not it.ok)
        log.info("batch prediction", extra={"n_images": len(files), "n_failed": n_failed})
        return BatchResponse(request_id=rid, model_version=clf.version, n_images=len(files), n_failed=n_failed,
                             items=items)

    @app.get("/metrics", tags=["ops"], include_in_schema=False)
    async def metrics():
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    return app


app = create_app()
