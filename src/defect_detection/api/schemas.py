"""Pydantic response models (they also drive the OpenAPI docs at /docs)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class PredictionResponse(BaseModel):
    request_id: str
    filename: str | None = None
    predicted_class: Literal["normal", "defective"]
    confidence: float = Field(..., ge=0, le=1, description="model probability of the predicted class")
    probabilities: dict[str, float]
    defect_probability: float = Field(..., ge=0, le=1)
    threshold: float = Field(..., description="decision threshold on defect_probability")
    requires_review: bool = Field(..., description="prediction is close to the threshold; route to manual inspection")
    model_version: str
    inference_ms: float

    model_config = {"protected_namespaces": (), "json_schema_extra": {"example": {
        "request_id": "4f1c2b9e8a7d4c3b", "filename": "part_0001.jpeg", "predicted_class": "defective",
        "confidence": 0.9931, "probabilities": {"normal": 0.0069, "defective": 0.9931},
        "defect_probability": 0.9931, "threshold": 0.31, "requires_review": False,
        "model_version": "efficientnet_b0-20261005-1a2b3c4d", "inference_ms": 21.4}}}


class BatchItem(BaseModel):
    filename: str | None
    ok: bool
    result: PredictionResponse | None = None
    error: dict | None = None


class BatchResponse(BaseModel):
    request_id: str
    model_version: str
    n_images: int
    n_failed: int
    items: list[BatchItem]

    model_config = {"protected_namespaces": ()}


class ErrorBody(BaseModel):
    code: str
    message: str
    request_id: str


class ErrorResponse(BaseModel):
    error: ErrorBody


class HealthResponse(BaseModel):
    status: Literal["ok"]


class ReadyResponse(BaseModel):
    status: Literal["ready", "not_ready"]
    model_loaded: bool
    model_version: str | None = None

    model_config = {"protected_namespaces": ()}
