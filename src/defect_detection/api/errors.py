"""Domain errors -> consistent JSON error envelopes ``{"error": {"code", "message", "request_id"}}``."""

from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from defect_detection.api.logging_config import request_id_var

log = logging.getLogger("api.errors")


class APIError(Exception):
    def __init__(self, status_code: int, code: str, message: str):
        super().__init__(message)
        self.status_code, self.code, self.message = status_code, code, message

    def body(self) -> dict:
        return {"code": self.code, "message": self.message}


class InvalidImageError(APIError):
    def __init__(self, message: str):
        super().__init__(422, "invalid_image", message)


class UnsupportedMediaTypeError(APIError):
    def __init__(self, message: str):
        super().__init__(415, "unsupported_media_type", message)


class PayloadTooLargeError(APIError):
    def __init__(self, message: str):
        super().__init__(413, "payload_too_large", message)


class ModelNotReadyError(APIError):
    def __init__(self, message: str = "model is not loaded"):
        super().__init__(503, "model_not_ready", message)


def _envelope(status: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(status_code=status,
                        content={"error": {"code": code, "message": message, "request_id": request_id_var.get()}})


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(APIError)
    async def _api_error(_: Request, exc: APIError):
        log.warning("request rejected", extra={"error_code": exc.code, "detail": exc.message,
                                               "status_code": exc.status_code})
        return _envelope(exc.status_code, exc.code, exc.message)

    @app.exception_handler(RequestValidationError)
    async def _validation(_: Request, exc: RequestValidationError):
        msg = "; ".join(f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors())
        log.warning("request validation failed", extra={"detail": msg})
        return _envelope(422, "validation_error", msg)

    @app.exception_handler(StarletteHTTPException)
    async def _http(_: Request, exc: StarletteHTTPException):
        return _envelope(exc.status_code, "http_error", str(exc.detail))

    @app.exception_handler(Exception)
    async def _unhandled(_: Request, exc: Exception):
        log.exception("unhandled error")
        return _envelope(500, "internal_error", "internal server error")
