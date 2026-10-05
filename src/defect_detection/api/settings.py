"""API configuration (12-factor: everything overridable through ``DD_*`` environment variables)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="DD_", env_file=".env", extra="ignore", protected_namespaces=())

    model_dir: Path = Field(Path("models/efficientnet_b0"), description="directory with model.onnx + metadata.json")
    onnx_filename: str = "model.onnx"
    threshold: float | None = Field(None, gt=0, lt=1, description="override the tuned threshold in metadata.json")
    review_band: float = Field(0.15, ge=0, lt=0.5, description="|P(def) - threshold| below this -> requires_review")
    intra_op_threads: int | None = Field(None, ge=1)

    max_upload_mb: float = Field(10.0, gt=0)
    max_batch_size: int = Field(16, ge=1)
    min_image_side: int = Field(64, ge=1)
    max_image_pixels: int = Field(40_000_000, ge=1, description="decompression-bomb guard")
    allowed_formats: tuple[str, ...] = ("JPEG", "PNG", "BMP", "TIFF", "WEBP")

    log_level: str = "INFO"
    log_json: bool = True
    cors_origins: list[str] = []

    @property
    def max_upload_bytes(self) -> int:
        return int(self.max_upload_mb * 1024 * 1024)


@lru_cache
def get_settings() -> Settings:
    return Settings()
