"""Request/response models for the inference API."""
from __future__ import annotations

from pydantic import BaseModel, Field


class Prediction(BaseModel):
    label: str
    probability: float = Field(ge=0.0, le=1.0)


class PredictResponse(BaseModel):
    predictions: list[Prediction]
    model_version: str
    inference_ms: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str
