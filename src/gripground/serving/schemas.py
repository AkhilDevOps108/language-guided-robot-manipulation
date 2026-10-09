from __future__ import annotations

import math

from pydantic import BaseModel, Field, field_validator


class PredictRequest(BaseModel):
    instruction: str = Field(min_length=3, max_length=256)
    image_base64: str = Field(min_length=20)
    state: list[float] | None = None

    @field_validator("state")
    @classmethod
    def validate_state(cls, value: list[float] | None) -> list[float] | None:
        if value is None:
            return value
        if len(value) != 12:
            raise ValueError("state must have 12 float values when provided")
        if not all(math.isfinite(item) for item in value):
            raise ValueError("state values must be finite")
        return value


class PredictResponse(BaseModel):
    predicted_action: list[float]
    model_version: str
    inference_latency_ms: float
