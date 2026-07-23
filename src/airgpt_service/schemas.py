from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ErrorBody(BaseModel):
    code: str
    message: str
    request_id: str | None = None
    details: dict[str, Any] = Field(default_factory=dict)


class ModelInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    type: Literal["vla", "world"]
    backend: str
    adapter: str
    status: Literal["registered", "loading", "ready", "failed", "unloading"]
    loaded: bool
    device: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None


class ModelsResponse(BaseModel):
    models: list[ModelInfo]


class RobotActionRequest(BaseModel):
    model: str = Field(min_length=1)
    image: str = Field(
        min_length=1, description="Base64 image payload, optionally data-URL prefixed"
    )
    instruction: str = Field(min_length=1, max_length=2000)
    metadata: dict[str, Any] = Field(default_factory=dict)


class RobotActionResponse(BaseModel):
    model: str
    action: list[float]
    action_dim: int
    action_metadata: dict[str, Any] = Field(default_factory=dict)


class WorldPredictRequest(BaseModel):
    model: str = Field(min_length=1)
    image: str | None = Field(default=None, description="Optional base64 conditioning image")
    scene: str | None = Field(default=None, max_length=2000)
    action: str | None = Field(default=None, max_length=2000)
    prompt: str | None = Field(default=None, max_length=2000)
    num_frames: int = Field(default=9, ge=1, le=257)
    width: int = Field(default=384, ge=64, le=2048)
    height: int = Field(default=256, ge=64, le=2048)
    seed: int = Field(default=0, ge=0, le=2**31 - 1)

    @field_validator("width", "height")
    @classmethod
    def dimensions_multiple_of_eight(cls, value: int) -> int:
        if value % 8:
            raise ValueError("must be a multiple of 8")
        return value


class WorldPredictResponse(BaseModel):
    model: str
    frames: list[str]
    metadata: dict[str, Any] = Field(default_factory=dict)


class LoadResponse(BaseModel):
    model: ModelInfo
