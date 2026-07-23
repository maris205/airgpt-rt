from __future__ import annotations

import base64
import binascii
import logging
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from .config import Settings
from .errors import InvalidInputError, ServiceError
from .manager import ModelManager
from .registry import ModelRegistry
from .schemas import (
    ErrorBody,
    LoadResponse,
    ModelsResponse,
    RobotActionRequest,
    RobotActionResponse,
    WorldPredictRequest,
    WorldPredictResponse,
)


def _validate_image(value: str, max_bytes: int) -> None:
    try:
        raw = value.split(",", 1)[1] if value.startswith("data:") and "," in value else value
        decoded = base64.b64decode(raw, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise InvalidInputError("image must be a valid base64 payload") from exc
    if not decoded:
        raise InvalidInputError("image payload is empty")
    if len(decoded) > max_bytes:
        raise InvalidInputError("image payload exceeds configured limit", {"max_bytes": max_bytes})


def _serializable_validation_errors(errors: list[dict]) -> list[dict]:
    """Convert Pydantic's exception objects in ``ctx`` into JSON-safe values."""
    normalized = []
    for error in errors:
        item = dict(error)
        ctx = item.get("ctx")
        if isinstance(ctx, dict):
            item["ctx"] = {
                key: str(value) if isinstance(value, BaseException) else value
                for key, value in ctx.items()
            }
        normalized.append(item)
    return normalized


def create_app(settings: Settings | None = None, manager: ModelManager | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    if manager is None:
        manager = ModelManager(ModelRegistry(settings.model_config_path), settings.worker_mode)
    logger = logging.getLogger("airgpt.service")

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        settings.artifact_dir.mkdir(parents=True, exist_ok=True)
        logger.info("AirGPT service started with %d registered models", len(manager.list_models()))
        yield
        for model in manager.list_models():
            if model.loaded:
                manager.unload(model.name)

    app = FastAPI(title="AirGPT Embodied AI Model Service", version="0.1.0", lifespan=lifespan)

    @app.middleware("http")
    async def request_id_middleware(request: Request, call_next):
        request_id = request.headers.get("x-request-id", str(uuid.uuid4()))
        request.state.request_id = request_id
        response = await call_next(request)
        response.headers["x-request-id"] = request_id
        return response

    @app.exception_handler(ServiceError)
    async def service_error_handler(request: Request, exc: ServiceError):
        body = ErrorBody(
            code=exc.code,
            message=exc.message,
            request_id=request.state.request_id,
            details=exc.details,
        )
        return JSONResponse(status_code=exc.status_code, content=jsonable_encoder(body))

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(request: Request, exc: RequestValidationError):
        body = ErrorBody(
            code="validation_error",
            message="request validation failed",
            request_id=request.state.request_id,
            details={"errors": _serializable_validation_errors(exc.errors())},
        )
        return JSONResponse(status_code=422, content=jsonable_encoder(body))

    @app.get("/health/live")
    async def live():
        return {"status": "ok"}

    @app.get("/health/ready")
    async def ready():
        return {"status": "ok", "models": len(manager.list_models())}

    @app.get("/v1/models", response_model=ModelsResponse)
    async def list_models():
        return ModelsResponse(models=manager.list_models())

    @app.get("/v1/models/{model_name}")
    async def get_model(model_name: str):
        return manager.get_model(model_name).info()

    @app.post("/v1/models/{model_name}/load", response_model=LoadResponse)
    async def load_model(model_name: str):
        return LoadResponse(model=manager.load(model_name))

    @app.post("/v1/models/{model_name}/unload", response_model=LoadResponse)
    async def unload_model(model_name: str):
        return LoadResponse(model=manager.unload(model_name))

    @app.post("/v1/robot/action", response_model=RobotActionResponse)
    async def robot_action(request: RobotActionRequest):
        _validate_image(request.image, settings.max_image_bytes)
        return manager.predict_action(request)

    @app.post("/v1/world/predict", response_model=WorldPredictResponse)
    async def world_predict(request: WorldPredictRequest):
        if request.image is not None:
            _validate_image(request.image, settings.max_image_bytes)
        if not any((request.image, request.scene, request.action, request.prompt)):
            raise InvalidInputError("world prediction requires image, scene, action, or prompt")
        return manager.predict_world(request)

    return app
