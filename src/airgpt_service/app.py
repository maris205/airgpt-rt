from __future__ import annotations

import base64
import binascii
import asyncio
import logging
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

from .config import Settings
from .artifacts import ArtifactStore
from .errors import InferenceTimeoutError, InvalidInputError, ServiceError
from .manager import ModelManager
from .registry import ModelRegistry
from .schemas import (
    ErrorBody,
    LoadResponse,
    ModelsResponse,
    RobotActionRequest,
    RobotActionResponse,
    StatePredictRequest,
    StatePredictResponse,
    WorldPredictRequest,
    WorldPredictResponse,
)


UI_DIR = Path(__file__).with_name("ui")


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
        artifacts = ArtifactStore(
            settings.artifact_dir,
            ttl_seconds=settings.artifact_ttl_seconds,
            max_runs=settings.max_artifact_runs,
        )
        manager = ModelManager(
            ModelRegistry(
                settings.model_config_path,
                selected_model=settings.selected_model,
                model_path=settings.model_path_override,
                revision=settings.revision_override,
                device=settings.device_override,
                dtype=settings.dtype_override,
                trust_remote_code=settings.trust_remote_code,
            ),
            settings.worker_mode,
            settings.max_concurrent_inference,
            artifacts,
        )
    settings.artifact_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=settings.log_level.upper())
    logger = logging.getLogger("airgpt.service")

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        settings.artifact_dir.mkdir(parents=True, exist_ok=True)
        if hasattr(manager, "artifacts"):
            manager.artifacts.cleanup()
        logger.info("AirGPT service started with %d registered models", len(manager.list_models()))
        if settings.preload_model:
            for model in manager.list_models():
                manager.load(model.name)
        yield
        for model in manager.list_models():
            if model.loaded:
                manager.unload(model.name)

    app = FastAPI(title="AirGPT Embodied AI Model Service", version="0.1.0", lifespan=lifespan)
    app.mount("/artifacts", StaticFiles(directory=settings.artifact_dir), name="artifacts")
    app.mount("/ui", StaticFiles(directory=UI_DIR, html=True), name="ui")

    @app.get("/", include_in_schema=False)
    async def index():
        return RedirectResponse(url="/ui/")

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
        return LoadResponse(model=await run_in_threadpool(manager.load, model_name))

    @app.post("/v1/models/{model_name}/unload", response_model=LoadResponse)
    async def unload_model(model_name: str):
        return LoadResponse(model=await run_in_threadpool(manager.unload, model_name))

    @app.post("/v1/robot/action", response_model=RobotActionResponse)
    async def robot_action(request: RobotActionRequest):
        _validate_image(request.image, settings.max_image_bytes)
        try:
            return await asyncio.wait_for(
                run_in_threadpool(manager.predict_action, request),
                timeout=settings.inference_timeout_seconds,
            )
        except asyncio.TimeoutError as exc:
            raise InferenceTimeoutError(settings.inference_timeout_seconds) from exc

    @app.post("/v1/world/predict", response_model=WorldPredictResponse)
    async def world_predict(request: WorldPredictRequest):
        if request.image is not None:
            _validate_image(request.image, settings.max_image_bytes)
        if not any((request.image, request.scene, request.action, request.prompt)):
            raise InvalidInputError("world prediction requires image, scene, action, or prompt")
        try:
            return await asyncio.wait_for(
                run_in_threadpool(manager.predict_world, request),
                timeout=settings.inference_timeout_seconds,
            )
        except asyncio.TimeoutError as exc:
            raise InferenceTimeoutError(settings.inference_timeout_seconds) from exc

    @app.post("/v1/state/predict", response_model=StatePredictResponse)
    async def state_predict(request: StatePredictRequest):
        try:
            return await asyncio.wait_for(
                run_in_threadpool(manager.predict_state, request),
                timeout=settings.inference_timeout_seconds,
            )
        except asyncio.TimeoutError as exc:
            raise InferenceTimeoutError(settings.inference_timeout_seconds) from exc

    return app
