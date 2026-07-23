from __future__ import annotations

from dataclasses import dataclass
from threading import BoundedSemaphore, RLock
from typing import Any

from .errors import ModelNotFoundError, ServiceBusyError, ServiceError
from .registry import ModelRegistry, RegistryModel
from .runtimes.mock import MockVLARuntime, MockWorldRuntime
from .runtimes.transformers_vla import TransformersVLARuntime
from .schemas import (
    ModelInfo,
    RobotActionRequest,
    RobotActionResponse,
    WorldPredictRequest,
    WorldPredictResponse,
)


@dataclass
class ManagedModel:
    name: str
    spec: RegistryModel
    runtime: Any | None = None
    status: str = "registered"
    error: str | None = None

    def info(self) -> ModelInfo:
        return ModelInfo(
            name=self.name,
            type=self.spec.type,
            backend=self.spec.backend,
            adapter=self.spec.adapter,
            status=self.status,
            loaded=self.runtime is not None and self.status == "ready",
            device=self.spec.device,
            metadata={**self.spec.metadata, "model_id": self.spec.model_id},
            error=self.error,
        )


class ModelManager:
    def __init__(
        self, registry: ModelRegistry, worker_mode: str = "mock", max_concurrent_inference: int = 1
    ):
        self.registry = registry
        self.worker_mode = worker_mode
        self._models = {name: ManagedModel(name, spec) for name, spec in registry.items()}
        self._lock = RLock()
        self._inference_slots = BoundedSemaphore(max_concurrent_inference)

    def list_models(self) -> list[ModelInfo]:
        with self._lock:
            return [model.info() for model in self._models.values()]

    def get_model(self, name: str) -> ManagedModel:
        try:
            return self._models[name]
        except KeyError as exc:
            raise ModelNotFoundError(name) from exc

    def load(self, name: str) -> ModelInfo:
        with self._lock:
            model = self.get_model(name)
            if model.status == "ready":
                return model.info()
            model.status = "loading"
            try:
                if self.worker_mode == "mock":
                    runtime = (
                        MockVLARuntime(model.spec)
                        if model.spec.type == "vla"
                        else MockWorldRuntime(model.spec)
                    )
                elif self.worker_mode == "real" and model.spec.adapter == "openvla":
                    runtime = TransformersVLARuntime(model.spec)
                else:
                    raise ServiceError(
                        "runtime_unavailable",
                        f"no '{self.worker_mode}' runtime is available for '{name}'",
                        503,
                    )
                runtime.load()
                model.runtime = runtime
                model.status = "ready"
                model.error = None
            except ServiceError as exc:
                model.status = "failed"
                model.error = exc.message
                raise
            except Exception as exc:
                model.status = "failed"
                model.error = str(exc)
                raise ServiceError(
                    "model_load_failed", f"failed to load '{name}': {exc}", 503
                ) from exc
            return model.info()

    def unload(self, name: str) -> ModelInfo:
        with self._lock:
            model = self.get_model(name)
            if model.runtime is None:
                model.status = "registered"
                return model.info()
            model.status = "unloading"
            model.runtime.unload()
            model.runtime = None
            model.status = "registered"
            return model.info()

    def predict_action(self, request: RobotActionRequest) -> RobotActionResponse:
        if not self._inference_slots.acquire(blocking=False):
            raise ServiceBusyError()
        try:
            with self._lock:
                model = self.get_model(request.model)
                if model.spec.type != "vla":
                    raise ServiceError(
                        "model_type_mismatch", f"'{request.model}' is not a VLA model", 422
                    )
                self.load(request.model)
                return model.runtime.predict_action(request)
        finally:
            self._inference_slots.release()

    def predict_world(self, request: WorldPredictRequest) -> WorldPredictResponse:
        if not self._inference_slots.acquire(blocking=False):
            raise ServiceBusyError()
        try:
            with self._lock:
                model = self.get_model(request.model)
                if model.spec.type != "world":
                    raise ServiceError(
                        "model_type_mismatch", f"'{request.model}' is not a world model", 422
                    )
                self.load(request.model)
                return model.runtime.predict_world(request)
        finally:
            self._inference_slots.release()
