from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .errors import ModelNotFoundError, ServiceError


class RegistryModel(BaseModel):
    model_config = ConfigDict(extra="allow", protected_namespaces=())

    type: str = Field(pattern="^(vla|world|state_world)$")
    backend: str = Field(min_length=1)
    adapter: str = Field(min_length=1)
    model_id: str | None = None
    device: str = "cpu"
    enabled: bool = True
    metadata: dict[str, Any] = Field(default_factory=dict)


class ModelRegistry:
    def __init__(
        self,
        path: Path,
        *,
        selected_model: str | None = None,
        model_path: str | None = None,
        revision: str | None = None,
        device: str | None = None,
        dtype: str | None = None,
        trust_remote_code: bool = False,
    ):
        self.path = path
        self._models = self._load(path)
        if selected_model:
            if selected_model not in self._models:
                raise ModelNotFoundError(selected_model)
            self._models = {selected_model: self._models[selected_model]}
            self._apply_overrides(
                self._models[selected_model],
                model_path=model_path,
                revision=revision,
                device=device,
                dtype=dtype,
                trust_remote_code=trust_remote_code,
            )

    @staticmethod
    def _apply_overrides(
        spec: RegistryModel,
        *,
        model_path: str | None,
        revision: str | None,
        device: str | None,
        dtype: str | None,
        trust_remote_code: bool,
    ) -> None:
        if device:
            spec.device = device
        if dtype:
            spec.metadata["dtype"] = dtype
        if revision:
            spec.metadata["revision"] = revision
        if trust_remote_code:
            spec.metadata["trust_remote_code"] = True
        if not model_path:
            return
        if spec.adapter == "openvla":
            spec.model_id = model_path
        elif spec.adapter in {"octo", "ltx_video"}:
            spec.metadata["local_path"] = model_path
            spec.metadata["local_files_only"] = True
        elif spec.adapter == "dreamerv3_cartpole":
            spec.metadata["checkpoint_path"] = model_path
        else:
            spec.model_id = model_path

    @staticmethod
    def _load(path: Path) -> dict[str, RegistryModel]:
        if not path.exists():
            raise ServiceError("registry_missing", f"model registry does not exist: {path}", 500)
        try:
            raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            models = raw.get("models", {})
            if not isinstance(models, dict) or not models:
                raise ValueError("models must be a non-empty mapping")
            parsed = {name: RegistryModel.model_validate(spec) for name, spec in models.items()}
        except (OSError, ValueError, ValidationError) as exc:
            raise ServiceError("registry_invalid", f"invalid model registry: {exc}", 500) from exc
        return parsed

    def names(self) -> list[str]:
        return list(self._models)

    def get(self, name: str) -> RegistryModel:
        try:
            spec = self._models[name]
        except KeyError as exc:
            raise ModelNotFoundError(name) from exc
        if not spec.enabled:
            raise ServiceError("model_disabled", f"model '{name}' is disabled", 404)
        return spec

    def items(self):
        return self._models.items()
