from __future__ import annotations

import os
import shlex
from pathlib import Path
from typing import Literal

import httpx
from pydantic import BaseModel, ConfigDict, Field, model_validator


DEFAULT_CATALOG_URL = (
    "https://modelscope.cn/api/v1/datasets/mariswang/AirGPT-Embodied-Model-Index/repo"
    "?Revision=master&FilePath=index.json"
)
BUNDLED_CATALOG = Path(__file__).with_name("hub_catalog") / "index.json"


class HubError(RuntimeError):
    pass


class HubSource(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: Literal["modelscope"] = "modelscope"
    repo_id: str = Field(pattern=r"^[^/]+/[^/]+$")
    revision: str = Field(min_length=7)
    gated: bool = False
    license: str
    license_url: str | None = None


class HubRuntime(BaseModel):
    model_config = ConfigDict(extra="forbid")

    registry_model: str
    backend: str
    adapter: str
    device: str = "cuda:0"
    dtype: str | None = None


class HubHardware(BaseModel):
    model_config = ConfigDict(extra="forbid")

    minimum_vram_gb: int = Field(ge=0)
    recommended_vram_gb: int = Field(ge=0)
    notes: str | None = None

    @model_validator(mode="after")
    def recommended_meets_minimum(self):
        if self.recommended_vram_gb < self.minimum_vram_gb:
            raise ValueError("recommended_vram_gb must be at least minimum_vram_gb")
        return self


class HubVerification(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["verified", "community", "unverified"]
    tested_at: str | None = None
    airgpt_version: str | None = None
    device: str | None = None
    checks: list[str] = Field(default_factory=list)


class HubModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    slug: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]*$")
    display_name: str
    summary: str
    task: Literal["vla", "world", "state_world"]
    tags: list[str] = Field(default_factory=list)
    source: HubSource
    runtime: HubRuntime
    hardware: HubHardware
    verification: HubVerification


class HubCatalog(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1]
    generated_at: str
    models: list[HubModel]

    @model_validator(mode="after")
    def unique_slugs(self):
        slugs = [model.slug for model in self.models]
        if len(slugs) != len(set(slugs)):
            raise ValueError("catalog model slugs must be unique")
        return self


class HubClient:
    def __init__(
        self,
        *,
        catalog_url: str | None = None,
        catalog_file: Path | None = None,
        timeout: float = 20.0,
    ):
        self.catalog_url = catalog_url or os.getenv("AIRGPT_HUB_CATALOG_URL", DEFAULT_CATALOG_URL)
        self.catalog_file = catalog_file
        self.timeout = timeout
        self.source = ""
        self._catalog: HubCatalog | None = None

    def catalog(self, *, refresh: bool = False) -> HubCatalog:
        if self._catalog is not None and not refresh:
            return self._catalog
        if self.catalog_file is not None:
            try:
                self._catalog = HubCatalog.model_validate_json(
                    self.catalog_file.read_text(encoding="utf-8")
                )
            except (OSError, ValueError) as exc:
                raise HubError(f"unable to load Hub catalog file: {exc}") from exc
            self.source = str(self.catalog_file)
            return self._catalog
        try:
            with httpx.Client(timeout=self.timeout, follow_redirects=True) as client:
                response = client.get(self.catalog_url)
                response.raise_for_status()
            self._catalog = HubCatalog.model_validate(response.json())
            self.source = self.catalog_url
        except (httpx.HTTPError, ValueError) as exc:
            if not BUNDLED_CATALOG.exists():
                raise HubError(f"unable to load Hub catalog: {exc}") from exc
            try:
                self._catalog = HubCatalog.model_validate_json(
                    BUNDLED_CATALOG.read_text(encoding="utf-8")
                )
            except (OSError, ValueError) as fallback_exc:
                raise HubError(f"unable to load bundled Hub catalog: {fallback_exc}") from exc
            self.source = f"bundled fallback ({exc})"
        return self._catalog

    def search(
        self,
        query: str = "",
        *,
        task: str | None = None,
        verified_only: bool = False,
        limit: int = 20,
    ) -> list[HubModel]:
        needle = query.casefold().strip()
        matches = []
        for model in self.catalog().models:
            haystack = " ".join(
                [model.slug, model.display_name, model.summary, model.source.repo_id, *model.tags]
            ).casefold()
            if needle and needle not in haystack:
                continue
            if task and model.task != task:
                continue
            if verified_only and model.verification.status != "verified":
                continue
            matches.append(model)
        matches.sort(key=lambda model: (model.verification.status != "verified", model.slug))
        return matches[:limit]

    def get(self, slug: str) -> HubModel:
        for model in self.catalog().models:
            if model.slug == slug or model.source.repo_id == slug:
                return model
        raise HubError(f"Hub model '{slug}' was not found")

    def pull(
        self,
        slug: str,
        *,
        cache_dir: Path | None = None,
        local_dir: Path | None = None,
        revision: str | None = None,
    ) -> tuple[HubModel, Path]:
        model = self.get(slug)
        try:
            from modelscope_hub import HubApi
        except ImportError as exc:
            raise HubError(
                "model downloads require `pip install 'airgpt-model-service[hub]'`"
            ) from exc
        kwargs = {
            "repo_id": model.source.repo_id,
            "repo_type": "model",
            "revision": revision or model.source.revision,
        }
        if cache_dir is not None:
            kwargs["cache_dir"] = str(cache_dir)
        if local_dir is not None:
            kwargs["local_dir"] = str(local_dir)
        try:
            path = HubApi(token=os.getenv("MODELSCOPE_API_TOKEN") or None).download_repo(**kwargs)
        except Exception as exc:
            raise HubError(f"failed to download '{model.source.repo_id}': {exc}") from exc
        return model, Path(path)


def launch_command(model: HubModel, model_path: Path | str) -> str:
    command = [
        "airgpt",
        "serve",
        "--model",
        model.runtime.registry_model,
        "--model-path",
        str(model_path),
        "--device",
        model.runtime.device,
    ]
    if model.runtime.dtype:
        command.extend(["--dtype", model.runtime.dtype])
    return shlex.join(command)
