from pathlib import Path

import pytest

from airgpt_service.errors import ModelNotFoundError, ServiceError
from airgpt_service.registry import ModelRegistry


def test_registry_rejects_empty_models(tmp_path: Path):
    path = tmp_path / "models.yaml"
    path.write_text("models: {}\n", encoding="utf-8")
    with pytest.raises(ServiceError) as exc_info:
        ModelRegistry(path)
    assert exc_info.value.code == "registry_invalid"


def test_registry_unknown_model():
    path = Path(__file__).resolve().parents[1] / "configs" / "models.yaml"
    registry = ModelRegistry(path)
    with pytest.raises(ModelNotFoundError):
        registry.get("missing")


def test_registry_filters_and_overrides_selected_model(tmp_path: Path):
    path = Path(__file__).resolve().parents[1] / "configs" / "models.yaml"
    model_path = tmp_path / "ltx"
    registry = ModelRegistry(
        path,
        selected_model="ltx_video",
        model_path=str(model_path),
        device="cuda:1",
        dtype="float16",
    )
    assert registry.names() == ["ltx_video"]
    spec = registry.get("ltx_video")
    assert spec.device == "cuda:1"
    assert spec.metadata["local_path"] == str(model_path)
    assert spec.metadata["dtype"] == "float16"
