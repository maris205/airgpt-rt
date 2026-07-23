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
