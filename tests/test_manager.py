from pathlib import Path

import pytest

from airgpt_service.errors import ServiceError
from airgpt_service.manager import ModelManager
from airgpt_service.registry import ModelRegistry


CONFIG = Path(__file__).resolve().parents[1] / "configs" / "models.yaml"


def test_real_worker_mode_fails_without_runtime_implementation():
    manager = ModelManager(ModelRegistry(CONFIG), worker_mode="real")
    with pytest.raises(ServiceError) as exc_info:
        manager.load("openvla")
    assert exc_info.value.code == "runtime_unavailable"
    assert manager.get_model("openvla").status == "failed"


def test_load_and_unload_are_idempotent():
    manager = ModelManager(ModelRegistry(CONFIG))
    first = manager.load("octo")
    second = manager.load("octo")
    assert first.status == second.status == "ready"
    assert manager.unload("octo").status == "registered"
    assert manager.unload("octo").status == "registered"
