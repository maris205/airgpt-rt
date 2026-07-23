from pathlib import Path

import pytest

from airgpt_service.errors import ServiceError
from airgpt_service.manager import ModelManager
from airgpt_service.registry import ModelRegistry


CONFIG = Path(__file__).resolve().parents[1] / "configs" / "models.yaml"


def test_real_worker_mode_fails_for_unsupported_runtime(tmp_path):
    config = tmp_path / "models.yaml"
    config.write_text(
        "models:\n  unsupported:\n    type: vla\n    backend: custom\n"
        "    adapter: unsupported\n    model_id: example/model\n",
        encoding="utf-8",
    )
    manager = ModelManager(ModelRegistry(config), worker_mode="real")
    with pytest.raises(ServiceError) as exc_info:
        manager.load("unsupported")
    assert exc_info.value.code == "runtime_unavailable"
    assert manager.get_model("unsupported").status == "failed"


def test_real_worker_mode_selects_openvla_runtime(monkeypatch):
    class FakeRuntime:
        def __init__(self, spec):
            self.device = spec.device
            self.loaded = False

        def load(self):
            self.loaded = True

        def unload(self):
            self.loaded = False

    monkeypatch.setattr("airgpt_service.manager.TransformersVLARuntime", FakeRuntime)
    manager = ModelManager(ModelRegistry(CONFIG), worker_mode="real")
    assert manager.load("openvla").status == "ready"
    assert manager.get_model("openvla").runtime.loaded


def test_real_worker_mode_selects_octo_runtime(monkeypatch):
    class FakeRuntime:
        def __init__(self, spec):
            self.device = spec.device
            self.loaded = False

        def load(self):
            self.loaded = True

        def unload(self):
            self.loaded = False

    monkeypatch.setattr("airgpt_service.manager.OctoJaxRuntime", FakeRuntime)
    manager = ModelManager(ModelRegistry(CONFIG), worker_mode="real")
    assert manager.load("octo").status == "ready"
    assert manager.get_model("octo").runtime.loaded


def test_real_worker_mode_selects_ltx_runtime(monkeypatch, tmp_path):
    class FakeRuntime:
        def __init__(self, spec, artifacts):
            self.device = spec.device
            self.artifacts = artifacts
            self.loaded = False

        def load(self):
            self.loaded = True

        def unload(self):
            self.loaded = False

    monkeypatch.setattr("airgpt_service.manager.LTXVideoRuntime", FakeRuntime)
    from airgpt_service.artifacts import ArtifactStore

    manager = ModelManager(
        ModelRegistry(CONFIG), worker_mode="real", artifact_store=ArtifactStore(tmp_path)
    )
    assert manager.load("ltx_video").status == "ready"
    assert manager.get_model("ltx_video").runtime.loaded


def test_real_worker_mode_selects_dreamer_runtime(monkeypatch):
    class FakeRuntime:
        def __init__(self, spec):
            self.device = spec.device
            self.loaded = False

        def load(self):
            self.loaded = True

        def unload(self):
            self.loaded = False

    monkeypatch.setattr("airgpt_service.manager.DreamerV3StateRuntime", FakeRuntime)
    manager = ModelManager(ModelRegistry(CONFIG), worker_mode="real")
    assert manager.load("dreamer_cartpole").status == "ready"
    assert manager.get_model("dreamer_cartpole").runtime.loaded


def test_load_and_unload_are_idempotent():
    manager = ModelManager(ModelRegistry(CONFIG))
    first = manager.load("octo")
    second = manager.load("octo")
    assert first.status == second.status == "ready"
    assert manager.unload("octo").status == "registered"
    assert manager.unload("octo").status == "registered"
