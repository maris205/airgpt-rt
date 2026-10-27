from pathlib import Path

import pytest

from airgpt_service.artifacts import ArtifactStore
from airgpt_service.errors import ServiceError
from airgpt_service.registry import RegistryModel
from airgpt_service.runtimes.diffusion_world import LTXVideoRuntime
from airgpt_service.runtimes.octo_jax import OctoJaxRuntime
from airgpt_service.runtimes.transformers_vla import TransformersVLARuntime


def _spec(adapter: str, model_id: str, local_path: Path) -> RegistryModel:
    return RegistryModel(
        type="world" if adapter == "ltx_video" else "vla",
        backend="test",
        adapter=adapter,
        model_id=model_id,
        metadata={"local_path": str(local_path), "local_files_only": True},
    )


@pytest.mark.parametrize("adapter", ["openvla", "octo", "ltx_video"])
def test_modelscope_runtimes_use_existing_local_checkpoint(tmp_path: Path, adapter: str):
    checkpoint = tmp_path / adapter
    checkpoint.mkdir()
    spec = _spec(adapter, f"owner/{adapter}", checkpoint)
    if adapter == "openvla":
        resolved = TransformersVLARuntime(spec)._model_ref()
    elif adapter == "octo":
        resolved = OctoJaxRuntime(spec)._model_path()
    else:
        resolved = LTXVideoRuntime(spec, ArtifactStore(tmp_path / "artifacts"))._model_ref()
    assert resolved == str(checkpoint)


@pytest.mark.parametrize("adapter", ["openvla", "octo", "ltx_video"])
def test_modelscope_runtimes_never_fall_back_to_remote_model_id(tmp_path: Path, adapter: str):
    checkpoint = tmp_path / "missing"
    spec = _spec(adapter, f"owner/{adapter}", checkpoint)
    with pytest.raises(ServiceError, match="checkpoint is missing"):
        if adapter == "openvla":
            TransformersVLARuntime(spec)._model_ref()
        elif adapter == "octo":
            OctoJaxRuntime(spec)._model_path()
        else:
            LTXVideoRuntime(spec, ArtifactStore(tmp_path / "artifacts"))._model_ref()
