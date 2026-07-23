import base64
import io

import numpy as np
import pytest
from PIL import Image

from airgpt_service.adapters.vla.octo import OctoAdapter
from airgpt_service.errors import InvalidInputError, ServiceError
from airgpt_service.registry import RegistryModel
from airgpt_service.schemas import RobotActionRequest


def _request(**updates):
    image = Image.new("RGB", (32, 24), "red")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    values = {
        "model": "octo",
        "image": base64.b64encode(buffer.getvalue()).decode("ascii"),
        "instruction": "pick up the fork",
        "metadata": {"seed": 7},
    }
    values.update(updates)
    return RobotActionRequest(**values)


def _adapter():
    return OctoAdapter(
        RegistryModel(
            type="vla",
            backend="octo_jax",
            adapter="octo",
            metadata={"dataset": "bridge_dataset", "image_size": 256},
        )
    )


def test_octo_adapter_preprocesses_bridge_contract():
    observation, instruction, seed = _adapter().preprocess(_request())
    assert observation["image_primary"].shape == (1, 1, 256, 256, 3)
    assert observation["image_primary"].dtype == np.uint8
    assert observation["timestep_pad_mask"].tolist() == [[True]]
    assert instruction == "pick up the fork"
    assert seed == 7


def test_octo_adapter_rejects_wrong_dataset_and_seed():
    adapter = _adapter()
    with pytest.raises(InvalidInputError):
        adapter.preprocess(_request(metadata={"dataset": "other"}))
    with pytest.raises(InvalidInputError):
        adapter.preprocess(_request(metadata={"seed": True}))


def test_octo_adapter_returns_first_action_and_chunk_metadata():
    adapter = _adapter()
    output = np.arange(28, dtype=np.float32).reshape(1, 4, 7) / 100
    response = adapter.postprocess(output, _request(), latency_seconds=0.25)
    assert response.action == output[0, 0].tolist()
    assert response.action_dim == 7
    assert response.action_metadata["chunk_horizon"] == 4
    assert len(response.action_metadata["action_chunk"]) == 4
    assert response.action_metadata["dataset"] == "bridge_dataset"

    with pytest.raises(ServiceError):
        adapter.postprocess(np.zeros((1, 4, 6)), _request(), latency_seconds=0.1)
