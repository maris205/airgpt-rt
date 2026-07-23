import os
from pathlib import Path

import pytest

from airgpt_sdk import Client


pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(os.getenv("RUN_GPU_TESTS") != "1", reason="set RUN_GPU_TESTS=1"),
]


def test_octo_live_api_returns_bridge_action_chunk():
    root = Path(__file__).resolve().parents[2]
    image = root / ".cache" / "samples" / "bridge_im_12.jpg"
    base_url = os.getenv("AIRGPT_OCTO_BASE_URL", "http://127.0.0.1:8003")
    with Client(base_url, timeout=180) as client:
        result = client.robot.action(
            "octo",
            image=image,
            instruction="pick up the fork",
            metadata={"seed": 0, "dataset": "bridge_dataset"},
        )
        repeated = client.robot.action(
            "octo",
            image=image,
            instruction="pick up the fork",
            metadata={"seed": 0, "dataset": "bridge_dataset"},
        )
    assert result["action_dim"] == 7
    assert len(result["action"]) == 7
    assert result["action_metadata"]["runtime"] == "octo_jax"
    assert len(result["action_metadata"]["action_chunk"]) == 4
    assert result["action"] == repeated["action"]
    assert result["action_metadata"]["action_chunk"] == repeated["action_metadata"]["action_chunk"]
