import os
from pathlib import Path

import pytest

from airgpt_sdk import Client


pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(os.getenv("RUN_GPU_TESTS") != "1", reason="set RUN_GPU_TESTS=1"),
]


def test_openvla_live_api_is_deterministic():
    root = Path(__file__).resolve().parents[2]
    image = root / "artifacts" / "m0" / "openvla" / "synthetic_tabletop.png"
    with Client(os.getenv("AIRGPT_BASE_URL", "http://127.0.0.1:8000"), timeout=180) as client:
        first = client.robot.action("openvla", image, "pick up the red cube")
        second = client.robot.action("openvla", image, "pick up the red cube")
    assert first == second
    assert first["action_dim"] == 7
    assert first["action_metadata"]["unnorm_key"] == "bridge_orig"
