import os
from pathlib import Path

import pytest

from airgpt_sdk import Client


pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(os.getenv("RUN_GPU_TESTS") != "1", reason="set RUN_GPU_TESTS=1"),
]


def test_ltx_live_api_returns_bounded_artifacts():
    root = Path(__file__).resolve().parents[2]
    image = root / ".cache" / "samples" / "bridge_im_12.jpg"
    base_url = os.getenv("AIRGPT_WORLD_BASE_URL", "http://127.0.0.1:8001")
    with Client(base_url, timeout=240) as client:
        result = client.world.predict(
            "ltx_video",
            image=image,
            action="move the gripper toward the fork",
            num_frames=9,
            width=384,
            height=256,
            num_inference_steps=8,
            seed=0,
        )
    assert len(result["frames"]) == 9
    assert result["frames"][0].startswith("/artifacts/ltx_video/")
    assert result["metadata"]["video"].endswith("/future.mp4")
