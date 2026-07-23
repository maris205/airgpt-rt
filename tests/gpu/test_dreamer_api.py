import os

import pytest

from airgpt_sdk import Client


pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(os.getenv("RUN_GPU_TESTS") != "1", reason="set RUN_GPU_TESTS=1"),
]


def test_dreamer_live_api_returns_latent_rollout():
    base_url = os.getenv("AIRGPT_DREAMER_BASE_URL", "http://127.0.0.1:8002")
    with Client(base_url, timeout=60) as client:
        result = client.state.predict(
            "dreamer_cartpole",
            observation=[0.01, 0.02, 0.03, 0.04],
            actions=[0, 1, 0, 1, 0],
            seed=0,
        )
    assert len(result["latent_states"]) == 5
    assert len(result["latent_states"][0]) == 16
    assert len(result["predicted_observations"][0]) == 4
    assert result["metadata"]["runtime"] == "dreamerv3"
    assert result["metadata"]["task"] == "gym_CartPole-v1"
