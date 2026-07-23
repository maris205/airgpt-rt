import base64
from pathlib import Path

from fastapi.testclient import TestClient

from airgpt_service import create_app
from airgpt_service.config import Settings
from airgpt_service.manager import ModelManager
from airgpt_service.registry import ModelRegistry
from airgpt_service.schemas import RobotActionResponse


CONFIG = Path(__file__).resolve().parents[1] / "configs" / "models.yaml"


class OctoWorkerStub:
    device = "cuda:0"

    def load(self):
        pass

    def unload(self):
        pass

    def predict_action(self, request):
        chunk = [[float(step + axis) for axis in range(7)] for step in range(4)]
        return RobotActionResponse(
            model=request.model,
            action=chunk[0],
            action_dim=7,
            action_metadata={
                "runtime": "octo_jax",
                "dataset": "bridge_dataset",
                "chunk_horizon": 4,
                "action_chunk": chunk,
            },
        )


def test_gateway_preserves_octo_worker_action_chunk(tmp_path):
    manager = ModelManager(ModelRegistry(CONFIG), worker_mode="real")
    model = manager.get_model("octo")
    model.runtime = OctoWorkerStub()
    model.status = "ready"
    settings = Settings(model_config_path=CONFIG, artifact_dir=tmp_path / "artifacts")

    with TestClient(create_app(settings=settings, manager=manager)) as client:
        response = client.post(
            "/v1/robot/action",
            json={
                "model": "octo",
                "image": base64.b64encode(b"image").decode("ascii"),
                "instruction": "pick up the fork",
                "metadata": {"seed": 0},
            },
        )

    assert response.status_code == 200
    body = response.json()
    assert body["action"] == body["action_metadata"]["action_chunk"][0]
    assert body["action_metadata"]["chunk_horizon"] == 4
    assert body["action_metadata"]["runtime"] == "octo_jax"
