import base64
import time
from pathlib import Path

from fastapi.testclient import TestClient

from airgpt_service import create_app
from airgpt_service.config import Settings
from airgpt_service.errors import ServiceBusyError
from airgpt_service.manager import ModelManager
from airgpt_service.registry import ModelRegistry
from airgpt_service.schemas import RobotActionRequest


CONFIG = Path(__file__).resolve().parents[1] / "configs" / "models.yaml"
IMAGE = base64.b64encode(b"mock-image").decode("ascii")


def test_busy_error_has_stable_contract():
    manager = ModelManager(ModelRegistry(CONFIG), max_concurrent_inference=1)
    assert manager._inference_slots.acquire(blocking=False)
    try:
        try:
            manager.predict_action(
                RobotActionRequest(model="openvla", image=IMAGE, instruction="move")
            )
        except ServiceBusyError as exc:
            assert exc.code == "service_busy"
            assert exc.status_code == 429
        else:
            raise AssertionError("busy manager accepted another inference")
    finally:
        manager._inference_slots.release()


def test_api_timeout_has_machine_readable_body(tmp_path):
    class SlowManager:
        def list_models(self):
            return []

        def predict_action(self, _request):
            time.sleep(0.05)

    settings = Settings(
        model_config_path=CONFIG,
        artifact_dir=tmp_path,
        inference_timeout_seconds=0.01,
    )
    with TestClient(create_app(settings=settings, manager=SlowManager())) as client:
        response = client.post(
            "/v1/robot/action",
            json={"model": "openvla", "image": IMAGE, "instruction": "move"},
        )
    assert response.status_code == 504
    assert response.json()["code"] == "inference_timeout"
