import base64
from pathlib import Path

from fastapi.testclient import TestClient

from airgpt_service import create_app
from airgpt_service.config import Settings


IMAGE = base64.b64encode(b"mock-image-bytes").decode("ascii")


def test_local_web_console_is_served(client):
    root = client.get("/", follow_redirects=False)
    assert root.status_code == 307
    assert root.headers["location"] == "/ui/"
    page = client.get("/ui/")
    assert page.status_code == 200
    assert "AirGPT RT" in page.text
    assert client.get("/ui/app.js").status_code == 200


def test_app_can_expose_one_selected_model(tmp_path: Path):
    config = Path(__file__).resolve().parents[1] / "configs" / "models.yaml"
    settings = Settings(
        model_config_path=config,
        selected_model="octo",
        worker_mode="mock",
        artifact_dir=tmp_path / "artifacts",
    )
    with TestClient(create_app(settings=settings)) as selected_client:
        models = selected_client.get("/v1/models").json()["models"]
    assert [model["name"] for model in models] == ["octo"]


def test_health_and_registered_models(client):
    assert client.get("/health/live").json() == {"status": "ok"}
    ready = client.get("/health/ready")
    assert ready.status_code == 200
    assert ready.json() == {"status": "ok", "models": 4}

    response = client.get("/v1/models")
    assert response.status_code == 200
    models = response.json()["models"]
    assert [model["name"] for model in models] == [
        "openvla",
        "octo",
        "ltx_video",
        "dreamer_cartpole",
    ]
    assert all(model["status"] == "registered" for model in models)


def test_robot_action_lazy_load_and_unload(client):
    response = client.post(
        "/v1/robot/action",
        json={"model": "openvla", "image": IMAGE, "instruction": "pick up the cup"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["model"] == "openvla"
    assert body["action_dim"] == 7
    assert len(body["action"]) == 7
    assert body["action_metadata"]["runtime"] == "mock"

    assert client.get("/v1/models/openvla").json()["status"] == "ready"
    unload = client.post("/v1/models/openvla/unload")
    assert unload.status_code == 200
    assert unload.json()["model"]["status"] == "registered"


def test_world_prediction_contract(client):
    response = client.post(
        "/v1/world/predict",
        json={
            "model": "ltx_video",
            "scene": "robot kitchen",
            "action": "move the gripper toward the fork",
            "num_frames": 9,
            "width": 384,
            "height": 256,
            "seed": 7,
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["model"] == "ltx_video"
    assert len(body["frames"]) == 9
    assert body["frames"][0].startswith("mock://ltx_video/7/")
    assert body["metadata"]["width"] == 384


def test_model_type_mismatch_is_structured(client):
    response = client.post(
        "/v1/robot/action",
        json={"model": "ltx_video", "image": IMAGE, "instruction": "move"},
    )
    assert response.status_code == 422
    body = response.json()
    assert body["code"] == "model_type_mismatch"
    assert body["request_id"]


def test_invalid_image_and_unknown_model_errors(client):
    invalid = client.post(
        "/v1/robot/action",
        json={"model": "openvla", "image": "not-base64", "instruction": "move"},
    )
    assert invalid.status_code == 422
    assert invalid.json()["code"] == "invalid_input"

    missing = client.post("/v1/models/missing/load")
    assert missing.status_code == 404
    assert missing.json()["code"] == "model_not_found"


def test_pydantic_validation_error_is_structured(client):
    response = client.post(
        "/v1/world/predict",
        json={"model": "ltx_video", "scene": "warehouse", "width": 383},
    )
    assert response.status_code == 422
    body = response.json()
    assert body["code"] == "validation_error"
    assert body["details"]["errors"]


def test_world_requires_conditioning_input(client):
    response = client.post("/v1/world/predict", json={"model": "ltx_video"})
    assert response.status_code == 422
    assert response.json()["code"] == "invalid_input"


def test_state_prediction_contract(client):
    response = client.post(
        "/v1/state/predict",
        json={
            "model": "dreamer_cartpole",
            "observation": [0.01, 0.02, 0.03, 0.04],
            "actions": [0, 1, 0, 1],
            "seed": 7,
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["model"] == "dreamer_cartpole"
    assert len(body["latent_states"]) == 4
    assert len(body["latent_states"][0]) == 16
    assert body["metadata"]["runtime"] == "mock"
