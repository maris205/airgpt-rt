import json
from pathlib import Path

import httpx
import pytest

from airgpt_sdk import AirGPTError, Client


def test_sdk_encodes_image_and_exposes_namespaces(tmp_path: Path):
    image = tmp_path / "image.bin"
    image.write_bytes(b"image-data")

    def handler(request: httpx.Request):
        body = json.loads(request.content)
        assert body["model"] == "openvla"
        assert body["image"] == "aW1hZ2UtZGF0YQ=="
        return httpx.Response(200, json={"model": "openvla", "action": [0.0] * 7, "action_dim": 7})

    http_client = httpx.Client(base_url="http://test", transport=httpx.MockTransport(handler))
    with Client("http://test", http_client=http_client) as client:
        result = client.robot.action("openvla", image, "pick cup")
    assert result["action_dim"] == 7


def test_sdk_raises_structured_error():
    def handler(_request: httpx.Request):
        return httpx.Response(404, json={"code": "model_not_found", "message": "missing"})

    http_client = httpx.Client(base_url="http://test", transport=httpx.MockTransport(handler))
    client = Client("http://test", http_client=http_client)
    with pytest.raises(AirGPTError) as exc_info:
        client.load("missing")
    assert exc_info.value.status_code == 404
    client.close()


def test_world_sdk_sends_generation_controls():
    def handler(request: httpx.Request):
        body = json.loads(request.content)
        assert body["num_inference_steps"] == 6
        assert body["guidance_scale"] == 2.5
        assert body["fps"] == 12
        return httpx.Response(200, json={"model": "ltx_video", "frames": [], "metadata": {}})

    http_client = httpx.Client(base_url="http://test", transport=httpx.MockTransport(handler))
    with Client("http://test", http_client=http_client) as client:
        client.world.predict(
            "ltx_video",
            image=b"image",
            action="move",
            num_inference_steps=6,
            guidance_scale=2.5,
            fps=12,
        )
