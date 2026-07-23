from __future__ import annotations

import base64
from pathlib import Path
from typing import Any

import httpx


class AirGPTError(RuntimeError):
    def __init__(self, status_code: int, body: dict[str, Any]):
        self.status_code = status_code
        self.body = body
        super().__init__(body.get("message", f"AirGPT request failed with {status_code}"))


class _Namespace:
    def __init__(self, client: "Client"):
        self._client = client


class RobotNamespace(_Namespace):
    def action(
        self, model: str, image: str | bytes | Path, instruction: str, metadata: dict | None = None
    ):
        if isinstance(image, Path):
            image = image.read_bytes()
        if isinstance(image, bytes):
            image = base64.b64encode(image).decode("ascii")
        return self._client._request(
            "POST",
            "/v1/robot/action",
            json={
                "model": model,
                "image": image,
                "instruction": instruction,
                "metadata": metadata or {},
            },
        )


class WorldNamespace(_Namespace):
    def predict(
        self,
        model: str,
        *,
        image: str | bytes | Path | None = None,
        scene: str | None = None,
        action: str | None = None,
        prompt: str | None = None,
        num_frames: int = 9,
        width: int = 384,
        height: int = 256,
        seed: int = 0,
        negative_prompt: str = "blurry, jittery, distorted, camera movement",
        num_inference_steps: int = 8,
        guidance_scale: float = 3.0,
        fps: int = 8,
    ):
        if isinstance(image, Path):
            image = image.read_bytes()
        if isinstance(image, bytes):
            image = base64.b64encode(image).decode("ascii")
        return self._client._request(
            "POST",
            "/v1/world/predict",
            json={
                "model": model,
                "image": image,
                "scene": scene,
                "action": action,
                "prompt": prompt,
                "num_frames": num_frames,
                "width": width,
                "height": height,
                "seed": seed,
                "negative_prompt": negative_prompt,
                "num_inference_steps": num_inference_steps,
                "guidance_scale": guidance_scale,
                "fps": fps,
            },
        )


class StateNamespace(_Namespace):
    def predict(
        self,
        model: str,
        observation: list[float],
        actions: list[int | list[float]],
        *,
        seed: int = 0,
        reset: bool = True,
    ):
        return self._client._request(
            "POST",
            "/v1/state/predict",
            json={
                "model": model,
                "observation": observation,
                "actions": actions,
                "seed": seed,
                "reset": reset,
            },
        )


class Client:
    def __init__(
        self, base_url: str, *, timeout: float = 60.0, http_client: httpx.Client | None = None
    ):
        self.base_url = base_url.rstrip("/")
        self._http = http_client or httpx.Client(base_url=self.base_url, timeout=timeout)
        self.robot = RobotNamespace(self)
        self.world = WorldNamespace(self)
        self.state = StateNamespace(self)

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> "Client":
        return self

    def __exit__(self, *_args) -> None:
        self.close()

    def models(self) -> dict[str, Any]:
        return self._request("GET", "/v1/models")

    def load(self, model: str) -> dict[str, Any]:
        return self._request("POST", f"/v1/models/{model}/load")

    def unload(self, model: str) -> dict[str, Any]:
        return self._request("POST", f"/v1/models/{model}/unload")

    def _request(self, method: str, path: str, **kwargs) -> dict[str, Any]:
        response = self._http.request(method, path, **kwargs)
        try:
            body = response.json()
        except ValueError:
            body = {"message": response.text}
        if response.is_error:
            raise AirGPTError(response.status_code, body)
        return body
