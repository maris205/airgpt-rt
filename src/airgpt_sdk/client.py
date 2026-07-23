from __future__ import annotations

import base64
import time
from pathlib import Path
from typing import Any, TypedDict, cast

import httpx


class AirGPTError(RuntimeError):
    def __init__(self, status_code: int, body: dict[str, Any]):
        self.status_code = status_code
        self.body = body
        self.code = str(body.get("code", "request_failed"))
        self.request_id = body.get("request_id")
        self.details = body.get("details", {})
        super().__init__(body.get("message", f"AirGPT request failed with {status_code}"))


class AirGPTModelNotFoundError(AirGPTError):
    pass


class AirGPTInvalidInputError(AirGPTError):
    pass


class AirGPTServiceUnavailableError(AirGPTError):
    pass


class AirGPTTimeoutError(AirGPTError):
    pass


class RobotActionResult(TypedDict):
    model: str
    action: list[float]
    action_dim: int
    action_metadata: dict[str, Any]


class WorldPredictResult(TypedDict):
    model: str
    frames: list[str]
    metadata: dict[str, Any]


class StatePredictResult(TypedDict):
    model: str
    latent_states: list[list[float]]
    predicted_observations: list[list[float]]
    predicted_rewards: list[float]
    uncertainty: list[float]
    metadata: dict[str, Any]


_ERROR_TYPES: dict[str, type[AirGPTError]] = {
    "model_not_found": AirGPTModelNotFoundError,
    "invalid_input": AirGPTInvalidInputError,
    "validation_error": AirGPTInvalidInputError,
    "runtime_unavailable": AirGPTServiceUnavailableError,
    "model_load_failed": AirGPTServiceUnavailableError,
    "model_load_oom": AirGPTServiceUnavailableError,
    "inference_failed": AirGPTServiceUnavailableError,
    "inference_oom": AirGPTServiceUnavailableError,
    "service_busy": AirGPTServiceUnavailableError,
    "inference_timeout": AirGPTTimeoutError,
    "transport_error": AirGPTServiceUnavailableError,
}


def encode_image(image: str | bytes | Path) -> str:
    if isinstance(image, Path):
        image = image.read_bytes()
    if isinstance(image, bytes):
        return base64.b64encode(image).decode("ascii")
    return image


class _Namespace:
    def __init__(self, client: "Client"):
        self._client = client


class RobotNamespace(_Namespace):
    def action(
        self, model: str, image: str | bytes | Path, instruction: str, metadata: dict | None = None
    ) -> RobotActionResult:
        return cast(
            RobotActionResult,
            self._client._request(
                "POST",
                "/v1/robot/action",
                json={
                    "model": model,
                    "image": encode_image(image),
                    "instruction": instruction,
                    "metadata": metadata or {},
                },
            ),
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
    ) -> WorldPredictResult:
        return cast(
            WorldPredictResult,
            self._client._request(
                "POST",
                "/v1/world/predict",
                json={
                    "model": model,
                    "image": encode_image(image) if image is not None else None,
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
            ),
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
    ) -> StatePredictResult:
        return cast(
            StatePredictResult,
            self._client._request(
                "POST",
                "/v1/state/predict",
                json={
                    "model": model,
                    "observation": observation,
                    "actions": actions,
                    "seed": seed,
                    "reset": reset,
                },
            ),
        )


class Client:
    def __init__(
        self,
        base_url: str,
        *,
        timeout: float = 60.0,
        retries: int = 2,
        retry_backoff: float = 0.25,
        http_client: httpx.Client | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.retries = max(0, retries)
        self.retry_backoff = max(0.0, retry_backoff)
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
        return self._request("GET", "/v1/models", retryable=True)

    def model(self, model: str) -> dict[str, Any]:
        return self._request("GET", f"/v1/models/{model}", retryable=True)

    def load(self, model: str) -> dict[str, Any]:
        return self._request("POST", f"/v1/models/{model}/load", retryable=True)

    def unload(self, model: str) -> dict[str, Any]:
        return self._request("POST", f"/v1/models/{model}/unload", retryable=True)

    def _request(
        self, method: str, path: str, *, retryable: bool = False, **kwargs
    ) -> dict[str, Any]:
        attempts = self.retries + 1 if retryable else 1
        for attempt in range(attempts):
            try:
                response = self._http.request(method, path, **kwargs)
            except httpx.TransportError as exc:
                if attempt + 1 < attempts:
                    time.sleep(self.retry_backoff * (2**attempt))
                    continue
                body = {"code": "transport_error", "message": str(exc)}
                raise AirGPTServiceUnavailableError(0, body) from exc
            try:
                parsed = response.json()
                body = parsed if isinstance(parsed, dict) else {"message": str(parsed)}
            except ValueError:
                body = {"message": response.text}
            if response.is_error:
                if retryable and response.status_code in {502, 503, 504} and attempt + 1 < attempts:
                    time.sleep(self.retry_backoff * (2**attempt))
                    continue
                error_type = _ERROR_TYPES.get(str(body.get("code")), AirGPTError)
                raise error_type(response.status_code, body)
            return body
        raise AssertionError("request retry loop ended unexpectedly")
