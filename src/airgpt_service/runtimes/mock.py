from __future__ import annotations

import base64
import hashlib

from ..registry import RegistryModel
from ..schemas import (
    RobotActionRequest,
    RobotActionResponse,
    WorldPredictRequest,
    WorldPredictResponse,
)


def _decode_image_payload(value: str) -> bytes:
    if "," in value and value.startswith("data:"):
        value = value.split(",", 1)[1]
    return base64.b64decode(value, validate=True)


class MockVLARuntime:
    def __init__(self, spec: RegistryModel):
        self.spec = spec
        self.device = spec.device
        self.loaded = False

    def load(self) -> None:
        self.loaded = True

    def unload(self) -> None:
        self.loaded = False

    def predict_action(self, request: RobotActionRequest) -> RobotActionResponse:
        payload = _decode_image_payload(request.image)
        digest = hashlib.sha256(payload + request.instruction.encode()).digest()
        action = [round((byte / 255.0) * 2.0 - 1.0, 6) for byte in digest[:7]]
        return RobotActionResponse(
            model=request.model,
            action=action,
            action_dim=len(action),
            action_metadata={"runtime": "mock", "normalized": True, "space": "bridge_7dof"},
        )


class MockWorldRuntime:
    def __init__(self, spec: RegistryModel):
        self.spec = spec
        self.device = spec.device
        self.loaded = False

    def load(self) -> None:
        self.loaded = True

    def unload(self) -> None:
        self.loaded = False

    def predict_world(self, request: WorldPredictRequest) -> WorldPredictResponse:
        prompt = request.prompt or request.action or request.scene or "unspecified scene"
        frame_prefix = f"mock://{request.model}/{request.seed}/{hashlib.sha256(prompt.encode()).hexdigest()[:12]}"
        frames = [f"{frame_prefix}/frame-{index:03d}" for index in range(request.num_frames)]
        return WorldPredictResponse(
            model=request.model,
            frames=frames,
            metadata={
                "runtime": "mock",
                "width": request.width,
                "height": request.height,
                "num_frames": request.num_frames,
                "seed": request.seed,
                "prompt": prompt,
            },
        )
