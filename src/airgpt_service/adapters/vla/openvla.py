from __future__ import annotations

import base64
import binascii
import io
from typing import Any

from PIL import Image

from ...errors import InvalidInputError, ServiceError
from ...registry import RegistryModel
from ...schemas import RobotActionRequest, RobotActionResponse


def decode_image(value: str) -> Image.Image:
    raw = value.split(",", 1)[1] if value.startswith("data:") and "," in value else value
    try:
        payload = base64.b64decode(raw, validate=True)
        return Image.open(io.BytesIO(payload)).convert("RGB")
    except (ValueError, binascii.Error, OSError) as exc:
        raise InvalidInputError("image payload is not a decodable image") from exc


class OpenVLAAdapter:
    """Translate the public action contract to OpenVLA's native interface."""

    def __init__(self, spec: RegistryModel):
        self.spec = spec
        self.unnorm_key = str(spec.metadata.get("unnorm_key", "bridge_orig"))

    def prompt(self, instruction: str) -> str:
        return f"In: What action should the robot take to {instruction.lower()}?\nOut:"

    def preprocess(self, request: RobotActionRequest) -> tuple[str, Image.Image]:
        return self.prompt(request.instruction), decode_image(request.image)

    def postprocess(
        self, model: Any, inputs: Any, request: RobotActionRequest
    ) -> RobotActionResponse:
        try:
            output = model.predict_action(**inputs, unnorm_key=self.unnorm_key, do_sample=False)
            values = output.tolist() if hasattr(output, "tolist") else list(output)
            action = [round(float(value), 6) for value in values]
        except (RuntimeError, ValueError, TypeError) as exc:
            if "out of memory" in str(exc).lower():
                raise ServiceError(
                    "inference_oom", "OpenVLA inference exhausted GPU memory", 503
                ) from exc
            raise ServiceError("inference_failed", f"OpenVLA inference failed: {exc}", 503) from exc
        if not action:
            raise ServiceError("inference_failed", "OpenVLA returned an empty action", 503)
        return RobotActionResponse(
            model=request.model,
            action=action,
            action_dim=len(action),
            action_metadata={
                "runtime": "transformers",
                "normalized": False,
                "space": "bridge_7dof",
                "unnorm_key": self.unnorm_key,
            },
        )
