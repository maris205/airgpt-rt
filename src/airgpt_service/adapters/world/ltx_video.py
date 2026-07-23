from __future__ import annotations

import base64
import binascii
import io

from PIL import Image

from ...errors import InvalidInputError
from ...schemas import WorldPredictRequest


def decode_image(value: str) -> Image.Image:
    raw = value.split(",", 1)[1] if value.startswith("data:") and "," in value else value
    try:
        payload = base64.b64decode(raw, validate=True)
        return Image.open(io.BytesIO(payload)).convert("RGB")
    except (ValueError, binascii.Error, OSError) as exc:
        raise InvalidInputError("image payload is not a decodable image") from exc


class LTXVideoAdapter:
    def prompt(self, request: WorldPredictRequest) -> str:
        parts = [value for value in (request.scene, request.action, request.prompt) if value]
        if not parts:
            raise InvalidInputError("LTX-Video requires scene, action, or prompt text")
        return ". ".join(parts)

    def preprocess(self, request: WorldPredictRequest):
        if not request.image:
            raise InvalidInputError("LTX-Video requires an initial conditioning image")
        if request.num_frames % 8 != 1:
            raise InvalidInputError("LTX-Video num_frames must be 8n + 1")
        return decode_image(request.image), self.prompt(request)
