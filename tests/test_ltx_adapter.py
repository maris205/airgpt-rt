import base64
import io

import pytest
from PIL import Image

from airgpt_service.adapters.world.ltx_video import LTXVideoAdapter
from airgpt_service.errors import InvalidInputError
from airgpt_service.schemas import WorldPredictRequest


def _image_payload() -> str:
    buffer = io.BytesIO()
    Image.new("RGB", (2, 2), "red").save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def test_ltx_adapter_requires_image_and_native_frame_count():
    adapter = LTXVideoAdapter()
    request = WorldPredictRequest(
        model="ltx_video", image=_image_payload(), action="move forward", num_frames=9
    )
    image, prompt = adapter.preprocess(request)
    assert image.size == (2, 2)
    assert prompt == "move forward"

    with pytest.raises(InvalidInputError):
        adapter.preprocess(request.model_copy(update={"num_frames": 3}))
