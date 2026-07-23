import base64
import io

from PIL import Image

from airgpt_service.adapters.vla.openvla import OpenVLAAdapter
from airgpt_service.registry import RegistryModel
from airgpt_service.schemas import RobotActionRequest


def test_openvla_preprocess_builds_native_prompt_and_decodes_image():
    buffer = io.BytesIO()
    Image.new("RGB", (2, 2), "red").save(buffer, format="PNG")
    request = RobotActionRequest(
        model="openvla",
        image=base64.b64encode(buffer.getvalue()).decode("ascii"),
        instruction="Pick Up The Cup",
    )
    adapter = OpenVLAAdapter(
        RegistryModel(
            type="vla",
            backend="transformers",
            adapter="openvla",
            model_id="openvla/openvla-7b",
            metadata={"unnorm_key": "bridge_orig"},
        )
    )
    prompt, image = adapter.preprocess(request)
    assert prompt == "In: What action should the robot take to pick up the cup?\nOut:"
    assert image.size == (2, 2)
