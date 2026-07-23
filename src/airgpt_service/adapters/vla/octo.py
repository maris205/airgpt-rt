from __future__ import annotations

from typing import Any

import numpy as np

from ...errors import InvalidInputError, ServiceError
from ...registry import RegistryModel
from ...schemas import RobotActionRequest, RobotActionResponse
from .openvla import decode_image


class OctoAdapter:
    """Translate the shared robot contract to Octo's BridgeV2 interface."""

    action_dim = 7
    default_dataset = "bridge_dataset"
    default_image_size = 256

    def __init__(self, spec: RegistryModel):
        self.spec = spec
        self.dataset = str(spec.metadata.get("dataset", self.default_dataset))
        self.image_size = int(spec.metadata.get("image_size", self.default_image_size))
        self.chunk_horizon = int(spec.metadata.get("action_chunk_horizon", 4))

    def preprocess(self, request: RobotActionRequest) -> tuple[dict[str, np.ndarray], str, int]:
        image = decode_image(request.image).resize((self.image_size, self.image_size))
        pixels = np.asarray(image, dtype=np.uint8)[None, None, ...]
        observation = {
            "image_primary": pixels,
            "timestep_pad_mask": np.ones((1, 1), dtype=bool),
        }
        requested_dataset = request.metadata.get("dataset")
        if requested_dataset is not None and str(requested_dataset) != self.dataset:
            raise InvalidInputError(
                f"Octo dataset must be '{self.dataset}' for the configured checkpoint"
            )
        seed = request.metadata.get("seed", 0)
        if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 2**31 - 1:
            raise InvalidInputError("Octo metadata.seed must be an integer in [0, 2**31-1]")
        return observation, request.instruction, seed

    def postprocess(
        self,
        output: Any,
        request: RobotActionRequest,
        *,
        latency_seconds: float,
        runtime: str = "octo_jax",
    ) -> RobotActionResponse:
        values = np.asarray(output, dtype=np.float32)
        if values.ndim == 3:
            if values.shape[0] != 1:
                raise ServiceError(
                    "inference_failed", "Octo returned an unexpected batch size", 503
                )
            values = values[0]
        if values.ndim != 2 or values.shape[-1] != self.action_dim:
            raise ServiceError(
                "inference_failed",
                f"Octo returned action shape {tuple(values.shape)}, expected [horizon, 7]",
                503,
            )
        if not np.isfinite(values).all() or len(values) == 0:
            raise ServiceError("inference_failed", "Octo returned non-finite or empty actions", 503)
        chunk = np.round(values, 6).tolist()
        seed = int(request.metadata.get("seed", 0))
        return RobotActionResponse(
            model=request.model,
            action=chunk[0],
            action_dim=self.action_dim,
            action_metadata={
                "runtime": runtime,
                "normalized": False,
                "space": "bridge_7dof",
                "normalization": "dataset_statistics",
                "dataset": self.dataset,
                "model_revision": self.spec.metadata.get("model_revision"),
                "device": self.spec.device,
                "chunk_horizon": len(chunk),
                "action_chunk": chunk,
                "seed": seed,
                "latency_seconds": round(latency_seconds, 6),
            },
        )
