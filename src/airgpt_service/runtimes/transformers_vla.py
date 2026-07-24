from __future__ import annotations

import gc
from typing import Any

from ..adapters.vla.openvla import OpenVLAAdapter
from ..errors import ServiceError
from ..registry import RegistryModel
from ..schemas import RobotActionRequest, RobotActionResponse


class TransformersVLARuntime:
    """Lazy OpenVLA runtime; heavyweight imports happen only in real mode."""

    def __init__(self, spec: RegistryModel):
        self.spec = spec
        self.device = spec.device or "cuda:0"
        self.adapter = OpenVLAAdapter(spec)
        self.processor: Any | None = None
        self.model: Any | None = None
        self._torch: Any | None = None
        self._dtype: Any | None = None
        self.loaded = False

    def load(self) -> None:
        if self.loaded:
            return
        try:
            import torch
            from transformers import AutoModelForVision2Seq, AutoProcessor

            self._torch = torch
            dtype_name = str(
                self.spec.metadata.get(
                    "dtype", "bfloat16" if self.device.startswith("cuda") else "float32"
                )
            )
            dtype_map = {
                "float16": torch.float16,
                "bfloat16": torch.bfloat16,
                "float32": torch.float32,
            }
            if dtype_name not in dtype_map:
                raise ServiceError(
                    "model_load_failed",
                    f"unsupported OpenVLA dtype '{dtype_name}'",
                    503,
                )
            dtype = dtype_map[dtype_name]
            self._dtype = dtype
            model_id = self.spec.model_id
            if not model_id:
                raise ServiceError("model_load_failed", "OpenVLA model_id is required", 503)
            local_files_only = bool(self.spec.metadata.get("local_files_only", False))
            trust_remote_code = bool(self.spec.metadata.get("trust_remote_code", False))
            revision = self.spec.metadata.get("revision")
            self.processor = AutoProcessor.from_pretrained(
                model_id,
                revision=revision,
                trust_remote_code=trust_remote_code,
                local_files_only=local_files_only,
            )
            self.model = AutoModelForVision2Seq.from_pretrained(
                model_id,
                revision=revision,
                attn_implementation=str(self.spec.metadata.get("attention", "sdpa")),
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
                trust_remote_code=trust_remote_code,
                local_files_only=local_files_only,
            ).to(self.device)
            self.model.eval()
            self.loaded = True
        except ServiceError:
            raise
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                self.unload()
                raise ServiceError(
                    "model_load_oom", "OpenVLA load exhausted GPU memory", 503
                ) from exc
            self.unload()
            raise ServiceError("model_load_failed", f"OpenVLA load failed: {exc}", 503) from exc
        except Exception as exc:
            self.unload()
            raise ServiceError("model_load_failed", f"OpenVLA load failed: {exc}", 503) from exc

    def unload(self) -> None:
        self.loaded = False
        self.model = None
        self.processor = None
        if self._torch is not None:
            gc.collect()
            if self._torch.cuda.is_available():
                self._torch.cuda.empty_cache()

    def predict_action(self, request: RobotActionRequest) -> RobotActionResponse:
        if not self.loaded or self.model is None or self.processor is None or self._torch is None:
            raise ServiceError("model_not_ready", "OpenVLA runtime is not loaded", 503)
        prompt, image = self.adapter.preprocess(request)
        try:
            inputs = self.processor(prompt, image).to(self.device, dtype=self._dtype)
            with self._torch.inference_mode():
                return self.adapter.postprocess(self.model, inputs, request)
        except ServiceError:
            raise
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                self._torch.cuda.empty_cache()
                raise ServiceError(
                    "inference_oom", "OpenVLA inference exhausted GPU memory", 503
                ) from exc
            raise ServiceError("inference_failed", f"OpenVLA inference failed: {exc}", 503) from exc
