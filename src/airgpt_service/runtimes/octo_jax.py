from __future__ import annotations

import gc
import time
from pathlib import Path
from typing import Any

from ..adapters.vla.octo import OctoAdapter
from ..errors import ServiceError
from ..registry import RegistryModel
from ..schemas import RobotActionRequest, RobotActionResponse


class OctoJaxRuntime:
    """Lazy Octo Small runtime isolated in the legacy JAX/TF environment."""

    def __init__(self, spec: RegistryModel):
        self.spec = spec
        self.device = spec.device or "cuda:0"
        self.adapter = OctoAdapter(spec)
        self.model: Any | None = None
        self._jax: Any | None = None
        self._tf: Any | None = None
        self.loaded = False
        self.model_path: str | None = None

    def _model_path(self) -> str:
        configured = self.spec.metadata.get("local_path") or self.spec.model_id
        if not configured:
            raise ServiceError("model_load_failed", "Octo model path is required", 503)
        path = Path(str(configured))
        if not path.is_absolute():
            path = Path(__file__).resolve().parents[3] / path
        if not path.exists():
            raise ServiceError(
                "model_load_failed",
                f"Octo checkpoint is missing: {path}. Download it with `airgpt pull octo-small-1.5 --local-dir {path}`.",
                503,
            )
        return str(path)

    @staticmethod
    def _block_until_ready(jax: Any, tree: Any) -> Any:
        return jax.tree_map(
            lambda value: value.block_until_ready() if isinstance(value, jax.Array) else value,
            tree,
        )

    def load(self) -> None:
        if self.loaded:
            return
        try:
            import tensorflow as tf

            tf.config.set_visible_devices([], "GPU")
            import jax
            from octo.model.octo_model import OctoModel

            devices = jax.devices()
            if not any(device.platform == "gpu" for device in devices):
                raise ServiceError(
                    "runtime_unavailable",
                    f"Octo requires a JAX GPU backend; found {devices}",
                    503,
                )
            model_path = self._model_path()
            model = OctoModel.load_pretrained(model_path)
            self._block_until_ready(jax, model.params)
            self.model = model
            self._jax = jax
            self._tf = tf
            self.model_path = model_path
            self.loaded = True
        except ServiceError:
            self.unload()
            raise
        except RuntimeError as exc:
            self.unload()
            if "out of memory" in str(exc).lower() or "resource exhausted" in str(exc).lower():
                raise ServiceError("model_load_oom", "Octo load exhausted GPU memory", 503) from exc
            raise ServiceError("model_load_failed", f"Octo load failed: {exc}", 503) from exc
        except Exception as exc:
            self.unload()
            raise ServiceError("model_load_failed", f"Octo load failed: {exc}", 503) from exc

    def unload(self) -> None:
        self.loaded = False
        self.model = None
        self.model_path = None
        if self._jax is not None:
            try:
                self._jax.clear_caches()
            except Exception:
                pass
        self._jax = None
        self._tf = None
        gc.collect()

    def predict_action(self, request: RobotActionRequest) -> RobotActionResponse:
        if not self.loaded or self.model is None or self._jax is None:
            raise ServiceError("model_not_ready", "Octo runtime is not loaded", 503)
        try:
            observation, instruction, seed = self.adapter.preprocess(request)
            task = self.model.create_tasks(texts=[instruction])
            statistics = self.model.dataset_statistics[self.adapter.dataset]["action"]
            start = time.perf_counter()
            output = self.model.sample_actions(
                observation,
                task,
                unnormalization_statistics=statistics,
                rng=self._jax.random.PRNGKey(seed),
            )
            output = self._block_until_ready(self._jax, output)
            return self.adapter.postprocess(
                output,
                request,
                latency_seconds=time.perf_counter() - start,
            )
        except ServiceError:
            raise
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower() or "resource exhausted" in str(exc).lower():
                raise ServiceError(
                    "inference_oom", "Octo inference exhausted GPU memory", 503
                ) from exc
            raise ServiceError("inference_failed", f"Octo inference failed: {exc}", 503) from exc
        except Exception as exc:
            raise ServiceError("inference_failed", f"Octo inference failed: {exc}", 503) from exc
