from __future__ import annotations

import gc
import time
from pathlib import Path
from typing import Any

from ..adapters.world.ltx_video import LTXVideoAdapter
from ..artifacts import ArtifactStore
from ..errors import ServiceError
from ..registry import RegistryModel
from ..schemas import WorldPredictRequest, WorldPredictResponse


class LTXVideoRuntime:
    """LTX-Video image-to-video runtime with CPU offload and bounded artifacts."""

    def __init__(self, spec: RegistryModel, artifacts: ArtifactStore):
        self.spec = spec
        self.artifacts = artifacts
        self.device = spec.device or "cuda:0"
        self.adapter = LTXVideoAdapter()
        self.pipe: Any | None = None
        self._torch: Any | None = None
        self.loaded = False

    def _model_ref(self) -> str:
        configured_path = self.spec.metadata.get("local_path") or self.spec.model_id
        if not configured_path:
            raise ServiceError("model_load_failed", "LTX-Video model path is required", 503)
        configured_path = str(configured_path)
        path = Path(configured_path)
        if not path.is_absolute():
            path = Path(__file__).resolve().parents[3] / path
        if not path.exists():
            raise ServiceError(
                "model_load_failed",
                f"LTX-Video checkpoint is missing: {path}. Download it with `airgpt pull ltx-video-0.9.5 --local-dir {path}`.",
                503,
            )
        return str(path)

    def load(self) -> None:
        if self.loaded:
            return
        try:
            import torch
            from diffusers import LTXImageToVideoPipeline

            self._torch = torch
            dtype_name = str(self.spec.metadata.get("dtype", "bfloat16"))
            dtype_map = {
                "float16": torch.float16,
                "bfloat16": torch.bfloat16,
                "float32": torch.float32,
            }
            if dtype_name not in dtype_map:
                raise ServiceError(
                    "model_load_failed",
                    f"unsupported LTX-Video dtype '{dtype_name}'",
                    503,
                )
            self.pipe = LTXImageToVideoPipeline.from_pretrained(
                self._model_ref(),
                torch_dtype=dtype_map[dtype_name],
                local_files_only=bool(self.spec.metadata.get("local_files_only", False)),
            )
            self.pipe.enable_model_cpu_offload()
            self.pipe.vae.enable_tiling()
            self.loaded = True
        except RuntimeError as exc:
            self.unload()
            if "out of memory" in str(exc).lower():
                raise ServiceError(
                    "model_load_oom", "LTX-Video load exhausted GPU memory", 503
                ) from exc
            raise ServiceError("model_load_failed", f"LTX-Video load failed: {exc}", 503) from exc
        except Exception as exc:
            self.unload()
            raise ServiceError("model_load_failed", f"LTX-Video load failed: {exc}", 503) from exc

    def unload(self) -> None:
        self.loaded = False
        self.pipe = None
        if self._torch is not None:
            gc.collect()
            if self._torch.cuda.is_available():
                self._torch.cuda.empty_cache()

    def predict_world(self, request: WorldPredictRequest) -> WorldPredictResponse:
        if not self.loaded or self.pipe is None or self._torch is None:
            raise ServiceError("model_not_ready", "LTX-Video runtime is not loaded", 503)
        try:
            from diffusers.utils import export_to_video

            image, prompt = self.adapter.preprocess(request)
            max_frames = int(self.spec.metadata.get("max_frames", 49))
            if request.num_frames > max_frames:
                raise ServiceError(
                    "invalid_input",
                    f"LTX-Video supports at most {max_frames} frames in this service",
                    422,
                )
            start = time.perf_counter()
            seed = self._torch.Generator(device=self.device).manual_seed(request.seed)
            with self._torch.inference_mode():
                output = self.pipe(
                    image=image,
                    prompt=prompt,
                    negative_prompt=request.negative_prompt,
                    height=request.height,
                    width=request.width,
                    num_frames=request.num_frames,
                    num_inference_steps=request.num_inference_steps,
                    guidance_scale=request.guidance_scale,
                    generator=seed,
                    max_sequence_length=128,
                ).frames[0]
            self._torch.cuda.synchronize() if self._torch.cuda.is_available() else None
            elapsed = time.perf_counter() - start
            run_id, run_dir = self.artifacts.create_run(request.model)
            frame_urls = []
            for index, frame in enumerate(output):
                frame_path = run_dir / f"frame-{index:03d}.png"
                frame.save(frame_path)
                frame_urls.append(self.artifacts.url(frame_path))
            video_path = run_dir / "future.mp4"
            export_to_video(output, str(video_path), fps=request.fps)
            return WorldPredictResponse(
                model=request.model,
                frames=frame_urls,
                metadata={
                    "runtime": "diffusers",
                    "artifact_id": run_id,
                    "video": self.artifacts.url(video_path),
                    "width": request.width,
                    "height": request.height,
                    "num_frames": len(output),
                    "fps": request.fps,
                    "seed": request.seed,
                    "num_inference_steps": request.num_inference_steps,
                    "guidance_scale": request.guidance_scale,
                    "latency_seconds": round(elapsed, 3),
                    "prompt": prompt,
                },
            )
        except ServiceError:
            raise
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                self._torch.cuda.empty_cache()
                raise ServiceError(
                    "inference_oom", "LTX-Video inference exhausted GPU memory", 503
                ) from exc
            raise ServiceError(
                "inference_failed", f"LTX-Video inference failed: {exc}", 503
            ) from exc
        except Exception as exc:
            raise ServiceError(
                "inference_failed", f"LTX-Video inference failed: {exc}", 503
            ) from exc
