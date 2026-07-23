from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "models.yaml"


@dataclass(frozen=True)
class Settings:
    model_config_path: Path = DEFAULT_CONFIG_PATH
    artifact_dir: Path = PROJECT_ROOT / "artifacts" / "runtime"
    worker_mode: str = "mock"
    log_level: str = "INFO"
    max_image_bytes: int = 10 * 1024 * 1024
    inference_timeout_seconds: float = 120.0
    max_concurrent_inference: int = 1

    @classmethod
    def from_env(cls) -> "Settings":
        config_path = Path(os.getenv("AIRGPT_MODEL_CONFIG", str(DEFAULT_CONFIG_PATH)))
        artifact_dir = Path(os.getenv("AIRGPT_ARTIFACT_DIR", str(cls.artifact_dir)))
        return cls(
            model_config_path=config_path,
            artifact_dir=artifact_dir,
            worker_mode=os.getenv("AIRGPT_WORKER_MODE", "mock"),
            log_level=os.getenv("AIRGPT_LOG_LEVEL", "INFO"),
            max_image_bytes=int(os.getenv("AIRGPT_MAX_IMAGE_BYTES", str(cls.max_image_bytes))),
            inference_timeout_seconds=float(
                os.getenv("AIRGPT_INFERENCE_TIMEOUT_SECONDS", str(cls.inference_timeout_seconds))
            ),
            max_concurrent_inference=int(
                os.getenv("AIRGPT_MAX_CONCURRENT_INFERENCE", str(cls.max_concurrent_inference))
            ),
        )
