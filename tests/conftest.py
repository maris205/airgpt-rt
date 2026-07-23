from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from airgpt_service import create_app
from airgpt_service.config import Settings


@pytest.fixture
def client(tmp_path: Path):
    root = Path(__file__).resolve().parents[1]
    settings = Settings(
        model_config_path=root / "configs" / "models.yaml",
        artifact_dir=tmp_path / "artifacts",
        worker_mode="mock",
        max_image_bytes=1024,
    )
    with TestClient(create_app(settings=settings)) as test_client:
        yield test_client
