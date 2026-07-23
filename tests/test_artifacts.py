import os
import time
from pathlib import Path

from airgpt_service.artifacts import ArtifactStore


def test_artifact_store_bounds_runs_and_generates_urls(tmp_path: Path):
    store = ArtifactStore(tmp_path, ttl_seconds=3600, max_runs=2)
    created = [store.create_run("ltx_video") for _ in range(3)]
    for _, path in created:
        (path / "frame.png").write_bytes(b"frame")
    store.cleanup()
    remaining = list((tmp_path / "ltx_video").iterdir())
    assert len(remaining) == 2
    assert store.url(remaining[0] / "frame.png").startswith("/artifacts/ltx_video/")


def test_artifact_store_removes_expired_runs(tmp_path: Path):
    store = ArtifactStore(tmp_path, ttl_seconds=1, max_runs=10)
    _, path = store.create_run("ltx_video")
    old = time.time() - 10
    os.utime(path, (old, old))
    store.cleanup()
    assert not path.exists()
