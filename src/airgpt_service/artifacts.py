from __future__ import annotations

import shutil
import time
import uuid
from pathlib import Path


class ArtifactStore:
    """Bounded filesystem store for generated world-model outputs."""

    def __init__(self, root: Path, ttl_seconds: int = 86_400, max_runs: int = 100):
        self.root = root
        self.ttl_seconds = ttl_seconds
        self.max_runs = max_runs

    def prepare(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self.cleanup()

    def create_run(self, model: str) -> tuple[str, Path]:
        self.prepare()
        run_id = f"{int(time.time())}-{uuid.uuid4().hex[:10]}"
        run_dir = self.root / model / run_id
        run_dir.mkdir(parents=True, exist_ok=False)
        return run_id, run_dir

    def url(self, path: Path) -> str:
        relative = path.relative_to(self.root).as_posix()
        return f"/artifacts/{relative}"

    def cleanup(self) -> None:
        if not self.root.exists():
            return
        now = time.time()
        runs: list[Path] = []
        for model_dir in self.root.iterdir():
            if not model_dir.is_dir():
                continue
            for run_dir in model_dir.iterdir():
                if not run_dir.is_dir():
                    continue
                if self.ttl_seconds >= 0 and now - run_dir.stat().st_mtime > self.ttl_seconds:
                    shutil.rmtree(run_dir, ignore_errors=True)
                else:
                    runs.append(run_dir)
            if not any(model_dir.iterdir()):
                model_dir.rmdir()
        runs.sort(key=lambda path: path.stat().st_mtime, reverse=True)
        for run_dir in runs[self.max_runs :]:
            shutil.rmtree(run_dir, ignore_errors=True)
