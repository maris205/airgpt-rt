from __future__ import annotations

import tempfile
from pathlib import Path

from airgpt_service.hub import HubCatalog


ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = ROOT / "src" / "airgpt_service" / "hub_catalog" / "index.json"


def main() -> int:
    try:
        from modelscope_hub import HubApi
    except ImportError as exc:
        raise SystemExit("install modelscope-hub before auditing") from exc

    catalog = HubCatalog.model_validate_json(CATALOG_PATH.read_text(encoding="utf-8"))
    api = HubApi()
    failures = []
    print(f"{'MODEL':24} {'REVISION':12} {'AVAILABLE':9} LICENSE")
    for model in catalog.models:
        try:
            available = api.repo_exists(model.source.repo_id, "model")
            with tempfile.TemporaryDirectory(prefix="airgpt-modelscope-audit-") as cache_dir:
                api.download_repo(
                    model.source.repo_id,
                    "model",
                    revision=model.source.revision,
                    cache_dir=cache_dir,
                    allow_patterns=["README.md"],
                )
        except Exception as exc:
            failures.append(f"{model.slug}: {exc}")
            continue
        print(
            f"{model.slug:24} {model.source.revision[:12]:12} {str(available).lower():9} "
            f"{model.source.license}"
        )
        if not available:
            failures.append(f"{model.slug}: repository is not available")
    if failures:
        print("\nAudit failures:")
        for failure in failures:
            print(f"- {failure}")
        return 1
    print("\nCatalog audit passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
