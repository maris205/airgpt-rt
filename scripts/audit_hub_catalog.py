from __future__ import annotations

import os
from pathlib import Path

from airgpt_service.hub import HubCatalog


ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = ROOT / "src" / "airgpt_service" / "hub_catalog" / "index.json"


def main() -> int:
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise SystemExit("install huggingface_hub before auditing") from exc

    catalog = HubCatalog.model_validate_json(CATALOG_PATH.read_text(encoding="utf-8"))
    api = HfApi(token=os.getenv("HF_TOKEN") or None)
    failures = []
    print(f"{'MODEL':24} {'REVISION':12} {'GATED':8} LICENSE")
    for model in catalog.models:
        try:
            info = api.model_info(model.source.repo_id, revision=model.source.revision)
        except Exception as exc:
            failures.append(f"{model.slug}: {exc}")
            continue
        card_data = getattr(info, "card_data", None)
        live_license = getattr(card_data, "license", None) or "unknown"
        live_gated = bool(getattr(info, "gated", False))
        print(f"{model.slug:24} {str(info.sha)[:12]:12} {str(live_gated).lower():8} {live_license}")
        if info.sha != model.source.revision:
            failures.append(f"{model.slug}: pinned revision resolved to {info.sha}")
        if live_gated != model.source.gated:
            failures.append(f"{model.slug}: gated state changed to {live_gated}")
        if live_license != model.source.license:
            failures.append(f"{model.slug}: license changed to {live_license}")
    if failures:
        print("\nAudit failures:")
        for failure in failures:
            print(f"- {failure}")
        return 1
    print("\nCatalog audit passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
