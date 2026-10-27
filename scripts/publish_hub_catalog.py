from __future__ import annotations

import argparse
import os
from pathlib import Path

from airgpt_service.hub import HubCatalog


ROOT = Path(__file__).resolve().parents[1]
CATALOG_DIR = ROOT / "src" / "airgpt_service" / "hub_catalog"


def main() -> int:
    parser = argparse.ArgumentParser(description="publish the AirGPT Hub catalog")
    parser.add_argument("--repo-id", default="mariswang/AirGPT-Embodied-Model-Index")
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args()

    token = os.getenv("MODELSCOPE_API_TOKEN")
    if not token:
        parser.error("MODELSCOPE_API_TOKEN is required")

    HubCatalog.model_validate_json((CATALOG_DIR / "index.json").read_text(encoding="utf-8"))
    try:
        from modelscope_hub import HubApi
    except ImportError as exc:
        raise SystemExit("install modelscope-hub before publishing") from exc

    api = HubApi(token=token)
    if not api.repo_exists(args.repo_id, "dataset"):
        api.create_repo(
            repo_id=args.repo_id,
            repo_type="dataset",
            visibility="private" if args.private else "public",
            license="Apache License 2.0",
            description="Versioned model catalog for AirGPT Runtime",
        )
    result = api.upload_folder(
        repo_id=args.repo_id,
        repo_type="dataset",
        folder_path=CATALOG_DIR,
        commit_message="Publish AirGPT Hub catalog v1",
    )
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
