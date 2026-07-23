#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
base_url="${AIRGPT_BASE_URL:-http://127.0.0.1:8000}"
image_path="${1:-${project_root}/artifacts/m0/openvla/synthetic_tabletop.png}"

python - "${base_url}" "${image_path}" <<'PY'
import json
import sys
import time
from pathlib import Path

from airgpt_sdk import Client

base_url, image_path = sys.argv[1:]
with Client(base_url, timeout=180) as client:
    start = time.perf_counter()
    result = client.robot.action(
        "openvla",
        Path(image_path),
        "pick up the red cube",
    )
    elapsed = time.perf_counter() - start

print(json.dumps({"elapsed_seconds": round(elapsed, 3), "result": result}, indent=2))
PY
