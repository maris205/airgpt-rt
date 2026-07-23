#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
base_url="${AIRGPT_BASE_URL:-http://127.0.0.1:8001}"
image_path="${1:-${project_root}/.cache/samples/bridge_im_12.jpg}"

python - "${base_url}" "${image_path}" <<'PY'
import json
import sys
import time
from pathlib import Path

from airgpt_sdk import Client

base_url, image_path = sys.argv[1:]
with Client(base_url, timeout=240) as client:
    start = time.perf_counter()
    result = client.world.predict(
        "ltx_video",
        image=Path(image_path),
        scene="A robot arm at a kitchen counter",
        action="The gripper moves toward the fork and lifts it slightly",
        num_frames=9,
        width=384,
        height=256,
        num_inference_steps=8,
        seed=0,
    )

print(json.dumps({"elapsed_seconds": round(time.perf_counter() - start, 3), "result": result}, indent=2))
PY
