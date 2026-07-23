#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${project_root}"
base_url="${AIRGPT_BASE_URL:-http://127.0.0.1:8003}"
image_path="${1:-${project_root}/.cache/samples/bridge_im_12.jpg}"
repo_dir="${AIRGPT_OCTO_REPO:-${project_root}/.cache/repos/octo}"

export PYTHONPATH="${repo_dir}:${PYTHONPATH:-}"
conda run --no-capture-output -n "${AIRGPT_OCTO_ENV:-airgpt-octo}" \
  python - "${base_url}" "${image_path}" <<'PY'
import json
import sys
from pathlib import Path

from airgpt_sdk import Client

base_url, image_path = sys.argv[1:]
with Client(base_url, timeout=180) as client:
    result = client.robot.action(
        "octo",
        image=Path(image_path),
        instruction="pick up the fork",
        metadata={"seed": 0, "dataset": "bridge_dataset"},
    )

assert result["action_dim"] == 7
assert len(result["action"]) == 7
assert result["action_metadata"]["runtime"] == "octo_jax"
assert result["action_metadata"]["dataset"] == "bridge_dataset"
assert len(result["action_metadata"]["action_chunk"]) == 4
print(json.dumps(result, indent=2))
PY
