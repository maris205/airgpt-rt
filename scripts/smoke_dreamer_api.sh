#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${project_root}"
base_url="${AIRGPT_BASE_URL:-http://127.0.0.1:8002}"
repo_dir="${AIRGPT_DREAMER_REPO:-${project_root}/.cache/repos/dreamerv3}"
export PYTHONPATH="${repo_dir}:${PYTHONPATH:-}"

conda run --no-capture-output -n "${AIRGPT_DREAMER_ENV:-airgpt-dreamer}" \
  python - "${base_url}" <<'PY'
import json
import sys

from airgpt_sdk import Client

base_url = sys.argv[1]
with Client(base_url, timeout=60) as client:
    result = client.state.predict(
        "dreamer_cartpole",
        observation=[0.01, 0.02, 0.03, 0.04],
        actions=[0, 1, 0, 1, 0],
        seed=0,
    )

assert len(result["latent_states"]) == 5
assert len(result["latent_states"][0]) == 16
assert len(result["predicted_observations"][0]) == 4
assert result["metadata"]["runtime"] == "dreamerv3"
print(json.dumps(result, indent=2))
PY
