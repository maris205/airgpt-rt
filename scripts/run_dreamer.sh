#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${project_root}"

repo_dir="${AIRGPT_DREAMER_REPO:-${project_root}/.cache/repos/dreamerv3}"
export PYTHONPATH="${repo_dir}:${PYTHONPATH:-}"
export AIRGPT_WORKER_MODE=real
export AIRGPT_PORT="${AIRGPT_PORT:-8002}"
export AIRGPT_ARTIFACT_DIR="${AIRGPT_ARTIFACT_DIR:-${project_root}/artifacts/runtime-dreamer}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

exec conda run --no-capture-output \
  -n "${AIRGPT_DREAMER_ENV:-airgpt-dreamer}" \
  python -m airgpt_service.main
