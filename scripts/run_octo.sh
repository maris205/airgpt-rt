#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${project_root}"

export PYTHONPATH="${AIRGPT_OCTO_REPO:-${project_root}/.cache/repos/octo}:${PYTHONPATH:-}"
export AIRGPT_WORKER_MODE=real
export AIRGPT_PORT="${AIRGPT_PORT:-8003}"
export AIRGPT_ARTIFACT_DIR="${AIRGPT_ARTIFACT_DIR:-${project_root}/artifacts/runtime-octo}"
export MODELSCOPE_CACHE="${MODELSCOPE_CACHE:-${project_root}/.cache/modelscope-octo}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-2}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

exec conda run --no-capture-output \
  -n "${AIRGPT_OCTO_ENV:-airgpt-octo}" \
  python -m airgpt_service.main
