#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${project_root}"

export AIRGPT_WORKER_MODE=real
export HF_HOME="${HF_HOME:-${project_root}/.cache/huggingface}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

exec conda run --no-capture-output \
  -n "${AIRGPT_OPENVLA_ENV:-airgpt-openvla}" \
  python -m airgpt_service.main
