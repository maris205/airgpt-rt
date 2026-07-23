#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${project_root}"

env_name="${AIRGPT_DREAMER_ENV:-airgpt-dreamer}"
repo_dir="${AIRGPT_DREAMER_REPO:-${project_root}/.cache/repos/dreamerv3}"
logdir="${AIRGPT_DREAMER_LOGDIR:-${project_root}/artifacts/dreamer/cartpole-debug}"
steps="${AIRGPT_DREAMER_STEPS:-5000}"

export PYTHONPATH="${repo_dir}:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

exec conda run --no-capture-output -n "${env_name}" python "${repo_dir}/dreamerv3/main.py" \
  --logdir "${logdir}" \
  --configs debug \
  --task gym_CartPole-v1 \
  --run.steps "${steps}" \
  --run.duration "${AIRGPT_DREAMER_DURATION:-120}" \
  --run.report_every 20 \
  --run.save_every 1 \
  --logger.outputs jsonl \
  --jax.platform cpu \
  --jax.prealloc False \
  --run.envs 1 \
  --run.debug True \
  --run.usage.nvsmi False \
  --run.usage.gputil False
