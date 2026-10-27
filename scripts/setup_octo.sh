#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${project_root}"

env_name="${AIRGPT_OCTO_ENV:-airgpt-octo}"
repo_dir="${AIRGPT_OCTO_REPO:-${project_root}/.cache/repos/octo}"
model_dir="${AIRGPT_OCTO_MODEL_DIR:-${project_root}/.cache/models/octo-small-1.5}"
tokenizer_dir="${AIRGPT_OCTO_TOKENIZER_DIR:-${project_root}/.cache/models/t5-base}"
modelscope_cache="${MODELSCOPE_CACHE:-${project_root}/.cache/modelscope-octo}"
commit="241fb3514b7c40957a86d869fecb7c7fc353f540"
repo_url="https://github.com/octo-models/octo.git"
model_revision="d8dc05826aa4086e0b7e88b11bcef9bce75534ad"
tokenizer_revision="f0469cad324cc448c2ef576a030f357b496eea5a"

mkdir -p "$(dirname "${repo_dir}")"
if [[ ! -d "${repo_dir}/.git" ]]; then
  git clone "${repo_url}" "${repo_dir}"
fi
if ! git -C "${repo_dir}" cat-file -e "${commit}^{commit}" 2>/dev/null; then
  git -C "${repo_dir}" fetch --depth=1 origin "${commit}"
fi
if [[ "$(git -C "${repo_dir}" rev-parse HEAD 2>/dev/null || true)" != "${commit}" ]]; then
  git -C "${repo_dir}" checkout --detach "${commit}"
fi

if ! conda env list | awk '{print $1}' | grep -qx "${env_name}"; then
  conda create -y -n "${env_name}" python=3.10 pip
fi

conda run --no-capture-output -n "${env_name}" python -m pip install \
  -r "${project_root}/requirements/octo.txt"

# The Octo compatibility report uses the CUDA 11 JAX wheel on this legacy stack.
# Override AIRGPT_OCTO_JAX_SPEC when the host requires another tested wheel.
jax_spec="${AIRGPT_OCTO_JAX_SPEC:-jax[cuda11_pip]==0.4.20}"
conda run --no-capture-output -n "${env_name}" python -m pip install \
  "${jax_spec}" \
  -f "${AIRGPT_OCTO_JAX_RELEASES:-https://storage.googleapis.com/jax-releases/jax_cuda_releases.html}"
conda run --no-capture-output -n "${env_name}" python -m pip install -e "${project_root}"

if [[ ! -f "${model_dir}/config.json" ]]; then
  mkdir -p "${model_dir}" "${modelscope_cache}"
  MODELSCOPE_CACHE="${modelscope_cache}" conda run --no-capture-output -n "${env_name}" \
    python - "${model_dir}" "${model_revision}" <<'PY'
import sys

from modelscope_hub import HubApi

HubApi().download_repo(
    "mariswang/octo-small-1.5",
    "model",
    revision=sys.argv[2],
    local_dir=sys.argv[1],
)
PY
fi

if [[ ! -f "${tokenizer_dir}/config.json" ]]; then
  mkdir -p "${tokenizer_dir}" "${modelscope_cache}"
  MODELSCOPE_CACHE="${modelscope_cache}" conda run --no-capture-output -n "${env_name}" \
    python - "${tokenizer_dir}" "${tokenizer_revision}" <<'PY'
import sys

from modelscope_hub import HubApi

HubApi().download_repo(
    "AI-ModelScope/t5-base",
    "model",
    revision=sys.argv[2],
    local_dir=sys.argv[1],
)
PY
fi

conda run --no-capture-output -n "${env_name}" \
  python - "${model_dir}/config.json" "${tokenizer_dir}" <<'PY'
import json
import sys
from pathlib import Path

config_path = Path(sys.argv[1])
tokenizer_dir = sys.argv[2]
config = json.loads(config_path.read_text(encoding="utf-8"))
config["text_processor"]["kwargs"]["tokenizer_name"] = tokenizer_dir
config["model"]["task_tokenizers"]["language"]["kwargs"]["encoder"] = tokenizer_dir
config_path.write_text(json.dumps(config), encoding="utf-8")
PY

echo "Octo source: ${repo_dir}"
echo "Octo commit: $(git -C "${repo_dir}" rev-parse HEAD)"
echo "Octo checkpoint: ${model_dir}"
echo "T5 tokenizer: ${tokenizer_dir}"
echo "ModelScope cache: ${modelscope_cache}"
echo "Environment: ${env_name}"
