#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${project_root}"

env_name="${AIRGPT_OCTO_ENV:-airgpt-octo}"
repo_dir="${AIRGPT_OCTO_REPO:-${project_root}/.cache/repos/octo}"
model_dir="${AIRGPT_OCTO_MODEL_DIR:-${project_root}/.cache/models/octo-small-1.5}"
hf_home="${HF_HOME:-${project_root}/.cache/huggingface-octo}"
commit="241fb3514b7c40957a86d869fecb7c7fc353f540"
repo_url="https://github.com/octo-models/octo.git"

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
  mkdir -p "${model_dir}" "${hf_home}"
  HF_HOME="${hf_home}" conda run --no-capture-output -n "${env_name}" \
    python - "${model_dir}" <<'PY'
import sys

from huggingface_hub import snapshot_download

snapshot_download(
    "rail-berkeley/octo-small-1.5",
    local_dir=sys.argv[1],
    local_dir_use_symlinks=False,
)
PY
fi

HF_HOME="${hf_home}" conda run --no-capture-output -n "${env_name}" python - <<'PY'
from transformers import AutoTokenizer

AutoTokenizer.from_pretrained("t5-base")
PY

echo "Octo source: ${repo_dir}"
echo "Octo commit: $(git -C "${repo_dir}" rev-parse HEAD)"
echo "Octo checkpoint: ${model_dir}"
echo "Hugging Face cache: ${hf_home}"
echo "Environment: ${env_name}"
