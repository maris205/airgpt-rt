#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${project_root}"

env_name="${AIRGPT_DREAMER_ENV:-airgpt-dreamer}"
repo_dir="${AIRGPT_DREAMER_REPO:-${project_root}/.cache/repos/dreamerv3}"
commit="e3f02248693a79dc8b0ebd62c93683888ddaccfe"
repo_url="https://github.com/danijar/dreamerv3.git"

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

patch_file="${project_root}/patches/dreamerv3-gym026.patch"
if git -C "${repo_dir}" apply --reverse --check "${patch_file}" 2>/dev/null; then
  echo "Dreamer Gym 0.26 patch already applied"
elif git -C "${repo_dir}" apply --check "${patch_file}"; then
  git -C "${repo_dir}" apply "${patch_file}"
else
  echo "Dreamer source does not match the pinned patch context" >&2
  exit 1
fi

if ! conda env list | awk '{print $1}' | grep -qx "${env_name}"; then
  conda create -y -n "${env_name}" python=3.11 pip
fi

conda run --no-capture-output -n "${env_name}" python -m pip install \
  -r "${project_root}/requirements/dreamer.txt"
conda run --no-capture-output -n "${env_name}" python -m pip install -e "${project_root}"

echo "DreamerV3 source: ${repo_dir}"
echo "DreamerV3 commit: $(git -C "${repo_dir}" rev-parse HEAD)"
echo "Environment: ${env_name}"
