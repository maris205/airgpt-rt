#!/usr/bin/env bash
set -euo pipefail

base_url="${AIRGPT_BASE_URL:-http://127.0.0.1:8000}"
curl --fail --silent "${base_url}/health/ready"
printf '\n'
curl --fail --silent "${base_url}/v1/models"
printf '\n'

