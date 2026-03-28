#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
RUNTIME_ROOT=$(cd -- "${SCRIPT_DIR}/../../.." && pwd)
ENV_FILE="${RUNTIME_ROOT}/deployment/runtime/.env"
PYTHON_BIN=${PYTHON_BIN:-python}

if [[ -f "${ENV_FILE}" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
  set +a
fi

: "${FLMM_WEB_HOST:=0.0.0.0}"
: "${FLMM_WEB_PORT:=9000}"

cd "${RUNTIME_ROOT}"
export PYTHONPATH="${RUNTIME_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
exec "${PYTHON_BIN}" -m uvicorn scripts.demo.web.backend.main:app \
  --host "${FLMM_WEB_HOST}" \
  --port "${FLMM_WEB_PORT}"
