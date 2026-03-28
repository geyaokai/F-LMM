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

: "${FLMM_WEB_HOST:=127.0.0.1}"
: "${FLMM_WEB_PORT:=9000}"
SMOKE_HOST=${FLMM_WEB_PUBLIC_HOST:-${FLMM_WEB_HOST}}
if [[ "${SMOKE_HOST}" == "0.0.0.0" ]]; then
  SMOKE_HOST="127.0.0.1"
fi

cd "${RUNTIME_ROOT}"
exec "${PYTHON_BIN}" deployment/runtime/smoke_test.py \
  --base-url "http://${SMOKE_HOST}:${FLMM_WEB_PORT}" \
  "$@"
