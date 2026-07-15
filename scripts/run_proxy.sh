#!/usr/bin/env bash
set -euo pipefail
export UPSTREAM_BASE_URL="${UPSTREAM_BASE_URL:?point at the engine currently running, e.g. http://10.0.0.5:8000 (vLLM) or http://10.0.0.5:30000 (SGLang)}"
# One log per run. The proxy APPENDS, so without a distinct path a second run silently
# concatenates onto the first -- and a log spanning two engines would join every request
# against whichever engine's series happened to be up. Set RUN_ID per run.
RUN_ID="${RUN_ID:-$(date +%Y%m%d-%H%M%S)}"
export PROXY_LOG_PATH="${PROXY_LOG_PATH:-logs/${RUN_ID}/requests.jsonl}"
echo "[proxy] run_id=${RUN_ID} log=${PROXY_LOG_PATH}"
export PROXY_INJECT_USAGE="${PROXY_INJECT_USAGE:-1}"
exec uvicorn proxy.main:app --host 0.0.0.0 --port "${PROXY_PORT:-9000}" --log-level warning
