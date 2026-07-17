#!/usr/bin/env bash
# setup_venv.sh - rebuild the agent-bench venv on a fresh pod.
# A venv is NOT portable across containers, so a new pod needs this even though
# .venv sits on the persistent volume. Skips rebuild if the existing venv works.
# Usage:  bash setup_venv.sh   (or  REBUILD=1 bash setup_venv.sh  to force)
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

echo "=== [1/5] checking existing venv ==="
if [[ -x .venv/bin/python ]] && \
   .venv/bin/python -c "import fastapi, uvicorn, openhands.sdk, litellm, pandas" >/dev/null 2>&1; then
  echo "  existing .venv works -- skipping rebuild."
  echo "  (force a rebuild with:  REBUILD=1 bash setup_venv.sh)"
  if [[ "${REBUILD:-0}" != "1" ]]; then
    source .venv/bin/activate
    echo "venv ready. Activate with:  source .venv/bin/activate"
    exit 0
  fi
fi

echo "  rebuilding venv..."
rm -rf .venv
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip -q

echo "=== [2/5] core deps (proxy, exporter, drivers) ==="
pip install -q \
  fastapi "uvicorn[standard]" httpx pandas pyarrow requests orjson \
  mini-swe-agent datasets huggingface_hub PyYAML

echo "=== [3/5] OpenHands (no-deps, to dodge the lmnr/opentelemetry conflict) ==="
pip install -q openhands-sdk openhands-tools --no-deps

echo "=== [4/5] OpenHands real runtime deps (lmnr deliberately excluded) ==="
pip install -q \
  litellm pydantic tenacity fastmcp python-frontmatter \
  agent-client-protocol deprecation "fakeredis[lua]" pillow \
  python-json-logger tree-sitter tree-sitter-bash \
  binaryornot func-timeout libtmux "websockets>=12"

echo "=== [5/5] verifying imports ==="
fail=0
check(){ python -c "$1" >/dev/null 2>&1 && echo "  ok: $2" || { echo "  FAIL: $2"; fail=1; }; }
check "import fastapi, uvicorn, httpx, pandas, pyarrow" "proxy + exporter"
check "import minisweagent" "mini-swe-agent"
check "import datasets, huggingface_hub" "datasets (SWE-bench + ShareChat)"
check "import openhands.sdk; from openhands.tools.preset.default import get_default_tools" "openhands"
check "import litellm" "litellm"

if [[ $fail -eq 0 ]]; then
  echo ""
  echo "venv ready. Activate with:  source .venv/bin/activate"
else
  echo ""
  echo "Some imports failed above. If it names a missing module, pip install that one"
  echo "package (it won't re-trigger the lmnr conflict) and re-run the verify."
  exit 1
fi