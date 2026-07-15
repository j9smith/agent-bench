#!/usr/bin/env bash
# SGLang on the GPU pod. Same model as launch_server_vllm.sh, for like-for-like.
#
# Unlike vLLM, SGLang's --enable-metrics flag IS real and IS required: without it
# /metrics is not served at all.
#
# RadixAttention (prefix caching) is on by default; there is no flag to add.
set -euo pipefail

MODEL="${MODEL:?set MODEL, e.g. Qwen/Qwen3-8B — must match the vLLM run}"
PORT="${PORT:-30000}"
MAX_LEN="${MAX_LEN:-65536}"
MEM_FRAC="${MEM_FRAC:-0.90}"   # SGLang's analogue of vLLM's --gpu-memory-utilization

exec python -m sglang.launch_server \
  --model-path "$MODEL" \
  --host 0.0.0.0 --port "$PORT" \
  --context-length "$MAX_LEN" \
  --mem-fraction-static "$MEM_FRAC" \
  --enable-metrics

# Phase-1 acceptance for this engine:
#   curl localhost:30000/metrics | grep -E 'cache_hit_rate'
# And settle the prefix question directly rather than trusting version docs:
#   curl localhost:30000/metrics | grep -o '^sglang[_:][a-z_]*' | sort -u
