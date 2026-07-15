#!/usr/bin/env bash
# vLLM on the GPU pod. Run this ON the pod. Same MODEL as launch_server_sglang.sh.
#
# Two corrections to the PRD, which anticipated them ("verify against the
# installed version, flag names have changed"):
#
#   * There is no --enable-metrics. /metrics is on by DEFAULT; it is switched
#     *off* by --disable-log-stats. So the correct action is to not pass that.
#   * Prefix caching is on by default in the V1 engine. --enable-prefix-caching
#     is a no-op on current builds and was required only on older ones. Passed
#     below anyway: harmless, and explicit is better for a measurement rig.
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen3-8B}"
PORT="${PORT:-8000}"
MAX_LEN="${MAX_LEN:-65536}"            # context length is a design variable here,
                                       # not a constant -- vary it across runs
GPU_UTIL="${GPU_UTIL:-0.90}"           # smaller KV pool => KV pressure at lower
                                       # concurrency; this is a knob, use it

exec vllm serve "$MODEL" \
  --host 0.0.0.0 --port "$PORT" \
  --max-model-len "$MAX_LEN" \
  --gpu-memory-utilization "$GPU_UTIL" \
  --enable-prefix-caching \
  --served-model-name "$MODEL"
  # deliberately NOT passing --disable-log-stats: that would kill /metrics
