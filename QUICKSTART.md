# QUICKSTART

Three stages. Do them in order; each one gates the next.

---

## Stage 0 — On your PC. No GPU, no cost, 2 minutes.

```bash
tar xzf agentic-bench.tar.gz && cd agentic-bench
pip install -r requirements.txt

python tools/verify_proxy.py       # proxy: byte-identity, usage capture, turn numbering
python tools/verify_export.py       # exporter: both engines, identical schemas
python tools/verify_branching.py    # sequence separation, fan-out, condensation
```

These run the whole pipeline against a mock server and a fake Prometheus. If they're green,
the proxy and the analysis are correct, and **every failure from here on is infrastructure** —
which is a much easier thing to debug at 2am with a GPU on the meter.

---

## Stage 1 — Decide where the drivers run

**Do not run them from your PC.** Two reasons:

1. `ttft_s` (measured by the proxy) is the primary latency metric — you join it against cache
   hits and context length. Over home broadband you'd be adding your ISP's round-trip *and its
   jitter* to every measurement. For short prompts, where TTFT is ~100ms, the WAN would be most
   of the number.
2. Agents resend their entire context every turn. A 50k-token prompt is ~200KB. Eight concurrent
   agents turning over every few seconds is tens of Mbps of *sustained upload* — more than most
   home connections have. You'd be measuring your router, not the serving stack.

So pick one:

| | Setup | Trade |
|---|---|---|
| **A. Everything on the GPU pod** *(recommended)* | Server, proxy, Prometheus, drivers all on one pod, all over `localhost` | Zero network in the measurement path. Drivers compete with vLLM for CPU (mostly harmless — vLLM is GPU-bound) |
| **B. GPU pod + CPU pod, same region** | Server on the GPU pod; proxy, Prometheus, drivers on a CPU pod | Clean isolation. Needs a RunPod TCP port exposed and the datacenter-internal hop |

**Not** your PC. Not "just for the first run".

### The Docker question — settle this before you provision

mini-swe-agent's SWE-bench runner puts each task in a **Docker container**, and most RunPod
templates can't do Docker-in-Docker without a privileged container. Options:

- Pick a RunPod template with **Docker/DinD enabled**. Cleanest; keeps SWE-bench semantics.
- Run mini-swe-agent with `--environment-class local`. No Docker, but the agent executes bash
  **directly on your pod**. On a disposable pod that's a fine trade — just make it knowingly.
- **Skip mini-swe-agent for now.** The OpenHands driver uses `git clone` + a local workspace and
  needs **no Docker at all**. The entire delegation/condensation matrix is available without
  solving this.

---

## Stage 2 — On the pod

Assuming path **A** (everything on the GPU pod). Four terminals, or `tmux`.

```bash
# --- once
git clone <your fork> && cd agentic-bench     # or scp the tarball up
pip install -r requirements.txt
export MODEL=Qwen/Qwen3-8B                    # must be IDENTICAL across both engines
```

**1. Serve.** One engine at a time — they want the same GPU.

```bash
MAX_LEN=65536 scripts/launch_server_vllm.sh
# check (phase 1 acceptance):
curl -s localhost:8000/metrics | grep vllm:prefix_cache
```

**2. Prometheus.** Everything is local now, so point it at localhost:

```bash
sed -i 's/GPU_POD_HOST/localhost/' prometheus/prometheus.yml
prometheus --config.file=prometheus/prometheus.yml &      # :9090
```

**3. Proxy.** Sits between the drivers and vLLM. `RUN_ID` keeps each run's log separate —
**use a fresh one per engine**, or you'll join requests against the wrong engine's metrics.

```bash
RUN_ID=vllm-baseline UPSTREAM_BASE_URL=http://localhost:8000 scripts/run_proxy.sh   # :9000
```

**4. One task first.** Do not start at concurrency 8.

```bash
scripts/run_openhands.sh --concurrency 1 --num-tasks 1
```

You are checking exactly one thing: does `logs/vllm-baseline/requests.jsonl` contain a run of
lines with the same `sequence_id`, an increasing `turn_number`, and a `prompt_tokens` that
climbs? If yes, the hard part is done.

**5. The real run.**

```bash
scripts/run_openhands.sh --concurrency 8 --num-tasks 8 &
scripts/run_chat.sh --num-conversations 32 --concurrency 8 &     # heterogeneous load
wait

python scripts/export_metrics.py --engine vllm \
  --requests logs/vllm-baseline/requests.jsonl --prometheus http://localhost:9090
```

**6. The first thing to check, before anything else:**

```python
import pandas as pd
df = pd.read_parquet("analysis/results_vllm.parquet")
df.eviction_events.max()
```

**If that's zero, the run told you nothing.** The KV pool never filled, nothing was preempted,
and every question about eviction, admission and retention is unanswerable — you measured an
idle server. The exporter prints a warning, but it's easy to scroll past and then build a
section on.

Fix it by raising `--concurrency`, or — cheaper and faster — shrink the KV pool:

```bash
GPU_UTIL=0.4 MAX_LEN=65536 scripts/launch_server_vllm.sh
```

Pool size is a legitimate design variable, not a constant. Shrinking it is a valid way to reach
the interesting regime without paying for more concurrency.

---

## Then

- Repeat everything with `scripts/launch_server_sglang.sh` (port 30000), `RUN_ID=sglang-baseline`,
  `--engine sglang`, **same model and same task set**. That's the cross-engine comparison.
- Run the scaffold matrix (README → *The scaffold matrix*): mini-swe-agent, OpenHands baseline,
  `--condenser`, `--delegation`.
- **Before sweeping the delegation arm, run `scripts/preflight_delegation.sh`.** If the model
  never calls `DelegateTool`, both arms are the same workload and you'd be sweeping noise. Two
  tasks to find out; a whole sweep if you don't.

## Ports

| Port | What |
|---|---|
| 8000 | vLLM |
| 30000 | SGLang |
| 9000 | proxy (drivers point here, **not** at the engine) |
| 9090 | Prometheus |

On setup **B**, expose the engine's port via RunPod TCP and set `UPSTREAM_BASE_URL` to the
public address; everything else stays on the driver box.
