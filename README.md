# agentic-bench

**New here? Read [QUICKSTART.md](QUICKSTART.md) first.** It covers setup, where to run the drivers (not your PC — see below), and the first-run checks.

Measurement harness for *The Other Inference*. Drives real agentic (SWE-bench Lite via
mini-swe-agent) and real chat (ShareGPT replay) load against a self-hosted vLLM endpoint,
captures sender-side timing through a transparent proxy and server-side scheduler/cache
state through Prometheus, and joins the two into one row per request.

Both engines, vLLM and SGLang, are first-class. The proxy and drivers don't know which is
behind them; the only engine-specific surface is server launch and metric naming.

Workload **shape** is the metric. Task success rate is not.

---

## Read this first: deviations and one open blocker

### Branching: resolved, and the finding is the point

The original PRD pinned mini-swe-agent, whose history is strictly linear — so branching
was unobservable. That's now handled, and what turned up along the way matters more than the
metric did.

**Sub-agent delegation is not a fork of the parent's context.** Every mainstream
implementation (OpenHands, Codex CLI, Claude Code, OpenCode) hands the sub-agent a *fresh*
context — isolation is the entire reason to delegate, since the parent delegates precisely so
it doesn't have to carry the sub-agent's tool spam. So a fan-out of N sub-agents produces N
independent sequences sharing only the system prompt: the same shallow prefix any two
unrelated requests share.

This is not an exception to the memory wall. It's its worst case:

- one logical task becomes N sequences, each carrying a full context, all resident at once;
- the shared prefix is a **constant** (the system prompt) while each sub-agent's own context
  grows without bound — so whatever amortisation the fan-out offers **decays toward zero**;
- the parent's KV sits idle across the whole delegation, with resumption probability ≈ 1.

The harness measures all three rather than taking the literature's word for them.
`shared_prefix_msgs` is the depth, in messages, that a sub-agent shares with its parent —
run it and see whether it's really 1. Note that the *fraction* is a trap: it starts high
(a big system prompt is most of a young sub-agent's context) and decays. Quote the depth and
the absolute bytes, not the fraction, or you're quoting a function of when you happened to look.

**The three structures, which are not one phenomenon:**

| | Where | KV structure |
|---|---|---|
| Linear | mini-swe-agent, SWE-agent — most of the population | one sequence, monotonic growth |
| Delegation fan-out | OpenHands, Codex CLI, Claude Code | N sequences, **shallow** shared prefix, isolated by design |
| Tree search / backtrack | moatless-tree-search, SWE-Search | **deep** shared prefix, re-expansion from earlier states |

Only the third is a genuine radix DAG, and it's a research artifact rather than what people
run. Not implemented here; the hooks would work unchanged if you ever want it.

### Condensation: the scaffold doing eviction in userspace

OpenHands' `LLMSummarizingCondenser` (the SDK default, **off** by default in this harness)
drops old events and replaces them with an LLM-written summary once history gets long. So
OpenHands context does **not** grow monotonically — it sawtooths.

Which means: if OpenHands were your only scaffold, you'd measure that sawtooth and report it
as *what agentic context growth looks like*. It isn't. It's what one scaffold's **mitigation**
looks like. **This is why mini-swe-agent stays** — no condenser, strictly linear, so it's the
only source of the *unmanaged* growth curve, which is the true shape of the workload and the
input section 1's arithmetic needs.

And there's a result sitting in it. OpenHands' own benchmark of the condenser against a
no-condensation baseline on SWE-bench Verified: 200 instances resolved vs 203, **$40 more
expensive — attributed to lower prompt cache utilisation** — while flattening latency from
12–16s to a steady 8s. Condensation rewrites the prefix, so the server's cache misses. You
pay in rebuilds what you save in context length. The scaffold and the server are both managing
the same KV, neither knows what the other is doing, and they fight.

It's directly visible in the output: at a condensation event `prompt_tokens` drops
(`is_condensation`, `context_shrink_tokens`), `prefix_seen_before` flips false, and TTFT spikes
on the rebuild.

### Corrections to the PRD's stated stack### Corrections to the PRD's stated stack (each one verified against current docs)

- **`--enable-metrics` does not exist *for vLLM*.** `/metrics` is on by default there and is
  switched *off* by `--disable-log-stats`, so `launch_server_vllm.sh` just doesn't pass that
  flag. Prefix caching is likewise default-on in V1. **For SGLang the flag is real and is
  required** — without it, `/metrics` isn't served at all. RadixAttention is default-on.
- **Counter metrics get a `_total` suffix on the wire** (`vllm:prefix_cache_queries_total`),
  even though docs write them without. `export_metrics.py` probes both spellings rather than
  hardcoding, so it works either way and will work across version bumps.
- **`vllm:gpu_prefix_cache_hit_rate` is gone** in V1, replaced by the queries/hits counters.
  Both are *token*-granularity, not request-granularity: queries increments by prompt tokens,
  hits by tokens found cached.
- **`sglang:num_retractions` is a Histogram, not a counter** — "distribution of retraction
  counts per request". The extension PRD maps it straight onto `vllm:num_preemptions_total`;
  done literally, that query returns nothing. The exporter resolves `_sum` (total retraction
  events), which is the closest equivalent. Not an exact one: vLLM counts *preemption events*,
  SGLang's histogram observes *retractions per request*, so the sum is total retractions rather
  than total retraction episodes. Close enough to answer "did the pool ever fill", not close
  enough to quote a cross-engine ratio to two significant figures.
- **SGLang's metric prefix is `sglang_` on v0.5.4+ and `sglang:` before it.** Both are probed
  at runtime, so neither version needs to be pinned. Settle it directly on the box if curious:
  `curl <pod>:30000/metrics | grep -o '^sglang[_:][a-z_]*' | sort -u`.
- **Metric names contain colons, which are reserved for recording rules in PromQL.** Every
  query goes through `sum({__name__="..."})` rather than `sum(vllm:foo)`.

### Design decisions the PRD left to the build

- **One `mini-extra swebench` process per task, not `--workers N`.** The batch runner has its
  own pool, but mini-swe-agent passes `model.model_kwargs` straight to `litellm.completion` and
  those kwargs are fixed for the process lifetime — so `extra_headers` would stamp *every* task
  in a batch with the same `X-Session-Id`, destroying per-session grouping. The driver runs one
  filtered process per instance and pools them itself. `--concurrency N` semantics unchanged;
  the scaffold is unmodified (we only write config files and set `api_base`).
- **Session identity is carried twice.** Via `extra_headers` (the PRD's design) *and* via the
  `api_base` path (`/sess/{type}/{id}/v1`). The proxy prefers the header. This is insurance
  against litellm's provider shims dropping headers, which would silently null out `session_id`
  for a whole run — the kind of failure you'd only notice at analysis time.
- **Usage capture on streams.** Streaming responses only carry `usage` if the request sets
  `stream_options.include_usage`, which scaffolds don't. The proxy injects it and then
  **suppresses the resulting usage-only chunk on the way back out**, so the client sees a
  byte-identical stream. `PROXY_INJECT_USAGE=0` turns both halves off.
- **Two cache-hit columns, on purpose.** The extension PRD's mapping table puts vLLM's
  `prefix_cache_hits / prefix_cache_queries` ratio and SGLang's `cache_hit_rate` gauge in the
  same logical slot. They are not the same measurement — vLLM's is token-granularity counters we
  diff ourselves; SGLang's is a number the engine computes about itself. Stacking them would make
  the cross-engine comparison, which is the entire point of the new acceptance criterion, partly
  an artifact of two different definitions. So:
    - `prefix_cache_hit_rate` — token-granularity, derived from counter deltas **the same way for
      both engines** (vLLM: hits/queries; SGLang: `cached_tokens_total`/`prompt_tokens_total`).
      **This is the column to compare across engines.**
    - `prefix_cache_hit_rate_reported` — whatever the engine says about itself. Sanity check only.
      vLLM V1 doesn't expose one, so it's NaN there.
  Both are scrape-bucket rates, not per-request facts — the counters are sampled at 15s, so a
  per-request hit rate isn't a thing that exists. The signals that *are* exact per request are
  `prefix_seen_before` (from our own hashes, no Prometheus involved, therefore identical across
  engines by construction — which is what makes it a fair denominator) and `cached_tokens`.
- **The server-side schema is pinned, not discovered.** Columns with no backing series come out
  as NaN rather than missing, so `results_vllm` and `results_sglang` concatenate cleanly. This
  is not tidiness — it's the thing that broke first when I tested it, precisely because of the
  vLLM-has-no-reported-gauge asymmetry above.

---

## Acceptance status

| Phase | Status |
|---|---|
| 1. Server: vLLM `/metrics` | **Not run** — needs the RunPod pod |
| 1b. Server: SGLang `/metrics`, same model | **Not run** — needs the pod |
| 2. Proxy: byte-identity + well-formed log | **Passing** — `tools/verify_proxy.py` (engine-agnostic, runs once) |
| 3. Agentic driver, single task | **Not run** — needs GPU + Docker (engine-agnostic, runs once) |
| 4. Agentic driver, concurrent | **Not run** (engine-agnostic) |
| 5. Chat driver | **Not run** (engine-agnostic) |
| 6. Concurrent heterogeneous run | **Not run** — must be run **once per engine** |
| 7. Export/join, both engines, identical schema | **Passing** — `tools/verify_export.py` |

Phases 2 and 7 are verified against a mock server and a fake Prometheus that serves **both**
engine shapes — including the SGLang retraction histogram and the prefix drift — so the proxy
and the join are known-good before you spend a GPU-hour. Phases 1/1b and 3–6 are gated on
hardware I don't have here; run them in order and don't skip ahead.

```bash
python tools/verify_proxy.py      # byte-identity, usage capture, turn numbering, null-safety
python tools/verify_export.py     # both engines, --engine auto, identical schemas
python tools/verify_branching.py  # sequence separation, fan-out, condensation detection
```

`verify_branching.py` simulates what OpenHands actually puts on the wire: **one** process,
**one** set of static headers, containing a parent agent, two isolated sub-agents, and the
condenser's summariser — and checks the proxy tells all four apart anyway, and that the
exporter reconstructs the tree from hashes alone.

---

## What gets kept

Three artifacts per run, in descending order of how much you'd regret losing them.

| Artifact | What it is |
|---|---|
| `logs/<run-id>/requests.jsonl` | **Raw sender-side.** One line per request, written as it happens. Irreplaceable. |
| `analysis/raw_series_<engine>.parquet` | **Raw server-side.** *Every* series the engine exposed during the run window, long-format, **labels intact** — including histogram buckets, `phase="prefill"/"decode"`, per-rank labels, and every series the mapping never asked for. |
| `analysis/results_<engine>.parquet` | The joined table. A **lossy reduction** of the two above. |
| `analysis/manifest_<engine>.json` | Provenance: engine, model, run window, which Prometheus series each logical column actually resolved to. |

The joined table is a *view*, chosen in advance, and it will be wrong about something — a
series you didn't think to want, a histogram you ignored, a label you summed away. So the
exporter archives everything **before** it reduces. If the reduction turns out to be the wrong
one you can redo it offline, without a GPU. `--no-raw` turns this off; you don't want that.

Two things that are *not* automatically conserved, so mind them:

- **Prometheus' own TSDB** is the ultimate source and has a default 15d retention. The raw
  archive above covers only the run window. Either run the exporter promptly or set
  `--storage.tsdb.retention.time` generously.
- **`logs/requests.jsonl` is append-mode.** `run_proxy.sh` now writes to `logs/<RUN_ID>/` by
  default so runs don't concatenate. **Use a fresh `RUN_ID` per engine** — a single log spanning
  both engines would join every request against whichever engine's series happened to be up.

## Setup

```bash
pip install -r requirements.txt
```

Proxy and both drivers must run **co-located with the GPU pod** (same pod, or a CPU pod in the
same region). Not across your home connection. This is a hard requirement.

## Run

**1. Server** (on the GPU pod) — one engine at a time; they want the same GPU. Same `MODEL`
for both, or the comparison means nothing:
```bash
MODEL=Qwen/Qwen3-8B MAX_LEN=65536 scripts/launch_server_vllm.sh
curl localhost:8000/metrics | grep vllm:prefix_cache        # phase 1 acceptance

MODEL=Qwen/Qwen3-8B MAX_LEN=65536 scripts/launch_server_sglang.sh
curl localhost:30000/metrics | grep cache_hit_rate          # phase 1b acceptance
```

**2. Prometheus** (on the driver host): edit `prometheus/prometheus.yml`, replace
`GPU_POD_HOST`, then `prometheus --config.file=prometheus/prometheus.yml`. It scrapes both jobs;
whichever engine is down just fails its scrapes, which is fine and is exactly why the exporter
probes for series that have data rather than trusting a config.

**3. Proxy**:
```bash
UPSTREAM_BASE_URL=http://<pod>:8000 scripts/run_proxy.sh
```

**4. Workload**:
```bash
scripts/run_agentic.sh --concurrency 8 --num-tasks 8 --model Qwen/Qwen3-8B
python drivers/chat_replay/fetch_sharegpt.py          # once
scripts/run_chat.sh --num-conversations 32 --concurrency 8
```
Run both at once for the heterogeneous case (phase 6).

**5. Export**, once per engine, pointing at that engine's run log:
```bash
python scripts/export_metrics.py --engine vllm \
  --requests logs/vllm-run/requests.jsonl --prometheus http://localhost:9090 --step 15
python scripts/export_metrics.py --engine sglang \
  --requests logs/sglang-run/requests.jsonl --prometheus http://localhost:9090 --step 15
```
`--engine auto` (the default) works out which engine served the run by probing Prometheus, so
you can drop the flag if only one engine's series exist for that window. Give each engine its
own `--requests` log (or move `logs/requests.jsonl` aside between runs) — one log spanning both
engines would join every request against whichever series happened to be up.

The new acceptance criterion — two tables, same schema, same SWE-bench Lite task set, same
concurrency — is exactly what `pd.concat([results_vllm, results_sglang])` should do without
complaint. `tools/verify_export.py` asserts the schema half of that.

## Knobs that matter

Concurrency 8 is a starting point, not a claim that it produces KV pressure. **Watch
`vllm:num_preemptions_total`** — `export_metrics.py` prints a warning if it never leaves zero,
which means the pool never filled and the eviction/admission questions are unanswerable from
that run. Raise `--concurrency`, or shrink the pool with `GPU_UTIL`, or raise `MAX_LEN`.
Context length is a design variable here; vary it across runs rather than fixing it at one value.

## Output columns

Per request: `session_id`, `session_type`, `turn_number`, `prompt_tokens`, `completion_tokens`,
`cached_tokens`, `prefix_hash`, `full_hash`, `ttft_s`, `itl_s`, `e2e_s`.

Derived session shape: `gap_s` (tool-exec time between turns), `context_growth_tokens`,
`is_resumption`, `prefix_seen_before`, `exact_resend`, `branch_factor`.

Server-side at request time: `vllm:kv_cache_usage_perc`, `vllm:num_requests_running`,
`vllm:num_requests_waiting`, `prefix_hit_rate_window`, `preemptions_window`.

Column names are the *logical* ones and are identical across engines, so downstream analysis
never branches on `engine` — it's there as a column to group by, not to `if` on.

---

## The actual runbook

Once, on the CPU pod (same region as the GPU pod — not your laptop):

```bash
pip install -r requirements.txt
python drivers/chat_replay/fetch_sharegpt.py
sed -i 's/GPU_POD_HOST/10.0.0.5/' prometheus/prometheus.yml     # your pod's address
prometheus --config.file=prometheus/prometheus.yml &            # :9090
```

### One engine's run, start to finish

Say vLLM. Four terminals; `MODEL` must be identical across both engines or the comparison is
meaningless.

```bash
# [pod]  1. serve
MODEL=Qwen/Qwen3-8B MAX_LEN=65536 scripts/launch_server_vllm.sh
curl localhost:8000/metrics | grep vllm:prefix_cache          # phase 1: must be non-empty

# [cpu]  2. proxy, tagged with a run id
RUN_ID=vllm-run UPSTREAM_BASE_URL=http://10.0.0.5:8000 scripts/run_proxy.sh

# [cpu]  3. workload — both drivers at once is the heterogeneous case (phase 6)
MODEL=Qwen/Qwen3-8B scripts/run_agentic.sh --concurrency 8 --num-tasks 8 &
MODEL=Qwen/Qwen3-8B scripts/run_chat.sh --num-conversations 32 --concurrency 8 &
wait

# [cpu]  4. archive + reduce, while Prometheus still has the window
python scripts/export_metrics.py --engine vllm \
  --requests logs/vllm-run/requests.jsonl --prometheus http://localhost:9090
```

Then kill vLLM, bring up SGLang on the same GPU with the same `MODEL`, and repeat with
`RUN_ID=sglang-run`, `UPSTREAM_BASE_URL=http://10.0.0.5:30000`, `--engine sglang`, and the
**same task set and concurrency**. That's the new acceptance criterion.

### Did the run tell you anything?

```python
import pandas as pd
v = pd.read_parquet("analysis/results_vllm.parquet")
s = pd.read_parquet("analysis/results_sglang.parquet")
df = pd.concat([v, s])          # schemas are pinned; this just works

# The check that decides whether the run is usable at all:
df.groupby("engine").eviction_events.max()
```

**If that is zero, stop.** The KV pool never filled, nothing preempted, and every question about
eviction, admission and retention is unanswerable from this data — you measured an idle server.
The exporter prints a warning to this effect. Fix by raising `--concurrency`, shrinking the pool
(`GPU_UTIL` / `MEM_FRAC`), or raising `MAX_LEN`. Concurrency 8 is a starting point, not a claim.

Assuming it's non-zero, the numbers section 0 of the post wants:

```python
a = df[df.session_type == "agentic"]
a.prompt_tokens.describe()                              # context length distribution
a.groupby("turn_number").prompt_tokens.median()         # how context grows per turn
a.gap_s.describe()                                      # gap durations = tool exec time
a.is_resumption.mean()                                  # resumption rate
a.branch_factor.mean()                                  # ~1.0 — see the blocker above

# chat vs agentic, the whole point:
df.groupby("session_type")[["prompt_tokens", "gap_s", "ttft_s"]].median()

# does prefix reuse actually buy TTFT?
df.groupby(["engine", "prefix_seen_before"]).ttft_s.median()

# cross-engine, the comparable column (not the engine's self-report):
df.groupby("engine")[["prefix_cache_hit_rate", "ttft_s"]].median()
```

TTFT distributions come from the **proxy**, per request (`ttft_s`), not from the server's
histograms — per-request is strictly better, since it's joinable to that request's context
length and cache state. The server histograms are in the raw archive if you want them anyway.

### If something didn't work

- **`prefix_cache_hit_rate` is all NaN** → the counters didn't move. Prefix caching off, or the
  scrape window missed the run. Check `manifest_<engine>.json` for what actually resolved.
- **`session_id` is null everywhere** → litellm dropped the headers. The path fallback should
  have caught it; check `session_id_source` in the log to see which route won.
- **`joined server metrics to 0% of requests`** → clock skew between the proxy host and the
  Prometheus host, or `--step` doesn't match `scrape_interval`.

---

## The scaffold matrix

Three configurations, of which the last three are a controlled A/B — same scaffold, same
prompts, same tasks, one variable at a time.

```bash
# 1. Linear baseline: unmanaged, monotonic context growth. The ground truth shape.
scripts/run_agentic.sh --concurrency 8 --num-tasks 8

# 2. OpenHands, both toggles off. Cross-check: should agree with (1) on the baseline.
#    If it doesn't, one of the two scaffolds is doing something you haven't noticed.
scripts/run_openhands.sh --concurrency 8 --num-tasks 8

# 3. Condensation on: userspace eviction vs. the server's prefix cache.
scripts/run_openhands.sh --concurrency 8 --num-tasks 8 --condenser

# 4. Delegation on: fan-out, shallow shared prefix, idle parent.
scripts/run_openhands.sh --concurrency 8 --num-tasks 8 --delegation
```

**Run the pre-flight before committing to any sweep:**

```bash
scripts/preflight_delegation.sh
```

The likeliest way this experiment fails isn't a bug — it's the model never calling
`DelegateTool`. If it doesn't, the "on" and "off" arms are the same workload and you'd be
sweeping noise. The driver prints a warning; the pre-flight makes it cost two tasks instead of
a sweep. Same for the condenser: if tasks are short it never fires, and you lower
`--condenser-max-size`.

### Two things that will bite you

**Driver concurrency stops being a valid x-axis the moment `--delegation` is on.** Eight tasks
with a fan-out of four is up to *forty* live sequences. The two arms are not comparable at
matched `--concurrency`. This is why the sweep must be plotted against a **server-side** axis
(`kv_cache_utilization` or `running_requests`), not against the driver's concurrency flag.
Then the fan-out shows up as *"reaches the same KV pressure at a fifth of the task count"* —
which is the actual finding.

**`--condenser` off is not "no context management".** OpenHands' controller still truncates
history as a fallback when a request exceeds the context window. Set `MAX_LEN` high enough that
tasks don't reach it, and check the data for the signature (a sudden halving of `prompt_tokens`,
which will show up as `is_condensation`) rather than assuming it never fired.

### Extra columns this produces

`task_id` (the logical unit of work) · `sequence_id` (one conversation — derived by the proxy
from the conversation's root, because OpenHands runs parent, sub-agents and summariser in one
process under identical static headers) · `sequence_role` (root / subagent / oneshot) ·
`fanout` · `shared_prefix_msgs` / `shared_prefix_chars` / `shared_prefix_frac` ·
`is_condensation` / `context_shrink_tokens`.

`sequence_role == "oneshot"` is a heuristic for the condenser's summariser (a conversation that
never gets a second turn). Cross-check it against the driver's own `Condensation` event count in
`logs/openhands/summary.json` — the driver counts them from OpenHands' event stream, which is
ground truth.
