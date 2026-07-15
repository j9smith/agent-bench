"""Join the proxy request log to server-side Prometheus series, for either engine.

`--engine vllm` and `--engine sglang` produce tables with IDENTICAL column names,
so downstream analysis never branches on engine.

Three things the naive mapping gets wrong, handled here:

1. Metric names contain colons (`vllm:prefix_cache_queries`, `sglang:cache_hit_rate`).
   Colons are reserved for recording rules in PromQL, so `sum(vllm:foo)` is not a safe
   thing to write. Every query goes through `sum({__name__="..."})` instead.

2. Names drift. vLLM Counters get a `_total` suffix on the wire that the docs omit;
   SGLang changed its prefix from `sglang:` to `sglang_` in v0.5.4+. Rather than
   trusting version docs, every logical metric carries a list of candidate names and we
   probe Prometheus for whichever one actually has data. `--engine auto` does the same.

3. `sglang:num_retractions` is a HISTOGRAM ("distribution of retraction counts per
   request"), not a counter. Mapping it directly onto `vllm:num_preemptions_total`
   returns nothing at all. We take `_sum` (total retraction events) as the closest
   equivalent -- see README for why the two are near-equivalent but not identical.

On cache hit rate, and why there are two columns
------------------------------------------------
vLLM exposes token-granularity counters (queries += prompt tokens, hits += tokens found
cached). SGLang exposes `cache_hit_rate`, a gauge the engine computes itself. These are
not the same measurement, and quietly stacking them in one column would make the
cross-engine comparison -- the entire point of the new acceptance criterion -- an
artifact of two different definitions.

So:
  prefix_cache_hit_rate           token-granularity, derived from counter deltas over the
                                  scrape window, THE SAME WAY for both engines. vLLM:
                                  hits/queries. SGLang: cached_tokens/prompt_tokens.
                                  This is the column to compare across engines.
  prefix_cache_hit_rate_reported  whatever the engine says about itself. Sanity check
                                  only; not cross-comparable.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]

# logical name -> candidate Prometheus series, in probe order.
ENGINE_METRICS: dict[str, dict[str, list[str]]] = {
    "vllm": {
        "kv_cache_utilization":    ["vllm:kv_cache_usage_perc", "vllm:gpu_cache_usage_perc"],
        "queue_depth":             ["vllm:num_requests_waiting"],
        "running_requests":        ["vllm:num_requests_running"],
        "eviction_events":         ["vllm:num_preemptions_total", "vllm:num_preemptions"],
        "prompt_tokens_total":     ["vllm:prompt_tokens_total", "vllm:prompt_tokens"],
        "generation_tokens_total": ["vllm:generation_tokens_total", "vllm:generation_tokens"],
        "_cache_hits":             ["vllm:prefix_cache_hits_total", "vllm:prefix_cache_hits"],
        "_cache_queries":          ["vllm:prefix_cache_queries_total",
                                    "vllm:prefix_cache_queries"],
        "_hit_rate_reported":      ["vllm:gpu_prefix_cache_hit_rate"],  # gone in V1; harmless
    },
    "sglang": {
        # both prefixes probed: sglang_ (v0.5.4+) and sglang: (older)
        "kv_cache_utilization":    ["sglang_token_usage", "sglang:token_usage"],
        "queue_depth":             ["sglang_num_queue_reqs", "sglang:num_queue_reqs"],
        # num_running_reqs is split by phase="prefill"/"decode" on some versions and
        # unlabelled on others. sum() collapses either shape to the vLLM-equivalent.
        "running_requests":        ["sglang_num_running_reqs", "sglang:num_running_reqs"],
        # HISTOGRAM: _sum is total retraction events. Not a counter.
        "eviction_events":         ["sglang_num_retractions_sum", "sglang:num_retractions_sum",
                                    "sglang_num_retractions_total", "sglang_num_retractions"],
        "prompt_tokens_total":     ["sglang_prompt_tokens_total", "sglang:prompt_tokens_total"],
        "generation_tokens_total": ["sglang_generation_tokens_total",
                                    "sglang:generation_tokens_total"],
        "_cache_hits":             ["sglang_cached_tokens_total", "sglang:cached_tokens_total"],
        "_cache_queries":          ["sglang_prompt_tokens_total", "sglang:prompt_tokens_total"],
        "_hit_rate_reported":      ["sglang_cache_hit_rate", "sglang:cache_hit_rate"],
    },
}

# Cumulative counters: diffed per scrape bucket.
COUNTER_LOGICALS = {"eviction_events", "prompt_tokens_total", "generation_tokens_total",
                    "_cache_hits", "_cache_queries"}

# The server-side schema is FIXED, not whatever each engine happens to expose. Any
# column with no backing series is emitted as NaN rather than omitted.
#
# This is load-bearing, not tidiness: vLLM V1 removed its engine-reported hit-rate
# gauge (replaced by the counters), so `prefix_cache_hit_rate_reported` exists for
# SGLang and not for vLLM. Without pinning, the two tables would silently differ in
# shape and any code that concatenates them -- which is the whole point of the
# cross-engine comparison -- would get a column of NaNs it didn't ask for, or throw.
SERVER_SCHEMA = [
    "kv_cache_utilization",
    "queue_depth",
    "running_requests",
    "eviction_events",
    "eviction_events_window",
    "prompt_tokens_total",
    "prompt_tokens_total_delta",
    "generation_tokens_total",
    "generation_tokens_total_delta",
    "prefix_cache_hit_rate",
    "prefix_cache_hit_rate_reported",
]


class PrometheusUnreachable(Exception):
    """Prometheus isn't answering at all -- distinct from 'answered, no data'."""


def query_range(base, expr, start, end, step):
    try:
        r = requests.get(f"{base}/api/v1/query_range",
                         params={"query": expr, "start": start, "end": end, "step": step},
                         timeout=60)
    except requests.exceptions.RequestException as exc:
        # Connection refused / DNS / timeout: the server isn't there. Surface it as
        # one clean signal instead of a stack trace, so a missing Prometheus degrades
        # the export to sender-side-only rather than killing it.
        raise PrometheusUnreachable(str(exc)) from None
    r.raise_for_status()
    d = r.json()
    return d["data"]["result"] if d.get("status") == "success" else []


def probe(base, candidates, start, end, step):
    """First candidate series that actually has data. Colon-safe."""
    for name in candidates:
        res = query_range(base, f'sum({{__name__="{name}"}})', start, end, step)
        if res and res[0].get("values"):
            return name, res[0]["values"]
    return None, None


def detect_engine(base, start, end, step):
    for engine in ("vllm", "sglang"):
        name, _ = probe(base, ENGINE_METRICS[engine]["kv_cache_utilization"], start, end, step)
        if name:
            return engine
    return None


def server_frame(base, engine, start, end, step):
    """Wide frame indexed by scrape ts, columns = LOGICAL names (engine-independent)."""
    cols, resolved, missing = {}, {}, []
    for logical, candidates in ENGINE_METRICS[engine].items():
        name, values = probe(base, candidates, start, end, step)
        if not name:
            missing.append(logical)
            continue
        resolved[logical] = name
        cols[logical] = pd.Series(
            {float(t): pd.to_numeric(v, errors="coerce") for t, v in values})

    if missing:
        print(f"[export] {engine}: no series found for {', '.join(missing)} "
              f"(skipped; check names with `curl <pod>/metrics`)", file=sys.stderr)
    if not cols:
        return pd.DataFrame(), resolved

    df = pd.DataFrame(cols).sort_index()

    # Counters -> per-bucket deltas, so a request maps to activity in its own scrape
    # window rather than to lifetime totals.
    for logical in COUNTER_LOGICALS & set(df.columns):
        df[logical + "_delta"] = df[logical].diff()

    # The one column that is genuinely comparable across engines.
    if {"_cache_hits_delta", "_cache_queries_delta"} <= set(df.columns):
        denom = df["_cache_queries_delta"].where(df["_cache_queries_delta"] > 0)
        df["prefix_cache_hit_rate"] = df["_cache_hits_delta"] / denom
    if "_hit_rate_reported" in df.columns:
        df["prefix_cache_hit_rate_reported"] = df["_hit_rate_reported"]

    df = df.rename(columns={"eviction_events_delta": "eviction_events_window"})
    df = df.drop(columns=[c for c in df.columns if c.startswith("_")], errors="ignore")
    for col in SERVER_SCHEMA:                       # pin the schema; NaN, never absent
        if col not in df.columns:
            df[col] = pd.NA
    df = df[SERVER_SCHEMA]
    return df.rename_axis("ts").reset_index(), resolved


def _lcp(a: list, b: list) -> int:
    """Length of the common leading run of two lists."""
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def derive_branching(df: pd.DataFrame) -> pd.DataFrame:
    """Reconstruct the KV tree from hashes alone -- no message content needed.

    A single OpenHands process can hold several conversations at once: the parent
    agent, each delegated sub-agent, and the condenser's summarisation calls. They all
    carry identical static headers, so they're separated by `sequence_id` (the hash of
    each conversation's root), which the proxy derives.

    What we compute here:

      sequence_role         root / subagent / oneshot. `oneshot` sequences never get a
                            second turn -- that's the condenser's summariser. HEURISTIC:
                            cross-check it against the driver's own Condensation event
                            count in logs/openhands/summary.json.
      fanout                distinct sequences in this task (1 = no delegation)
      shared_prefix_msgs    how many leading messages this sequence's first request
      shared_prefix_chars   shares with the task's ROOT sequence.

    That last pair is the measurement that matters. The delegation literature says
    sub-agents get an isolated context -- so the shared prefix should be just the
    system prompt, and the fan-out should buy the KV cache essentially nothing. This
    turns that from a claim we read into a number we measured.
    """
    if "task_id" not in df.columns or "sequence_id" not in df.columns:
        return df
    df = df.copy()

    g = df.groupby("task_id")["sequence_id"]
    df["fanout"] = g.transform("nunique")

    # Root = the sequence that spoke first in the task.
    first_ts = df.groupby(["task_id", "sequence_id"])["ts_request_in"].transform("min")
    df["_seq_first_ts"] = first_ts
    roots = (df.sort_values("_seq_first_ts")
               .groupby("task_id")["sequence_id"].first().to_dict())
    df["_is_root"] = df.apply(
        lambda r: roots.get(r["task_id"]) == r["sequence_id"], axis=1)

    max_turn = df.groupby(["task_id", "sequence_id"])["turn_number"].transform("max")
    df["sequence_role"] = "subagent"
    df.loc[max_turn == 0, "sequence_role"] = "oneshot"   # condenser summariser
    df.loc[df["_is_root"], "sequence_role"] = "root"

    # Shared prefix depth against the task's root sequence.
    if "cum_prefix_hashes" in df.columns:
        root_spine = {}
        for (tid, sid), grp in df.groupby(["task_id", "sequence_id"]):
            if roots.get(tid) == sid:
                first = grp.sort_values("ts_request_in").iloc[0]
                root_spine[tid] = (first.get("cum_prefix_hashes") or [],
                                   first.get("cum_prefix_chars") or [])

        def depth(row):
            spine, chars = root_spine.get(row["task_id"], ([], []))
            mine = row.get("cum_prefix_hashes") or []
            n = _lcp(spine, mine)
            return pd.Series({
                "shared_prefix_msgs": n,
                "shared_prefix_chars": (chars[n - 1] if 0 < n <= len(chars) else 0),
            })

        df[["shared_prefix_msgs", "shared_prefix_chars"]] = df.apply(depth, axis=1)
        df["shared_prefix_frac"] = (df["shared_prefix_chars"]
                                    / df["prompt_chars"].where(df["prompt_chars"] > 0))

    return df.drop(columns=["_seq_first_ts", "_is_root"], errors="ignore")


def derive_session_shape(df: pd.DataFrame) -> pd.DataFrame:
    # Group by SEQUENCE, not session. A session (= one process) may contain a parent,
    # its sub-agents and the condenser's summariser; diffing prompt_tokens across an
    # interleaving of those would produce garbage.
    key = "sequence_id" if "sequence_id" in df.columns else "session_id"
    df = df.sort_values([key, "ts_request_in"], kind="stable").copy()

    # e2e is always meaningful: request issued -> last byte back.
    df["e2e_s"] = df["ts_last_byte"] - df["ts_request_in"]

    # TTFT and ITL only mean something for STREAMED responses. A non-streamed call
    # (OpenHands' default) returns the whole body at once, so ts_first_byte ==
    # ts_last_byte and "time to first token" is not a distinct quantity -- reporting
    # ts_first_byte - ts_request_in as TTFT would be a lie (it's really e2e). So we
    # null TTFT/ITL where the proxy saw no stream chunks, rather than emit a
    # misleading ~0 decode window. Chat traffic streams, so it keeps real TTFT/ITL.
    streamed = df["stream_chunks"].notna() if "stream_chunks" in df.columns else False
    df["ttft_s"] = (df["ts_first_byte"] - df["ts_request_in"]).where(streamed)
    decode_s = (df["ts_last_byte"] - df["ts_first_byte"]).where(streamed)
    df["itl_s"] = decode_s / df["completion_tokens"].where(df["completion_tokens"] > 0)
    df["streamed"] = streamed if isinstance(streamed, bool) else streamed.fillna(False)

    g = df.groupby(key, dropna=True)
    # Gap: wall time from the previous response completing to this request being issued.
    # Agent sessions: tool execution. Chat: the replay delay.
    df["gap_s"] = df["ts_request_in"] - g["ts_last_byte"].shift(1)
    df["context_growth_tokens"] = df["prompt_tokens"] - g["prompt_tokens"].shift(1)
    df["is_resumption"] = df["gap_s"].notna()

    # Exact per-request prefix reuse, from our own hashes. Independent of Prometheus and
    # therefore identical across engines by construction -- which is what makes it a fair
    # denominator when comparing what the two engines' caches did with the same workload.
    df["prefix_seen_before"] = df.duplicated(subset=["prefix_hash"], keep="first")
    df["exact_resend"] = df.duplicated(subset=["full_hash"], keep="first")
    df["branch_factor"] = (df.groupby([key, "prefix_hash"])["full_hash"]
                             .transform("nunique").fillna(1))

    # Condensation: within one sequence, the context SHRANK. Only a scaffold that
    # rewrites its own history can do that -- OpenHands' condenser, or the controller's
    # fallback truncation. Either way the prefix has changed underneath the server, and
    # the next request's prefix-cache lookup will miss. That miss is the cost the
    # condenser trades latency for, and this column is where you can see it.
    df["is_condensation"] = df["context_growth_tokens"] < 0
    df["context_shrink_tokens"] = (-df["context_growth_tokens"]).where(
        df["is_condensation"])
    return df


def discover_series(base: str, engine: str) -> list[str]:
    """Every series name the engine exposes, not just the ones we mapped.

    The mapping in ENGINE_METRICS is a *view*. It is chosen in advance, and it will be
    wrong about something -- a series we didn't think to want, a histogram we ignored, a
    label we summed away. GPU-hours are expensive and this workload is not cheap to
    reproduce, so we archive everything the engine emits and reduce afterwards.
    """
    try:
        r = requests.get(f"{base}/api/v1/label/__name__/values", timeout=30)
        r.raise_for_status()
        names = r.json().get("data", [])
    except Exception as exc:
        print(f"[export] could not enumerate series: {exc}", file=sys.stderr)
        return []
    prefixes = ("vllm:",) if engine == "vllm" else ("sglang:", "sglang_")
    return sorted(n for n in names if n.startswith(prefixes))


def dump_raw_series(base, engine, start, end, step, out: Path) -> int:
    """Long-format archive: one row per (series, label-set, timestamp).

    Labels are preserved rather than summed away -- SGLang's phase="prefill"/"decode"
    split and per-rank labels are exactly the sort of thing you don't know you needed
    until the joined table says something surprising. Histogram buckets come along for
    free, which is where the server-side TTFT/e2e distributions live.
    """
    names = discover_series(base, engine)
    if not names:
        return 0
    rows = []
    for name in names:
        for series in query_range(base, f'{{__name__="{name}"}}', start, end, step):
            labels = {k: v for k, v in series.get("metric", {}).items() if k != "__name__"}
            for ts, val in series.get("values", []):
                rows.append({"metric": name, "labels": json.dumps(labels, sort_keys=True),
                             "ts": float(ts), "value": pd.to_numeric(val, errors="coerce")})
    if not rows:
        return 0
    df = pd.DataFrame(rows)
    try:
        df.to_parquet(out.with_suffix(".parquet"), index=False)
        written = out.with_suffix(".parquet")
    except Exception:
        df.to_csv(out.with_suffix(".csv"), index=False)
        written = out.with_suffix(".csv")
    print(f"[export] archived {len(names)} raw series "
          f"({len(df):,} samples) -> {written}")
    return len(names)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", choices=["vllm", "sglang", "auto"], default="auto")
    ap.add_argument("--requests", type=Path, default=ROOT / "logs" / "requests.jsonl")
    ap.add_argument("--prometheus", default="http://127.0.0.1:9090")
    ap.add_argument("--step", type=int, default=15, help="must match scrape_interval")
    ap.add_argument("--out", type=Path, default=None, help="default: analysis/results_<engine>")
    ap.add_argument("--no-prometheus", action="store_true")
    ap.add_argument("--no-raw", action="store_true",
                    help="skip the raw series archive (you probably don't want this)")
    args = ap.parse_args()

    if not args.requests.exists():
        print(f"no request log at {args.requests}", file=sys.stderr)
        return 2
    rows = [json.loads(l) for l in args.requests.read_text().splitlines() if l.strip()]
    if not rows:
        print("request log is empty", file=sys.stderr)
        return 2

    req = derive_branching(derive_session_shape(pd.DataFrame(rows)))
    print(f"[export] {len(req)} requests, {req['session_id'].nunique()} sessions "
          f"({req.groupby('session_type').size().to_dict()})")
    if "sequence_role" in req.columns:
        by_role = req.groupby("sequence_role").sequence_id.nunique().to_dict()
        print(f"[export] sequences by role: {by_role}")
        fan = req.groupby("task_id").sequence_id.nunique()
        print(f"[export] fan-out per task: median={fan.median():.0f} max={fan.max():.0f}")
        if "shared_prefix_frac" in req.columns:
            sub = req[req.sequence_role == "subagent"]
            if len(sub):
                print(f"[export] sub-agent context shared with parent: "
                      f"median {sub.shared_prefix_frac.median():.1%} of its prompt")
        n_cond = int(req.is_condensation.sum())
        if n_cond:
            print(f"[export] {n_cond} condensation events "
                  f"(context shrank mid-sequence)")

    engine = args.engine
    resolved: dict = {}
    n_raw = 0
    if not args.no_prometheus:
        start = float(req["ts_request_in"].min()) - args.step * 2
        end = float(req["ts_last_byte"].max()) + args.step * 2

        try:
            if engine == "auto":
                engine = detect_engine(args.prometheus, start, end, args.step)
                if not engine:
                    print("[export] Prometheus is up but no engine series matched. Is the "
                          "scrape job configured, and does retention cover the run window? "
                          "Writing sender-side-only table.", file=sys.stderr)
                    engine = "unknown"
                    raise PrometheusUnreachable("no engine detected")
                print(f"[export] detected engine: {engine}")

            # Archive FIRST. The joined table is a lossy reduction of this; if the
            # reduction turns out wrong, we want the inputs still on disk.
            if not args.no_raw:
                n_raw = dump_raw_series(args.prometheus, engine, start, end, args.step,
                                        (args.out.parent if args.out else ROOT / "analysis")
                                        / f"raw_series_{engine}")

            srv, resolved = server_frame(args.prometheus, engine, start, end, args.step)
            if resolved:
                print("[export] resolved series:")
                for k, v in resolved.items():
                    print(f"           {k:<26} -> {v}")

            if srv.empty:
                print("[export] Prometheus returned nothing for the run window.",
                      file=sys.stderr)
                for col in ["ts"] + SERVER_SCHEMA:
                    req[col] = pd.NA
            else:
                req = pd.merge_asof(
                    req.sort_values("ts_request_in"), srv.sort_values("ts"),
                    left_on="ts_request_in", right_on="ts",
                    direction="backward", tolerance=float(args.step * 2))
                print(f"[export] joined server metrics to "
                      f"{req['ts'].notna().mean():.0%} of requests")
                ev = req.get("eviction_events")
                if ev is not None and pd.notna(ev.max()) and ev.max() == 0:
                    print("[export] NOTE: eviction/preemption count never left zero. The "
                          "KV pool never filled, so nothing about eviction or admission is "
                          "answerable from this run. Raise --concurrency, or shrink the pool.")

        except PrometheusUnreachable as exc:
            # The whole point of the fix: a missing/empty Prometheus must NOT lose you
            # the sender-side table, which is most of section 0 anyway.
            print(f"[export] Prometheus unreachable at {args.prometheus} ({exc}).",
                  file=sys.stderr)
            print("[export] Writing sender-side-only table. Start Prometheus and re-run "
                  "to attach server-side metrics -- the raw request log is unchanged, so "
                  "nothing is lost as long as Prometheus retention still covers the window.",
                  file=sys.stderr)
            for col in ["ts"] + SERVER_SCHEMA:
                req[col] = pd.NA

    req["engine"] = engine if engine not in ("auto", "unknown") else None

    out = args.out or (ROOT / "analysis" / f"results_{engine}")
    out.parent.mkdir(parents=True, exist_ok=True)

    # A joined table with no provenance is not evidence. Record what produced it.
    manifest = {
        "engine": engine,
        "requests_log": str(args.requests),
        "prometheus": args.prometheus,
        "scrape_step_s": args.step,
        "exported_at": time.time(),
        "run_start": float(req["ts_request_in"].min()),
        "run_end": float(req["ts_last_byte"].max()),
        "n_requests": int(len(req)),
        "n_sessions": int(req["session_id"].nunique()),
        "sessions_by_type": {k: int(v) for k, v in req.groupby("session_type").size().items()},
        "models_seen": sorted(x for x in req["model"].dropna().unique()),
        "resolved_series": resolved,
        "n_raw_series_archived": n_raw,
        "server_schema": SERVER_SCHEMA,
    }
    (out.parent / f"manifest_{engine}.json").write_text(json.dumps(manifest, indent=2))

    # The per-message hash spine is an analysis input, not an output column -- it
    # would bloat the table by orders of magnitude. It's already been consumed by
    # derive_branching(), and the raw JSONL still has it if you need to redo that.
    req = req.drop(columns=["cum_prefix_hashes", "cum_prefix_chars"], errors="ignore")

    req.to_csv(out.with_suffix(".csv"), index=False)
    try:
        req.to_parquet(out.with_suffix(".parquet"), index=False)
        print(f"[export] wrote {out.with_suffix('.csv')} and {out.with_suffix('.parquet')}")
    except Exception as exc:
        print(f"[export] wrote {out.with_suffix('.csv')} (parquet skipped: {exc})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
