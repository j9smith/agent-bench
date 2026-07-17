"""Poisson arrival orchestrator -- open-loop, time-bounded load.

WHY THIS EXISTS
Fixed-concurrency batch launch models a CLOSED system: a fixed population launched
at t=0, draining over time. A real inference server is an OPEN system: requests
arrive independently and unpredictably, and the concurrency in flight is an EMERGENT
result of arrival rate x session duration, not something you set. This orchestrator
generates that open-loop load.

WHAT IT DOES
- Runs for a fixed WALL-CLOCK window (--duration), not a fixed task count. You then
  analyse the middle of the window (trim ramp-up/drain) as a snapshot of a busy
  server at steady state.
- Launches new sessions as a POISSON process at rate --lambda (sessions/sec).
  Inter-arrival times are exponential; the count in any window is itself random.
  This is the honest model of independent, exogenous arrivals.
- Concurrency is NOT controlled. It floats. num_requests_running (already scraped by
  Prometheus every 5s) is the measured concurrency; you plot pressure AGAINST it,
  bucketing 5s samples by their instantaneous concurrency downstream.
- A fraction of sessions (--abandon-frac) are ABANDONED: they run a few turns to
  establish KV in the pool, then stop and never resume. This is the pure LRU stress
  case -- KV that will never be reused but ages out only on recency.
- Supports chat, agents, or BOTH at once (--mix), into one run-id. Heterogeneous
  tenancy falls out: two arrival streams into one pool, separated downstream by the
  session_type column the proxy already logs.

WHY BOTH CHAT AND AGENTS GET ARRIVALS
A real server doesn't distinguish "agent traffic" from "chat traffic" at the arrival
layer -- it just receives requests. The agent-vs-chat distinction lives in the SHAPE
of each session (endogenous tool gaps + resumption for agents; exogenous think-time
for chat), which is preserved per-session. The arrival process is open-loop for both.

TESTABILITY NOTE
The arrival scheduling + session lifecycle is concurrent and timing-dependent -- the
part most likely to have races that a mock can't surface. Verified here against a
mock endpoint for control-flow correctness; SHAKE OUT ON THE POD under real load and
expect to fix timing edge cases.
"""
from __future__ import annotations

import argparse
import os
import random
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


# ------------------------------------------------------------------ session runners
# Each returns quickly-ish; they're run in their own threads. They reuse the existing
# per-session logic rather than reimplementing it.

def _run_chat_session(sess_id: int, conv: dict, args, stop_at: float,
                      abandon_after: int | None) -> None:
    """One chat session: replay user turns, sleeping the REAL (capped) inter-turn gap
    from ShareChat between turns. Streams, so TTFT is real. `abandon_after` (if set)
    stops the session after that many turns -- the cache-polluter case."""
    import httpx
    import json as _json

    base = args.proxy_base_url.rstrip("/")
    sid = f"chat-{sess_id:06d}"
    url = f"{base}/sess/chat/{sid}/v1/chat/completions"
    headers = {"X-Session-Id": sid, "X-Session-Type": "chat", "X-Run-Id": args.run_id,
               "Authorization": f"Bearer {os.environ.get('PROXY_API_KEY', 'dummy')}"}

    turns = conv["turns"]
    gaps = conv.get("gaps") or []
    history: list[dict] = []
    user_k = 0
    with httpx.Client(timeout=args.timeout) as client:
        for t in turns:
            if t["role"] != "user":
                continue
            if time.time() > stop_at:
                return
            if abandon_after is not None and user_k >= abandon_after:
                return  # abandoned: leaves its KV resident, never resumes
            # real inter-turn gap (capped), before issuing this turn
            g = gaps[user_k] if user_k < len(gaps) else None
            if g is not None and user_k > 0:
                time.sleep(min(g, args.gap_cap))
            user_k += 1
            history.append({"role": "user", "content": t["content"]})
            body = {"model": args.model, "messages": history, "stream": True,
                    "max_tokens": args.max_tokens, "temperature": args.temperature}
            reply = ""
            try:
                with client.stream("POST", url, json=body, headers=headers) as r:
                    if r.status_code >= 400:
                        r.read()
                        return
                    for line in r.iter_lines():
                        if not line.startswith("data: "):
                            continue
                        p = line[6:].strip()
                        if p == "[DONE]":
                            break
                        try:
                            obj = _json.loads(p)
                        except ValueError:
                            continue
                        for ch in obj.get("choices") or []:
                            reply += (ch.get("delta") or {}).get("content") or ""
            except httpx.HTTPError:
                return
            history.append({"role": "assistant", "content": reply})


def _run_agent_session(sess_id: int, instance: dict, args, stop_at: float,
                       abandon_after: int | None) -> None:
    """One agent session via the OpenHands driver's real run_one(). Agents carry their
    own endogenous gaps (tool execution) internally, so we don't inject gaps -- the
    point is that the ARRIVAL of the session is Poisson, not its internal rhythm.

    Abandonment isn't cleanly expressible for agents (run_one spawns a subprocess that
    runs the whole task), so for agents --abandon-frac is a no-op here; abandonment is a
    chat-side lever. Documented rather than faked."""
    from types import SimpleNamespace
    from drivers.openhands.run_openhands import run_one

    logdir = ROOT / "logs" / "openhands"
    logdir.mkdir(parents=True, exist_ok=True)
    # run_one reads these off `args`; build a shim carrying exactly what it needs.
    shim = SimpleNamespace(
        proxy_base_url=args.proxy_base_url, run_id=args.run_id, model=args.model,
        condenser=False, condenser_max_size=0, delegation=False,
        task_timeout=args.timeout)
    try:
        run_one(instance, shim, logdir)
    except Exception as exc:  # a failed agent must not kill the arrival loop
        print(f"[arrivals] agent session {sess_id} error: {exc}", file=sys.stderr)


# ------------------------------------------------------------------ arrival loop

def poisson_arrivals(args) -> None:
    rnd = random.Random(args.seed)
    live: list[threading.Thread] = []
    start = time.time()
    stop_at = start + args.duration
    sess_id = 0
    launched = {"chat": 0, "agent": 0}

    # load session sources
    chat_pool = agent_pool = None
    if args.mix in ("chat", "both"):
        import json as _json
        chat_pool = _json.loads(Path(args.chat_file).read_text())
        if not chat_pool:
            print("[arrivals] no chat conversations loaded", file=sys.stderr)
    if args.mix in ("agent", "both"):
        from drivers.openhands.run_openhands import swebench_instances
        agent_pool = swebench_instances(args.swe_subset, args.swe_split, 0, True)

    print(f"[arrivals] duration={args.duration}s lambda={args.lam}/s mix={args.mix} "
          f"abandon={args.abandon_frac} run={args.run_id}", file=sys.stderr)

    while time.time() < stop_at:
        # exponential inter-arrival wait -> Poisson process
        wait = rnd.expovariate(args.lam) if args.lam > 0 else 1.0
        # don't oversleep past the window end
        if time.time() + wait > stop_at:
            break
        time.sleep(wait)

        # pick session type per the mix ratio
        if args.mix == "both":
            kind = "chat" if rnd.random() < args.chat_frac else "agent"
        else:
            kind = args.mix
        abandon_after = None
        if rnd.random() < args.abandon_frac:
            abandon_after = rnd.randint(args.abandon_min_turns, args.abandon_max_turns)

        if kind == "chat" and chat_pool:
            conv = rnd.choice(chat_pool)
            th = threading.Thread(target=_run_chat_session,
                                  args=(sess_id, conv, args, stop_at, abandon_after),
                                  daemon=True)
        elif kind == "agent" and agent_pool:
            inst = rnd.choice(agent_pool)
            th = threading.Thread(target=_run_agent_session,
                                  args=(sess_id, inst, args, stop_at, abandon_after),
                                  daemon=True)
        else:
            continue
        th.start()
        live.append(th)
        launched[kind] += 1
        sess_id += 1
        # opportunistic reap of finished threads so the list doesn't grow unbounded
        live = [t for t in live if t.is_alive()]

    elapsed = time.time() - start
    print(f"[arrivals] window closed at {elapsed:.0f}s. launched: "
          f"chat={launched['chat']} agent={launched['agent']}. "
          f"draining {sum(t.is_alive() for t in live)} in-flight...", file=sys.stderr)
    # let in-flight sessions finish their current turn but don't wait forever
    deadline = time.time() + args.drain
    for t in live:
        t.join(timeout=max(0.0, deadline - time.time()))
    print("[arrivals] done.", file=sys.stderr)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default="default")
    ap.add_argument("--duration", type=float, default=300, help="wall-clock window, s")
    ap.add_argument("--lam", "--lambda", dest="lam", type=float, default=0.5,
                    help="arrival rate, sessions/sec (Poisson)")
    ap.add_argument("--mix", choices=["chat", "agent", "both"], default="chat")
    ap.add_argument("--chat-frac", type=float, default=0.5,
                    help="fraction of arrivals that are chat when --mix both")
    ap.add_argument("--abandon-frac", type=float, default=0.0,
                    help="fraction of sessions abandoned mid-way (LRU stress)")
    ap.add_argument("--abandon-min-turns", type=int, default=1)
    ap.add_argument("--abandon-max-turns", type=int, default=3)
    ap.add_argument("--gap-cap", type=float, default=60.0,
                    help="cap real ShareChat inter-turn gaps at this many seconds")
    ap.add_argument("--drain", type=float, default=120,
                    help="max seconds to let in-flight sessions finish after window")
    ap.add_argument("--chat-file", default=str(ROOT / "data" / "sharechat.json"))
    ap.add_argument("--swe-subset", default="lite")
    ap.add_argument("--swe-split", default="test")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--timeout", type=float, default=600)
    ap.add_argument("--model", default=os.environ.get("MODEL", "Qwen/Qwen3-32B-FP8"))
    ap.add_argument("--proxy-base-url",
                    default=os.environ.get("PROXY_BASE_URL", "http://127.0.0.1:9000"))
    args = ap.parse_args()

    if args.mix in ("chat", "both") and not Path(args.chat_file).exists():
        print(f"[arrivals] no chat file at {args.chat_file} -- run "
              f"drivers/chat_replay/fetch_sharechat.py first", file=sys.stderr)
        return 2
    poisson_arrivals(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
