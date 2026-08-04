"""
Logging proxy for agentic-bench.

Sits between the workload drivers (mini-swe-agent, chat replay) and the vLLM
server. Forwards requests untouched, streams responses back untouched, and
writes one JSONL line per request to logs/requests.jsonl.

Design notes
------------
1. Byte-identity. Streaming responses are re-emitted at SSE event boundaries,
   not raw socket reads, so the concatenated body the client sees is identical
   to what upstream sent -- with one exception, see (2).

2. Usage capture on streamed requests. OpenAI-compatible streaming responses
   only carry a `usage` block if the request sets
   `stream_options.include_usage = true`. Agent scaffolds generally don't.
   So the proxy *injects* that option (PROXY_INJECT_USAGE=1, the default) and
   then *suppresses the resulting usage-only chunk* on the way back out, so the
   client sees the stream it would have seen anyway. If the client asked for
   usage itself, we pass the chunk through untouched.

   Set PROXY_INJECT_USAGE=0 to disable both halves (used for the phase-2
   byte-identity acceptance check; also the escape hatch if some upstream
   rejects stream_options).

3. Turn numbering. `X-Turn-Number` is honoured if the driver sends it, but
   mini-swe-agent can only attach *static* headers per process (via litellm
   extra_headers), so a per-call turn number can't be injected without forking
   the scaffold. Instead the proxy maintains a monotonic per-session counter and
   fills `turn_number` itself when the header is absent. Requests with no
   session header at all are logged with nulls and are never failed.

4. KV retention hints (optional; OFF unless asked for). The proxy is the only
   component that sees both the harness's notion of a trajectory AND vLLM's
   `request_id` (the `id` field of every completion response), so it is where a
   client-side retention hint has to be assembled.

   Both hints POST to the upstream `/kv_hint` endpoint, which only exists on a
   hint-enabled vLLM build. A 404 is logged and otherwise ignored, so pointing
   this proxy at stock vLLM changes nothing.

   dontneed  POST /_hint/session_done {"task_id": ...} when a trajectory ends.
             The client knows this exactly -- no estimate, no threshold. LRU
             does the opposite of the right thing here: freed blocks land at the
             most-recently-used end of the queue, so a FINISHED trajectory's
             context becomes the most protected material in the pool, while a
             live sequence paused mid-tool-call ages toward eviction.

   willneed/pageout  Needs a predicted gap, so it is only as good as its
             predictor. The predictor here is a per-tool EWMA of the OBSERVED
             dispatch gap: the proxy reads the tool name from each response and,
             when the next request on that sequence arrives, attributes the
             measured gap to it. Self-calibrating, needs no model introspection,
             and measures the gap the serving layer actually experiences rather
             than tool execution time. Enabled per-request via the header
             `X-KV-Hint-Gap: 1`, so arms switch without restarting the proxy.
             UNTESTED as of writing -- run the dontneed arm first.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import time
import uuid
from collections import OrderedDict, defaultdict
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse

UPSTREAM_BASE_URL = os.environ.get("UPSTREAM_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
INJECT_USAGE = os.environ.get("PROXY_INJECT_USAGE", "1") not in ("0", "false", "False")
REQUEST_TIMEOUT = float(os.environ.get("PROXY_TIMEOUT_S", "1800"))

# --- KV hint knobs. All inert unless a hint is actually requested. ---
# Bound on the (task, sequence) -> last upstream request id map. Entries only
# need to outlive one trajectory.
KV_HINT_CAP = int(os.environ.get("PROXY_KV_HINT_CAP", "4096"))
# Gap-hint predictor: EWMA smoothing, and how many observations a tool needs
# before its EWMA is trusted enough to hint on. Below that we stay silent
# rather than send a made-up number.
KV_GAP_ALPHA = float(os.environ.get("PROXY_KV_GAP_ALPHA", "0.2"))
KV_GAP_MIN_SAMPLES = int(os.environ.get("PROXY_KV_GAP_MIN_SAMPLES", "5"))

# --- PLAS (program-level attained service). Inert unless asked for. ---
# Autellix Eq. 1: a call's priority is the summed service of its program's prior
# completed calls. vLLM's `priority` uses the same convention (larger = lower
# priority), so the formula ports with no sign flip.
PLAS_ENABLED = os.environ.get("PROXY_PLAS", "0") in ("1", "true", "True")
# "task": one program = one trajectory (Autellix's pid). "sequence": one program
# = one conversation. These diverge once the condenser or delegation is on, and
# "sequence" is then wrong -- a sub-agent would look like a fresh program.
PLAS_KEY = os.environ.get("PROXY_PLAS_KEY", "task")
# "decode": completion_tokens only. "computed": (prompt - cached) + completion,
# i.e. tokens actually pushed through the model. Falls back to decode when the
# upstream omits cached_tokens.
PLAS_SERVICE = os.environ.get("PROXY_PLAS_SERVICE", "decode")
# 0/1 = continuous priority. >1 buckets it, the cheapest approximation of
# Autellix's MLFQ discretisation; they discretise specifically because
# continuous priority degenerates toward round-robin and thrashes KV.
PLAS_QUANTUM = int(os.environ.get("PROXY_PLAS_QUANTUM", "0"))


# Hop-by-hop headers we must not forward.
_DROP_REQUEST_HEADERS = {"host", "content-length", "connection", "accept-encoding"}
_DROP_RESPONSE_HEADERS = {"content-length", "content-encoding", "transfer-encoding", "connection"}


LOG_DIR = Path(os.environ.get("PROXY_LOG_DIR", "logs"))
DEFAULT_RUN = os.environ.get("PROXY_DEFAULT_RUN", "default")

KV_GAP_MAX_S = float(os.environ.get("PROXY_KV_GAP_MAX_S", "300"))

class RunRouter:
    """Routes each request to logs/<run_id>/requests.jsonl, opening files on demand.

    This is what lets one long-lived proxy serve many cleanly-separated runs: the run
    id arrives per-request (X-Run-Id header), so there's no proxy restart between runs
    and no cross-run pollution in a single file. Falls back to PROXY_DEFAULT_RUN for
    requests that carry no run id (e.g. stray health checks).
    """

    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self._files: dict[str, Any] = {}
        self._lock = asyncio.Lock()

    def _sanitize(self, run_id: str) -> str:
        # never let a header escape the logs dir
        safe = "".join(c for c in run_id if c.isalnum() or c in "-_.")
        return safe or DEFAULT_RUN

    async def write(self, record: dict[str, Any]) -> None:
        run = self._sanitize(record.get("run_id") or DEFAULT_RUN)
        line = json.dumps(record, separators=(",", ":"), default=str)
        async with self._lock:
            fh = self._files.get(run)
            if fh is None:
                path = self.base_dir / run / "requests.jsonl"
                path.parent.mkdir(parents=True, exist_ok=True)
                fh = path.open("a", buffering=1)
                self._files[run] = fh
            fh.write(line + "\n")

    def close(self) -> None:
        for fh in self._files.values():
            try:
                fh.close()
            except Exception:
                pass


class KVHintState:
    """Everything the hint paths need, in one lock.

    Two maps:

    _last  (task_id, sequence_id) -> most recent upstream request id. Only the
           LAST request of a sequence matters for dontneed: its block chain is a
           superset of every earlier request's in that sequence, because context
           only grows. One POST per sequence, not per turn.

    _gap   tool name -> EWMA of observed dispatch gap in ms, plus a sample
           count. Populated by observe_gap() when the next request of a sequence
           arrives, so the predictor is trained on exactly the quantity it
           predicts. No model introspection involved.

    _pending  (task, sequence) -> (tool name, ts_last_byte) of the response we
           are waiting to see a gap for.
    """

    def __init__(self, cap: int):
        self._cap = cap
        self._last: OrderedDict[tuple, str] = OrderedDict()
        self._pending: OrderedDict[tuple, tuple] = OrderedDict()
        self._gap: dict[str, list] = {}   # tool -> [ewma_ms, n]
        self._lock = asyncio.Lock()
        self.stats = defaultdict(int)

    @staticmethod
    def _trim(d: OrderedDict, cap: int) -> None:
        while len(d) > cap:
            d.popitem(last=False)

    async def note_response(self, run_id, task_id, sequence_id, req_id, tool, ts_last_byte):
        if not task_id:
            return
        key = (run_id, task_id, sequence_id)
        async with self._lock:
            if req_id:
                self._last[key] = req_id
                self._last.move_to_end(key)
                self._trim(self._last, self._cap)
            if tool and ts_last_byte:
                self._pending[key] = (tool, ts_last_byte)
                self._pending.move_to_end(key)
                self._trim(self._pending, self._cap)

    async def observe_gap(self, run_id, task_id, sequence_id, ts_request_in) -> None:
        key = (run_id, task_id, sequence_id)
        async with self._lock:
            prev = self._pending.pop(key, None)
            if not prev or not ts_request_in:
                return
            tool, ts_prev = prev
            gap_ms = (ts_request_in - ts_prev) * 1000.0
            # TTL: a "gap" longer than any real tool call is a stale entry
            # (retry, crash, or a keying escape), not training data.
            if gap_ms <= 0 or gap_ms > KV_GAP_MAX_S * 1000.0:
                if gap_ms > KV_GAP_MAX_S * 1000.0:
                    self.stats["gap_discarded_stale"] += 1
                return
            slot = self._gap.get(tool)
            if slot is None:
                self._gap[tool] = [gap_ms, 1]
            else:
                slot[0] = KV_GAP_ALPHA * gap_ms + (1 - KV_GAP_ALPHA) * slot[0]
                slot[1] += 1

    async def predict_gap_ms(self, tool: str | None) -> float | None:
        """EWMA for this tool, or None until it has enough samples."""
        if not tool:
            return None
        async with self._lock:
            slot = self._gap.get(tool)
            if not slot or slot[1] < KV_GAP_MIN_SAMPLES:
                return None
            return slot[0]

    async def pop_task(self, run_id: str, task_id: str) -> list[tuple[str, str]]:
        async with self._lock:
            keys = [k for k in self._last if k[0] == run_id and k[1] == task_id]
            out = [(k[2], self._last.pop(k)) for k in keys]
        return out

    async def snapshot(self) -> dict:
        async with self._lock:
            return {
                "tracked_sequences": len(self._last),
                "tools_learned": {t: {"ewma_gap_ms": round(v[0], 1), "n": v[1]}
                                  for t, v in sorted(self._gap.items())},
                **{k: v for k, v in sorted(self.stats.items())},
            }


async def _post_kv_hint(client: httpx.AsyncClient, payload: dict, stats) -> dict:
    """Fire one hint upstream. Never raises: a hint is advisory by construction,
    and a proxy that 500s because the serving build lacks /kv_hint would be
    strictly worse than one that logs and carries on."""
    try:
        r = await client.post("/kv_hint", json=payload, timeout=10.0)
    except Exception as exc:
        stats["hint_post_errors"] += 1
        return {"error": repr(exc)}
    if r.status_code == 404:
        stats["hint_post_404"] += 1
        return {"status": 404, "note": "upstream has no /kv_hint (stock vLLM)"}
    stats["hint_post_ok" if r.status_code < 400 else "hint_post_failed"] += 1
    return {"status": r.status_code, "body": r.text[:200]}


def _first_tool_name(payload: dict | None) -> str | None:
    """Tool name from a completion response, if it called one.

    This is the whole hint source: available to the proxy for free, before the
    tool has run, and invisible to the serving layer.
    """
    if not payload:
        return None
    try:
        for ch in payload.get("choices") or []:
            msg = ch.get("message") or ch.get("delta") or {}
            for tc in msg.get("tool_calls") or []:
                fn = (tc.get("function") or {}).get("name")
                if fn:
                    return fn
    except Exception:
        pass
    return None

def _call_service(usage: dict) -> int:
    """Service consumed by one completed call, in tokens."""
    ct = int(usage.get("completion_tokens") or 0)
    if PLAS_SERVICE != "computed":
        return ct
    pt = usage.get("prompt_tokens")
    cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
    if pt is None or cached is None:
        return ct
    return max(0, int(pt) - int(cached)) + ct


class ProcessTable:
    """PLAS process table. Single-threaded PLAS only; no ATLAS, no critical path.

    The proxy is the only component that can build this. The harness knows which
    calls belong to one program and discards that at the API boundary; the engine
    never sees it. Same structural gap the retention hints exploit, different
    resource.
    """

    def __init__(self) -> None:
        self._service: dict[str, int] = defaultdict(int)
        self._calls: dict[str, int] = defaultdict(int)
        self._lock = asyncio.Lock()

    async def service_of(self, key: str | None) -> int:
        if not key:
            return 0
        async with self._lock:
            return self._service[key]

    async def complete(self, key: str | None, service: int) -> None:
        if not key:
            return
        async with self._lock:
            self._service[key] += max(0, int(service))
            self._calls[key] += 1

    async def reset(self) -> None:
        async with self._lock:
            self._service.clear()
            self._calls.clear()

    async def snapshot(self) -> dict:
        async with self._lock:
            v = sorted(self._service.values())
            n = len(v)
            q = lambda f: v[min(n - 1, int(f * n))] if n else 0
            return {"programs": n, "calls": sum(self._calls.values()),
                    "key": PLAS_KEY, "service_metric": PLAS_SERVICE,
                    "quantum": PLAS_QUANTUM,
                    "service_p10": q(0.10), "service_p50": q(0.50),
                    "service_p90": q(0.90),
                    "spread_p90_p10": round(q(0.90) / max(q(0.10), 1), 2)}

class TurnCounter:
    def __init__(self) -> None:
        self._counts: dict[str, int] = defaultdict(int)
        self._lock = asyncio.Lock()

    async def next(self, session_id: str) -> int:
        async with self._lock:
            n = self._counts[session_id]
            self._counts[session_id] = n + 1
            return n


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.client = httpx.AsyncClient(
        base_url=UPSTREAM_BASE_URL,
        timeout=httpx.Timeout(REQUEST_TIMEOUT, connect=30.0),
    )
    app.state.log = RunRouter(LOG_DIR)
    app.state.turns = TurnCounter()
    app.state.hints = KVHintState(KV_HINT_CAP)
    app.state.plas = ProcessTable()
    print(f"[proxy] upstream={UPSTREAM_BASE_URL} log_dir={LOG_DIR} inject_usage={INJECT_USAGE}")
    yield
    await app.state.client.aclose()
    app.state.log.close()


app = FastAPI(lifespan=lifespan, title="agentic-bench proxy")


# --------------------------------------------------------------------------
# hashing
# --------------------------------------------------------------------------

def _canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha256(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def compute_hashes(body: dict[str, Any]) -> dict[str, Any]:
    """Hashes that let us reconstruct the KV tree without storing any content.

    prefix_hash   SHA256 over the message list *excluding the final turn* -- the
                  portion a prefix cache could plausibly already hold.
    full_hash     every message; detects exact re-sends (retries, re-expansions).
    sequence_root SHA256 over the conversation's ROOT (system message + first user
                  message). This is how a parent, its delegated sub-agents, and the
                  condenser's summarisation calls get told apart when they all come
                  from ONE process and therefore carry identical static headers.
                  Distinct conversations have distinct roots: the parent's first user
                  message is the SWE-bench issue; a sub-agent's is its delegated
                  instruction; a summariser call's is a summarisation prompt.
    cum_prefix_hashes
                  SHA256 of messages[0..i] for every i -- a Merkle-ish spine of the
                  context. Two requests' shared prefix DEPTH is the length of the
                  common leading run of these lists. That's what lets us measure how
                  much KV a parent and its children actually share, rather than
                  taking the delegation literature's word for it -- and it needs no
                  message content on disk.
    """
    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        return {"prefix_hash": None, "full_hash": None, "sequence_root": None,
                "cum_prefix_hashes": [], "cum_prefix_chars": [],
                "num_messages": 0, "prompt_chars": 0}

    cum_hashes, cum_chars, running = [], [], []
    for m in messages:
        running.append(m)
        cum_hashes.append(_sha256(_canonical(running)))
        cum_chars.append(len(_canonical(running)))

    # Root = system prompt + first non-system message. Falls back to messages[0].
    root = messages[:1]
    for i, m in enumerate(messages):
        if m.get("role") != "system":
            root = messages[: i + 1]
            break

    prefix = messages[:-1]
    return {
        "prefix_hash": _sha256(_canonical(prefix)) if prefix else _sha256("[]"),
        "full_hash": cum_hashes[-1],
        "sequence_root": _sha256(_canonical(root)),
        "cum_prefix_hashes": cum_hashes,
        "cum_prefix_chars": cum_chars,
        "num_messages": len(messages),
        "prompt_chars": cum_chars[-1],
    }


# --------------------------------------------------------------------------
# SSE handling
# --------------------------------------------------------------------------

def _parse_sse_event(event: bytes) -> dict[str, Any] | None:
    """Return the decoded JSON payload of an SSE `data:` event, or None."""
    for raw_line in event.split(b"\n"):
        line = raw_line.strip()
        if not line.startswith(b"data:"):
            continue
        payload = line[len(b"data:"):].strip()
        if not payload or payload == b"[DONE]":
            return None
        try:
            return json.loads(payload)
        except json.JSONDecodeError:
            return None
    return None


def _is_usage_only_chunk(obj: dict[str, Any] | None) -> bool:
    """The chunk emitted because of stream_options.include_usage: empty choices,
    populated usage."""
    if not obj:
        return False
    return obj.get("usage") is not None and not obj.get("choices")


# --------------------------------------------------------------------------
# proxy routes
# --------------------------------------------------------------------------

@app.get("/healthz")
async def healthz() -> PlainTextResponse:
    return PlainTextResponse("ok")


# NOTE ON ROUTE ORDER: these must be declared BEFORE the /{path:path} catch-all
# below, or FastAPI matches the catch-all first and forwards them upstream.
@app.post("/_hint/session_done")
async def hint_session_done(request: Request):
    """The trajectory is over: its KV is a corpse.

    Zero estimation error -- the client knows this as a fact, and the serving
    layer has no channel to learn it. Called by the driver after
    conversation.run() returns, success or failure: a failed trajectory's blocks
    are just as dead.
    """
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "body must be json"}, status_code=400)
    task_id = body.get("task_id")
    if not task_id:
        return JSONResponse({"error": "task_id required"}, status_code=400)
    run_id = body.get("run_id")
    hints: KVHintState = app.state.hints
    pairs = await hints.pop_task(run_id, task_id)
    if not pairs:
        hints.stats["done_no_known_requests"] += 1
        return JSONResponse({"task_id": task_id, "applied": 0,
                             "reason": "no_known_requests"})

    hints.stats["done_hints_sent"] += len(pairs)
    results = [await _post_kv_hint(app.state.client,
                                   {"request_id": rid, "done": True,
                                    "sequence_id": seq}, hints.stats)
               for seq, rid in pairs]
    return JSONResponse({"task_id": task_id, "sequences": len(pairs),
                         "results": results})

@app.post("/_plas/reset")
async def plas_reset():
    """Clear attained service. Call between sweep points, same as
    /reset_prefix_cache -- otherwise point N inherits point N-1's priorities."""
    await app.state.plas.reset()
    return JSONResponse({"ok": True})


@app.get("/_plas/stats")
async def plas_stats():
    return JSONResponse(await app.state.plas.snapshot())

@app.get("/_hint/stats")
async def hint_stats():
    """Proxy-side view: what the proxy tried to send, and what its gap predictor
    has learned. Distinct from vLLM's /kv_hint/stats, which reports what the
    server did with them."""
    return JSONResponse(await app.state.hints.snapshot())


def _strip_session_path(path: str) -> tuple[str, str | None, str | None]:
    """Support `/sess/{session_type}/{session_id}/v1/chat/completions`.

    Belt and braces: litellm is supposed to forward `extra_headers`, but the
    exact behaviour varies by provider shim. Encoding the session in the URL
    path means the driver only has to set `api_base`, which every provider
    honours. Headers still win if both are present.
    """
    parts = path.split("/")
    if len(parts) >= 3 and parts[0] == "sess":
        return "/".join(parts[3:]), parts[1] or None, parts[2] or None
    return path, None, None


@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
async def proxy(path: str, request: Request):
    client: httpx.AsyncClient = app.state.client
    raw = await request.body()
    path, path_session_type, path_session_id = _strip_session_path(path)

    fwd_headers = {
        k: v for k, v in request.headers.items() if k.lower() not in _DROP_REQUEST_HEADERS
    }

    # Non-chat traffic (tokenizer endpoints, /metrics passthrough, model list):
    # forward and don't log.
    is_completion = path.endswith(("chat/completions", "completions")) and request.method == "POST"
    if not is_completion:
        upstream = await client.request(
            request.method, "/" + path, content=raw,
            headers=fwd_headers, params=dict(request.query_params),
        )
        return _plain_response(upstream)

    try:
        body = json.loads(raw) if raw else {}
    except json.JSONDecodeError:
        # Malformed body: still forward it, let upstream decide. Don't log.
        upstream = await client.request(
            request.method, "/" + path, content=raw, headers=fwd_headers
        )
        return _plain_response(upstream)

    session_id = request.headers.get("X-Session-Id") or path_session_id
    session_type = request.headers.get("X-Session-Type") or path_session_type
    hdr_turn = request.headers.get("X-Turn-Number")
    id_source = ("header" if request.headers.get("X-Session-Id")
                 else "path" if path_session_id else None)

    # task_id is the LOGICAL unit of work (one SWE-bench instance). session_id is the
    # process. sequence_id is one conversation -- and a single task/process can hold
    # several: the parent, each delegated sub-agent, and the condenser's summariser.
    # OpenHands runs all of those in ONE process with ONE set of static headers, so
    # sequence_id has to be DERIVED, not declared. It's the hash of the conversation
    # root; see compute_hashes.
    task_id = request.headers.get("X-Task-Id") or session_id
    run_id = request.headers.get("X-Run-Id")  # routes the log file; None -> default

    h = compute_hashes(body)
    sequence_id = h["sequence_root"]

    # Turns are counted per SEQUENCE, not per session -- otherwise a parent and its
    # four sub-agents would share one interleaved counter and every turn number in
    # the run would be meaningless.
    if hdr_turn is not None:
        try:
            turn_number = int(hdr_turn)
        except ValueError:
            turn_number = None
    elif sequence_id:
        turn_number = await app.state.turns.next(f"{task_id}:{sequence_id}")
    else:
        turn_number = None

    record: dict[str, Any] = {
        "request_id": str(uuid.uuid4()),
        "run_id": run_id,
        "task_id": task_id,
        "session_id": session_id,
        "sequence_id": sequence_id,
        "session_type": session_type,
        "session_id_source": id_source,
        "turn_number": turn_number,
        "turn_number_source": ("header" if hdr_turn is not None
                               else "proxy" if sequence_id else None),
        "model": body.get("model"),
        "stream": bool(body.get("stream", False)),
        "path": "/" + path,
        **h,
    }

    # Close the predictor's loop before dispatching: this request arriving IS the
    # end of the previous turn's gap. Cheap, and unconditional so the EWMA keeps
    # learning even in arms that emit no hints -- which means a later gap-hint arm
    # starts warm rather than spending its first minutes silent.
    await app.state.hints.observe_gap(run_id, task_id, sequence_id, time.time())
    gap_hint = request.headers.get("X-KV-Hint-Gap") in ("1", "true", "True")

    # --- PLAS: read the program's attained service, stamp it on the call. ---
    # Accumulation happens unconditionally (see _handle_*); only the WRITE is
    # gated, so an FCFS arm still leaves a warm table and the two arms differ
    # in exactly one thing: whether the field is on the wire.
    plas_key = task_id if PLAS_KEY == "task" else sequence_id
    plas_cum = await app.state.plas.service_of(plas_key)
    record["plas_key"] = plas_key
    record["plas_service_cum"] = plas_cum
    record["plas_priority"] = None

    _hdr_plas = request.headers.get("X-PLAS")
    plas_on = (_hdr_plas in ("1", "true", "True")) if _hdr_plas is not None else PLAS_ENABLED

    if plas_on and plas_key:
        p = int(plas_cum // PLAS_QUANTUM) if PLAS_QUANTUM > 1 else int(plas_cum)
        body["priority"] = p
        record["plas_priority"] = p
        raw = json.dumps(body).encode("utf-8")
    elif "priority" in body:
        # Never let a client-supplied priority through in an FCFS arm.
        body.pop("priority")
        raw = json.dumps(body).encode("utf-8")

    if record["stream"]:
        return await _handle_stream(client, path, raw, body, fwd_headers, record,
                                    gap_hint)
    return await _handle_unary(client, path, raw, fwd_headers, record, gap_hint)


def _plain_response(upstream: httpx.Response):
    headers = {
        k: v for k, v in upstream.headers.items() if k.lower() not in _DROP_RESPONSE_HEADERS
    }
    return PlainTextResponse(
        content=upstream.content,
        status_code=upstream.status_code,
        headers=headers,
        media_type=upstream.headers.get("content-type"),
    )


async def _handle_unary(client, path, raw, headers, record, gap_hint=False):
    t0 = time.time()
    try:
        upstream = await client.post("/" + path, content=raw, headers=headers)
    except httpx.HTTPError as exc:
        record.update(
            ts_request_in=t0, ts_first_byte=None, ts_last_byte=time.time(),
            status_code=None, error=repr(exc),
        )
        await app.state.log.write(record)
        return JSONResponse({"error": {"message": str(exc), "type": "proxy_upstream_error"}},
                            status_code=502)

    t_end = time.time()
    payload: dict[str, Any] | None = None
    try:
        payload = upstream.json()
    except Exception:
        pass
    usage = (payload or {}).get("usage") or {}
    resp_id = (payload or {}).get("id")
    tool = _first_tool_name(payload)

    record.update(
        ts_request_in=t0,
        ts_first_byte=t_end,   # unary: no meaningful TTFT, first byte == last byte
        ts_last_byte=t_end,
        status_code=upstream.status_code,
        prompt_tokens=usage.get("prompt_tokens"),
        completion_tokens=usage.get("completion_tokens"),
        cached_tokens=(usage.get("prompt_tokens_details") or {}).get("cached_tokens"),
        usage_source="response" if usage else None,
        stream_chunks=None,
        error=None,
        response_id=resp_id,
        tool_called=tool,
    )
    await app.state.hints.note_response(record.get("run_id"),
                                        record.get("task_id"),
                                        record.get("sequence_id"),
                                        resp_id, tool, t_end)
    await _maybe_gap_hint(gap_hint, resp_id, tool)
    await app.state.plas.complete(record.get("plas_key"), _call_service(usage))
    await app.state.log.write(record)
    return _plain_response(upstream)


async def _maybe_gap_hint(enabled: bool, resp_id: str | None, tool: str | None):
    """Send expect_return_ms for the gap this tool call is about to open.

    Fires only when the header asked for it AND the predictor has enough samples
    for this tool. The wire carries a DURATION, not a decision -- the server maps
    it onto retain/offload against its own thresholds, so those can be swept
    without re-running any agents.
    """
    if not enabled or not resp_id or not tool:
        return
    hints: KVHintState = app.state.hints
    predicted = await hints.predict_gap_ms(tool)
    if predicted is None:
        hints.stats["gap_hint_skipped_cold"] += 1
        return
    hints.stats["gap_hints_sent"] += 1
    await _post_kv_hint(app.state.client,
                        {"request_id": resp_id,
                         "expect_return_ms": round(predicted, 1)},
                        hints.stats)


async def _handle_stream(client, path, raw, body, headers, record, gap_hint=False):
    client_wants_usage = bool((body.get("stream_options") or {}).get("include_usage"))
    send_raw = raw
    if INJECT_USAGE and not client_wants_usage:
        patched = dict(body)
        patched["stream_options"] = {**(body.get("stream_options") or {}), "include_usage": True}
        send_raw = json.dumps(patched).encode("utf-8")
        headers = {k: v for k, v in headers.items() if k.lower() != "content-length"}
    suppress_usage_chunk = INJECT_USAGE and not client_wants_usage

    log = app.state.log
    t0 = time.time()
    state: dict[str, Any] = {"first_byte": None, "chunks": 0, "usage": None,
                             "status": None, "error": None,
                             "resp_id": None, "tool": None}

    async def body_iter():
        buf = b""
        try:
            async with client.stream("POST", "/" + path, content=send_raw,
                                     headers=headers) as upstream:
                state["status"] = upstream.status_code

                if upstream.status_code >= 400:
                    err = await upstream.aread()
                    state["first_byte"] = state["first_byte"] or time.time()
                    state["error"] = err.decode("utf-8", "replace")[:2000]
                    yield err
                    return

                async for raw_chunk in upstream.aiter_raw():
                    if not raw_chunk:
                        continue
                    if state["first_byte"] is None:
                        state["first_byte"] = time.time()
                    buf += raw_chunk

                    # Re-emit at SSE event boundaries so we can inspect (and
                    # optionally drop) individual events without altering bytes.
                    while b"\n\n" in buf:
                        event, buf = buf.split(b"\n\n", 1)
                        obj = _parse_sse_event(event)
                        if obj is not None and obj.get("usage"):
                            state["usage"] = obj["usage"]
                        if obj is not None:
                            # Read-only inspection; the event is re-emitted below
                            # byte-for-byte regardless.
                            if state["resp_id"] is None:
                                state["resp_id"] = obj.get("id")
                            if state["tool"] is None:
                                state["tool"] = _first_tool_name(obj)
                        if suppress_usage_chunk and _is_usage_only_chunk(obj):
                            continue  # injected on the client's behalf; don't leak it
                        state["chunks"] += 1
                        yield event + b"\n\n"
                if buf:
                    yield buf
        except httpx.HTTPError as exc:
            state["error"] = repr(exc)
        finally:
            usage = state["usage"] or {}
            t_end = time.time()
            record.update(
                ts_request_in=t0,
                ts_first_byte=state["first_byte"],
                ts_last_byte=t_end,
                status_code=state["status"],
                prompt_tokens=usage.get("prompt_tokens"),
                completion_tokens=usage.get("completion_tokens"),
                cached_tokens=(usage.get("prompt_tokens_details") or {}).get("cached_tokens"),
                usage_source=("stream_injected" if usage and suppress_usage_chunk
                              else "stream_client" if usage else None),
                stream_chunks=state["chunks"],
                error=state["error"],
                response_id=state["resp_id"],
                tool_called=state["tool"],
            )
            await app.state.hints.note_response(record.get("run_id"),
                                                record.get("task_id"),
                                                record.get("sequence_id"),
                                                state["resp_id"], state["tool"],
                                                t_end)
            await _maybe_gap_hint(gap_hint, state["resp_id"], state["tool"])
            await app.state.plas.complete(record.get("plas_key"),
                                                      _call_service(usage))
            await log.write(record)

    return StreamingResponse(
        body_iter(),
        media_type="text/event-stream",
        headers={"cache-control": "no-cache", "x-accel-buffering": "no"},
    )