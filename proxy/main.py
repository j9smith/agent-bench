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
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import time
import uuid
from collections import defaultdict
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse

UPSTREAM_BASE_URL = os.environ.get("UPSTREAM_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
INJECT_USAGE = os.environ.get("PROXY_INJECT_USAGE", "1") not in ("0", "false", "False")
REQUEST_TIMEOUT = float(os.environ.get("PROXY_TIMEOUT_S", "1800"))

# Hop-by-hop headers we must not forward.
_DROP_REQUEST_HEADERS = {"host", "content-length", "connection", "accept-encoding"}
_DROP_RESPONSE_HEADERS = {"content-length", "content-encoding", "transfer-encoding", "connection"}


LOG_DIR = Path(os.environ.get("PROXY_LOG_DIR", "logs"))
DEFAULT_RUN = os.environ.get("PROXY_DEFAULT_RUN", "default")


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

    if record["stream"]:
        return await _handle_stream(client, path, raw, body, fwd_headers, record)
    return await _handle_unary(client, path, raw, fwd_headers, record)


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


async def _handle_unary(client, path, raw, headers, record):
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
    usage: dict[str, Any] = {}
    try:
        usage = (upstream.json() or {}).get("usage") or {}
    except Exception:
        pass

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
    )
    await app.state.log.write(record)
    return _plain_response(upstream)


async def _handle_stream(client, path, raw, body, headers, record):
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
                             "status": None, "error": None}

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
            record.update(
                ts_request_in=t0,
                ts_first_byte=state["first_byte"],
                ts_last_byte=time.time(),
                status_code=state["status"],
                prompt_tokens=usage.get("prompt_tokens"),
                completion_tokens=usage.get("completion_tokens"),
                cached_tokens=(usage.get("prompt_tokens_details") or {}).get("cached_tokens"),
                usage_source=("stream_injected" if usage and suppress_usage_chunk
                              else "stream_client" if usage else None),
                stream_chunks=state["chunks"],
                error=state["error"],
            )
            await log.write(record)

    return StreamingResponse(
        body_iter(),
        media_type="text/event-stream",
        headers={"cache-control": "no-cache", "x-accel-buffering": "no"},
    )
