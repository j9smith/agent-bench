"""Phase-2 acceptance check, runnable without a GPU.

Boots tools/mock_vllm.py as upstream and proxy/main.py in front of it, then:
  1. streams a completion directly from upstream and through the proxy, and
     asserts the response bodies are byte-identical;
  2. asserts the proxy captured usage anyway (the injected chunk was suppressed,
     not leaked);
  3. asserts one well-formed JSONL line per request, with session headers
     honoured and turn numbers auto-assigned.

Against a real vLLM, run the same comparison by hand -- see README phase 2.
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
LOGDIR = ROOT / "logs" / "_test_verify"
LOG = LOGDIR / "default" / "requests.jsonl"
UPSTREAM, PROXY = 8100, 8101
REQ = {"model": "mock", "stream": True,
       "messages": [{"role": "system", "content": "you are an agent"},
                    {"role": "user", "content": "run the tests"}]}

failures = []


def check(name, cond, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}{'  ' + detail if detail and not cond else ''}")
    if not cond:
        failures.append(name)


def wait(url, timeout=25):
    end = time.time() + timeout
    while time.time() < end:
        try:
            if httpx.get(url, timeout=1).status_code < 500:
                return True
        except Exception:
            time.sleep(0.2)
    return False


def stream_body(url, headers=None):
    out = b""
    with httpx.Client(timeout=30) as c:
        with c.stream("POST", url, json=REQ, headers=headers or {}) as r:
            for chunk in r.iter_raw():
                out += chunk
    return out


def main():
    LOG.parent.mkdir(parents=True, exist_ok=True)
    LOG.unlink(missing_ok=True)
    env = {**os.environ,
           "UPSTREAM_BASE_URL": f"http://127.0.0.1:{UPSTREAM}",
           "PROXY_LOG_DIR": str(LOGDIR),
           "PROXY_INJECT_USAGE": "1",
           "PYTHONPATH": str(ROOT)}

    procs = [
        subprocess.Popen([sys.executable, "-m", "uvicorn", "tools.mock_vllm:app",
                          "--port", str(UPSTREAM), "--log-level", "error"],
                         cwd=ROOT, env=env),
        subprocess.Popen([sys.executable, "-m", "uvicorn", "proxy.main:app",
                          "--port", str(PROXY), "--log-level", "error"],
                         cwd=ROOT, env=env),
    ]
    try:
        assert wait(f"http://127.0.0.1:{UPSTREAM}/v1/models"), "mock upstream did not start"
        assert wait(f"http://127.0.0.1:{PROXY}/healthz"), "proxy did not start"

        direct = stream_body(f"http://127.0.0.1:{UPSTREAM}/v1/chat/completions")
        viaproxy = stream_body(f"http://127.0.0.1:{PROXY}/v1/chat/completions",
                               {"X-Session-Id": "task-001", "X-Session-Type": "agentic"})
        stream_body(f"http://127.0.0.1:{PROXY}/v1/chat/completions",
                    {"X-Session-Id": "task-001", "X-Session-Type": "agentic"})
        stream_body(f"http://127.0.0.1:{PROXY}/v1/chat/completions")  # no headers: must not fail

        time.sleep(0.5)
        rows = [json.loads(l) for l in LOG.read_text().splitlines() if l.strip()]

        print("\nphase 2 acceptance:")
        check("streamed body through proxy is byte-identical to direct",
              direct == viaproxy, f"{len(direct)}B vs {len(viaproxy)}B")
        check("no usage chunk leaked to client", b'"usage"' not in viaproxy)
        check("one log line per request", len(rows) == 3, f"got {len(rows)}")

        r0 = rows[0]
        check("usage captured despite suppression",
              r0.get("prompt_tokens") and r0.get("completion_tokens"),
              json.dumps({k: r0.get(k) for k in ("prompt_tokens", "completion_tokens")}))
        check("timestamps ordered and populated",
              None not in (r0["ts_request_in"], r0["ts_first_byte"], r0["ts_last_byte"])
              and r0["ts_request_in"] <= r0["ts_first_byte"] <= r0["ts_last_byte"])
        check("session headers honoured",
              r0["session_id"] == "task-001" and r0["session_type"] == "agentic")
        check("prefix_hash present and excludes final turn", bool(r0["prefix_hash"]))
        check("turn_number auto-assigned and increments",
              [rows[0]["turn_number"], rows[1]["turn_number"]] == [0, 1])
        check("headerless request logged with nulls, not dropped",
              rows[2]["session_id"] is None and rows[2]["status_code"] == 200)
        check("status_code recorded", all(r["status_code"] == 200 for r in rows))
    finally:
        for p in procs:
            p.terminate()
        for p in procs:
            p.wait(timeout=10)

    print("\n" + ("ALL CHECKS PASSED" if not failures else f"FAILED: {failures}"))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
