"""Live per-turn table. Reads proxy JSONL from stdin (via `tail -f`).

None-safe: turns that errored (no token counts) print with dashes and their
status code, so failures are visible rather than crashing the watcher.
"""
import json
import sys

HDR = f"{'type':>8} {'turn':>4} {'prompt':>7} {'compl':>6} {'e2e_s':>6} {'stat':>4}  note"
print(HDR)
print("-" * len(HDR))

for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    try:
        d = json.loads(line)
    except json.JSONDecodeError:
        continue
    p = d.get("prompt_tokens")
    c = d.get("completion_tokens")
    p = f"{p:>7}" if p is not None else f"{'—':>7}"
    c = f"{c:>6}" if c is not None else f"{'—':>6}"
    fb, rq, lb = d.get("ts_first_byte"), d.get("ts_request_in"), d.get("ts_last_byte")
    e2e = f"{lb - rq:6.1f}" if (lb and rq) else f"{'—':>6}"
    stat = d.get("status_code")
    note = ""
    if stat and stat != 200:
        note = (d.get("error") or "")[:48]
    elif d.get("stream_chunks") is None:
        note = "non-streamed"
    typ = (d.get("session_type") or "?")[:8]
    print(f"{typ:>8} {str(d.get('turn_number')):>4} {p} {c} {e2e} {str(stat):>4}  {note}",
          flush=True)
