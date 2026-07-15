"""Post-run summary of a proxy JSONL log: status breakdown, per-sequence turn
tables, context-growth summary. Read-only; the export step is separate."""
import collections
import json
import sys
from pathlib import Path

if len(sys.argv) < 2:
    print("usage: show_run.py <requests.jsonl>", file=sys.stderr)
    sys.exit(1)

rows = []
for line in Path(sys.argv[1]).read_text().splitlines():
    line = line.strip()
    if line:
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            pass

if not rows:
    print("empty log")
    sys.exit(0)

status = collections.Counter(r.get("status_code") for r in rows)
ok = [r for r in rows if r.get("status_code") == 200]
print(f"\n{len(rows)} requests   status: {dict(status)}   ({len(ok)} ok)")

by_type = collections.Counter(r.get("session_type") for r in ok)
print(f"successful by type: {dict(by_type)}")

# group successful turns by sequence
seqs = collections.defaultdict(list)
for r in ok:
    seqs[(r.get("task_id"), r.get("sequence_id"))].append(r)

print(f"\n{len(seqs)} sequence(s):")
for (task, seq), turns in list(seqs.items())[:12]:
    turns.sort(key=lambda r: (r.get("turn_number") or 0))
    prompts = [t.get("prompt_tokens") for t in turns if t.get("prompt_tokens")]
    label = f"{task or '?'}"
    if len(label) > 40:
        label = "…" + label[-39:]
    print(f"\n  {label}  seq={str(seq)[:8]}  {len(turns)} turns")
    print(f"    {'turn':>4} {'prompt':>7} {'compl':>6} {'e2e_s':>6}")
    for t in turns[:30]:
        p = t.get("prompt_tokens")
        c = t.get("completion_tokens")
        rq, lb = t.get("ts_request_in"), t.get("ts_last_byte")
        e2e = f"{lb - rq:6.1f}" if (lb and rq) else "     —"
        print(f"    {str(t.get('turn_number')):>4} "
              f"{(str(p) if p is not None else '—'):>7} "
              f"{(str(c) if c is not None else '—'):>6} {e2e}")
    if len(prompts) >= 2:
        growth = (prompts[-1] - prompts[0]) / max(len(prompts) - 1, 1)
        print(f"    context growth: {prompts[0]} -> {prompts[-1]} "
              f"(~{growth:+.0f} tokens/turn)")

if len(seqs) > 12:
    print(f"\n  ... and {len(seqs) - 12} more sequences")

errs = [r for r in rows if r.get("status_code") not in (200, None)]
if errs:
    print(f"\n{len(errs)} errors. sample:")
    for r in errs[:3]:
        print(f"  turn {r.get('turn_number')} status {r.get('status_code')}: "
              f"{(r.get('error') or '')[:80]}")
