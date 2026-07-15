"""Acceptance for branching, condensation, and sequence separation.

Simulates what OpenHands actually does to the wire: ONE process, ONE set of static
headers (X-Task-Id), containing several conversations at once --

  * a parent agent whose context grows, then SHRINKS when the condenser fires;
  * two delegated sub-agents, each with its OWN context (system prompt + its
    delegated instruction), i.e. NOT a fork of the parent;
  * the condenser's summarisation call, a one-shot with no resumption.

All four carry identical headers. The proxy has to tell them apart anyway, by deriving
sequence_id from each conversation's root. Then the exporter has to reconstruct the
tree from hashes alone and measure how much context the sub-agents actually share with
their parent -- which is the number that decides whether delegation buys the KV cache
anything.
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "logs" / "branch_requests.jsonl"
OUT = ROOT / "analysis" / "branch"
UPSTREAM, PROXY, PROM = 8120, 8121, 8122
SYS = {"role": "system", "content": "You are a coding agent. " + "TOOLS. " * 200}
failures = []


def check(name, cond, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))
    if not cond:
        failures.append(name)


def wait(url, t=25):
    end = time.time() + t
    while time.time() < end:
        try:
            if httpx.get(url, timeout=1).status_code < 500:
                return True
        except Exception:
            time.sleep(0.2)
    return False


def call(msgs, task_id):
    """Every call carries the SAME static headers -- as OpenHands would."""
    h = {"X-Task-Id": task_id, "X-Session-Id": task_id, "X-Session-Type": "agentic"}
    with httpx.Client(timeout=30) as c:
        with c.stream("POST", f"http://127.0.0.1:{PROXY}/v1/chat/completions",
                      json={"model": "mock", "stream": True, "messages": msgs},
                      headers=h) as r:
            for _ in r.iter_raw():
                pass


def main():
    LOG.parent.mkdir(parents=True, exist_ok=True)
    LOG.unlink(missing_ok=True)
    env = {**os.environ, "UPSTREAM_BASE_URL": f"http://127.0.0.1:{UPSTREAM}",
           "PROXY_LOG_PATH": str(LOG), "PYTHONPATH": str(ROOT)}
    procs = [
        subprocess.Popen([sys.executable, "-m", "uvicorn", "tools.mock_vllm:app",
                          "--port", str(UPSTREAM), "--log-level", "error"], cwd=ROOT, env=env),
        subprocess.Popen([sys.executable, "-m", "uvicorn", "proxy.main:app",
                          "--port", str(PROXY), "--log-level", "error"], cwd=ROOT, env=env),
        subprocess.Popen([sys.executable, "-m", "uvicorn", "tools.fake_prom:app",
                          "--port", str(PROM), "--log-level", "error"],
                         cwd=ROOT, env={**env, "FAKE_ENGINE": "vllm"}),
    ]
    task = "django__django-11099"
    try:
        assert wait(f"http://127.0.0.1:{PROXY}/healthz")

        # --- parent agent: context grows over 5 turns
        parent = [SYS, {"role": "user", "content": "ISSUE: the regex is wrong"}]
        for k in range(5):
            call(parent, task)
            parent += [{"role": "assistant", "content": f"step {k}"},
                       {"role": "user", "content": "<out>" + "x" * 600 * (k + 1) + "</out>"}]
            time.sleep(0.15)

        # --- condenser fires: a one-shot summarisation call (its own conversation)
        call([{"role": "system", "content": "Summarize the conversation so far."},
              {"role": "user", "content": "EVENTS: " + "y" * 3000}], task)

        # --- parent resumes with a REWRITTEN, SHORTER history. Root is preserved
        #     (keep_first=2), so it is still the same sequence -- but the prefix the
        #     server cached is now gone.
        parent = [SYS, parent[1],
                  {"role": "assistant", "content": "SUMMARY: tried A, B failed"},
                  {"role": "user", "content": "continue"}]
        for k in range(2):
            call(parent, task)
            parent += [{"role": "assistant", "content": f"post-condense {k}"},
                       {"role": "user", "content": "<out>" + "z" * 400 + "</out>"}]
            time.sleep(0.15)

        # --- two delegated sub-agents. ISOLATED contexts: same system prompt, but
        #     their own first user message. NOT a fork of the parent's history.
        for sub in ("find the failing test", "write the patch"):
            child = [SYS, {"role": "user", "content": f"SUBTASK: {sub}"}]
            for k in range(3):
                call(child, task)
                child += [{"role": "assistant", "content": f"sub step {k}"},
                          {"role": "user", "content": "<out>" + "q" * 500 + "</out>"}]
                time.sleep(0.1)

        time.sleep(0.6)
        r = subprocess.run([sys.executable, "scripts/export_metrics.py", "--engine", "vllm",
                            "--requests", str(LOG), "--prometheus", f"http://127.0.0.1:{PROM}",
                            "--out", str(OUT), "--step", "15"],
                           cwd=ROOT, env=env, capture_output=True, text=True)
        print(r.stdout.strip())
        if r.returncode != 0:
            print(r.stderr, file=sys.stderr)
        df = pd.read_csv(OUT.with_suffix(".csv"))

        roots = df[df.sequence_role == "root"]
        subs = df[df.sequence_role == "subagent"]
        ones = df[df.sequence_role == "oneshot"]

        print("\nbranching acceptance:")
        check("export exited 0", r.returncode == 0)
        check("one task, despite 4 conversations", df.task_id.nunique() == 1)
        check("4 sequences separated from identical headers",
              df.sequence_id.nunique() == 4, f"got {df.sequence_id.nunique()}")
        check("fanout reported as 4", set(df.fanout) == {4})
        check("parent identified as root", roots.sequence_id.nunique() == 1)
        check("2 sub-agents identified", subs.sequence_id.nunique() == 2,
              f"got {subs.sequence_id.nunique()}")
        check("condenser summariser identified as one-shot",
              ones.sequence_id.nunique() == 1)
        check("turn numbers are per-sequence, not interleaved",
              sorted(roots.turn_number.tolist()) == [0, 1, 2, 3, 4, 5, 6])

        # THE measurement: does delegation give the KV cache anything?
        #
        # The honest metric is DEPTH, not fraction. Each sub-agent shares exactly one
        # message with its parent -- the system prompt -- and not a single token of the
        # parent's actual work. That is what "isolated context" means on the wire.
        check("sub-agent shares exactly the system message with the parent, nothing more",
              set(subs.shared_prefix_msgs) == {1}, f"depth={set(subs.shared_prefix_msgs)}")
        check("shared bytes are CONSTANT across a sub-agent's turns (a fixed prompt, "
              "not a growing fork)",
              subs.groupby("sequence_id").shared_prefix_chars.nunique().eq(1).all())

        # And therefore: the shared fraction DECAYS. The reusable prefix is a fixed
        # cost while each sub-agent's own context grows without bound, so whatever
        # amortisation the fan-out offers tends to zero as the task proceeds. Anyone
        # quoting a single "sub-agents share X% of context" number is quoting a
        # function of how far into the trajectory they happened to look.
        first_last = subs.sort_values("turn_number").groupby("sequence_id") \
                         .shared_prefix_frac.agg(["first", "last"])
        check("shared FRACTION decays as the sub-agent's own context grows",
              (first_last["last"] < first_last["first"]).all(),
              f"{first_last['first'].median():.0%} -> {first_last['last'].median():.0%}")

        # condensation
        cond = df[df.is_condensation]
        check("condensation detected: parent context shrank", len(cond) == 1,
              f"got {len(cond)}")
        check("condensation happened in the ROOT sequence",
              (cond.sequence_role == "root").all())
        check("shrink magnitude recorded",
              cond.context_shrink_tokens.notna().all()
              and (cond.context_shrink_tokens > 0).all())
        check("post-condensation prefix is NOT in the cache (rewritten history)",
              not cond.iloc[0].prefix_seen_before)
        check("parent context grows monotonically apart from the condensation",
              (roots[~roots.is_condensation].context_growth_tokens.dropna() > 0).all())
    finally:
        for p in procs:
            p.terminate()
        for p in procs:
            p.wait(timeout=10)
    print("\n" + ("ALL CHECKS PASSED" if not failures else f"FAILED: {failures}"))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
