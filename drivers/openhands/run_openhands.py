"""Agentic workload driver #2: OpenHands, with delegation and condensation as toggles.

Why this exists alongside mini-swe-agent
----------------------------------------
mini-swe-agent has a strictly linear history and no context management: context grows
monotonically, and what you measure is the *unmanaged* shape of the workload.

OpenHands does two things to that shape, and both are switchable, which is the whole
point of this driver:

  --condenser on   LLMSummarizingCondenser (the SDK default) drops old events and
                   replaces them with an LLM-written summary once history gets long.
                   Context stops growing monotonically and becomes a sawtooth. Note
                   what this *is*: the scaffold performing eviction in userspace,
                   because the serving stack gives it no way to express retention.
                   And every time it fires it rewrites the prefix, so the server's
                   prefix cache misses. OpenHands' own benchmark found the condenser
                   cost $40 MORE on SWE-bench Verified than the no-condensation
                   baseline, attributed to lower prompt cache utilisation, while
                   flattening latency. That trade is this project's thesis in one
                   line, and this flag is how you measure it.

  --delegation on  DelegateTool lets the agent spawn sub-agents. Each sub-agent gets
                   its OWN context -- that isolation is the point of delegating -- so
                   fan-out produces N independent sequences sharing only the system
                   prompt, NOT a deep fork of the parent. Meanwhile the parent's KV
                   sits idle for the whole delegation with resumption probability ~1.

Everything below is stock SDK. No forks, no patches.

Two known-and-accepted limitations, so they don't surprise you in the data:

  * --condenser off does NOT mean "no context management". If a request exceeds the
    context window, OpenHands' controller still truncates history as a fallback. Set
    --max-len high enough that tasks don't reach it, and check the data for the
    signature (a sudden halving of prompt_tokens) rather than assuming it never fired.
  * The condenser's summarisation calls are LLM calls too, and they go through the
    proxy. They have a completely different shape -- one-shot, no growth, no
    resumption. The proxy separates them out via the derived sequence_id, so they
    don't pollute the agentic distribution. They are labelled in the exporter.
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LOGDIR = ROOT / "logs" / "openhands"

# Run each task in its own subprocess. Same reason as the mini-swe-agent driver:
# LLM.extra_headers is fixed for the life of an LLM object, so a single process
# running N tasks would stamp every one of them with the same X-Task-Id.
WORKER = r'''
import json, os, sys, traceback
from pathlib import Path

from openhands.sdk import LLM, Agent, Conversation
from openhands.sdk.context.condenser import LLMSummarizingCondenser, NoOpCondenser
from openhands.sdk.event.condenser import Condensation

cfg = json.loads(sys.argv[1])
task_id, workdir = cfg["task_id"], cfg["workdir"]

llm = LLM(
    usage_id="agent",
    model=f"hosted_vllm/{cfg['model']}",
    base_url=cfg["api_base"],
    api_key=cfg["api_key"],
    # Static for the process -- which is exactly why task_id lives here and
    # sequence_id is derived by the proxy from the conversation root instead.
    extra_headers={
        "X-Task-Id": task_id,
        "X-Session-Id": task_id,
        "X-Session-Type": "agentic",
        "X-Run-Id": cfg["run_id"],
    },
)

# Condenser toggle. NoOpCondenser is a real class in the SDK, not a None sentinel.
if cfg["condenser"]:
    condenser = LLMSummarizingCondenser(
        llm=llm.model_copy(update={"usage_id": "condenser"}),
        max_size=cfg["condenser_max_size"],
        keep_first=2,
    )
else:
    condenser = NoOpCondenser()

# Tool set. The delegation toggle is literally "is DelegateTool registered".
from openhands.tools.preset.default import get_default_tools, register_builtins_agents
if cfg["delegation"]:
    register_builtins_agents(enable_browser=False)
tools = get_default_tools(enable_browser=False)

agent = Agent(llm=llm, tools=tools, condenser=condenser)

# Count condensation events and delegate calls from the event stream. This is the
# ground truth for "did the toggle actually do anything", independent of anything we
# infer from the proxy log -- which matters, because the most likely way this
# experiment fails is the model simply never calling DelegateTool.
counts = {"condensations": 0, "delegations": 0, "llm_messages": 0}

def on_event(event):
    if isinstance(event, Condensation):
        counts["condensations"] += 1
    name = type(event).__name__
    if "Action" in name and "delegate" in json.dumps(
            getattr(event, "model_dump", lambda: {})(), default=str).lower():
        counts["delegations"] += 1
    counts["llm_messages"] += 1

result = {"task_id": task_id, "exit": "ok"}
try:
    conversation = Conversation(agent=agent, workspace=workdir, callbacks=[on_event])
    conversation.send_message(cfg["task_text"])
    conversation.run()
except Exception as exc:
    result.update(exit="error", error=repr(exc), traceback=traceback.format_exc())

result["counts"] = counts
Path(cfg["result_path"]).write_text(json.dumps(result, indent=2))
print(json.dumps(result["counts"]))
'''


def swebench_instances(subset: str, split: str, n: int, shuffle: bool,
                       diverse_repos: bool = True) -> list[dict]:
    from datasets import load_dataset
    name = {"lite": "princeton-nlp/SWE-bench_Lite",
            "verified": "princeton-nlp/SWE-bench_Verified",
            "full": "princeton-nlp/SWE-bench"}.get(subset, subset)
    ds = load_dataset(name, split=split)
    rows = [{"instance_id": r["instance_id"], "problem_statement": r["problem_statement"],
             "repo": r["repo"], "base_commit": r["base_commit"]} for r in ds]
    rows.sort(key=lambda r: r["instance_id"])
    if shuffle:
        import random
        random.Random(42).shuffle(rows)

    if n <= 0:
        return rows

    if not diverse_repos:
        # old behaviour: first n after sort/shuffle. Tends to over-sample whichever
        # repo sorts first (astropy, django), correlating the workload.
        return rows[:n]

    # Round-robin across repos so N tasks spread over as many distinct codebases as
    # possible. Picking the alphabetical head gives 6 astropy + 2 django; this gives
    # one from each repo before taking a second from any. De-correlates the workload:
    # different repos -> different context structure, different prefix trees, no
    # accidental cross-sequence cache sharing.
    from collections import defaultdict, deque
    by_repo: dict[str, deque] = defaultdict(deque)
    for r in rows:
        by_repo[r["repo"]].append(r)
    queues = list(by_repo.values())
    picked: list[dict] = []
    i = 0
    while len(picked) < n and any(queues):
        q = queues[i % len(queues)]
        if q:
            picked.append(q.popleft())
        i += 1
        # drop empty queues so we don't spin forever
        if i % len(queues) == 0:
            queues = [q for q in queues if q]
            i = 0
    return picked


def prepare_workspace(inst: dict, root: Path) -> Path:
    """Check out the repo at the task's base commit. Cheap, and avoids depending on
    the SWE-bench Docker images being buildable, which is the flakiest part of the
    whole stack. Tests still run -- they just run in this checkout.

    Disk: a full clone of e.g. django is ~2.9GB of history for a single commit we
    actually want. At concurrency 24 that fills the volume. So we fetch ONLY the
    base commit at depth 1 (~150MB), which is the one thing SWE-bench needs. 20x
    smaller, and faster over a slow link.
    """
    wd = root / inst["instance_id"]
    if wd.exists():
        return wd
    wd.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://github.com/{inst['repo']}.git"
    commit = inst["base_commit"]
    # init + fetch just the one commit at depth 1, rather than clone-everything.
    subprocess.run(["git", "init", "--quiet", str(wd)], check=True, timeout=60)
    subprocess.run(["git", "-C", str(wd), "remote", "add", "origin", url],
                   check=True, timeout=60)
    try:
        subprocess.run(["git", "-C", str(wd), "fetch", "--quiet", "--depth", "1",
                        "origin", commit], check=True, timeout=900)
        subprocess.run(["git", "-C", str(wd), "checkout", "--quiet", "FETCH_HEAD"],
                       check=True, timeout=300)
    except subprocess.CalledProcessError:
        # Some servers won't let you fetch an arbitrary SHA directly. Fall back to a
        # shallow clone of the default branch, then check the commit out (still far
        # smaller than a full clone if the commit is recent-ish).
        import shutil
        shutil.rmtree(wd, ignore_errors=True)
        subprocess.run(["git", "clone", "--quiet", "--depth", "50", url, str(wd)],
                       check=True, timeout=900)
        subprocess.run(["git", "-C", str(wd), "checkout", "--quiet", commit],
                       check=True, timeout=300)
    return wd


def run_one(inst: dict, args, logdir: Path) -> dict:
    iid = inst["instance_id"]
    out = logdir / iid
    out.mkdir(parents=True, exist_ok=True)
    wd = prepare_workspace(inst, logdir / "_workspaces")

    base = args.proxy_base_url.rstrip("/")
    cfg = {
        "task_id": iid,
        "run_id": args.run_id,
        "workdir": str(wd),
        "model": args.model,
        # Session also encoded in the path, in case litellm drops extra_headers.
        "api_base": f"{base}/sess/agentic/{iid}/v1",
        "api_key": os.environ.get("PROXY_API_KEY", "dummy"),
        "condenser": args.condenser,
        "condenser_max_size": args.condenser_max_size,
        "delegation": args.delegation,
        "task_text": inst["problem_statement"],
        "result_path": str(out / "result.json"),
    }

    t0 = time.time()
    with (out / "stdout.log").open("w") as fh:
        p = subprocess.run([sys.executable, "-c", WORKER, json.dumps(cfg)],
                           stdout=fh, stderr=subprocess.STDOUT,
                           timeout=args.task_timeout,
                           env={**os.environ, "LLM_API_KEY": cfg["api_key"]})
    rec = {"instance_id": iid, "returncode": p.returncode,
           "wall_s": round(time.time() - t0, 2)}
    try:
        rec["counts"] = json.loads((out / "result.json").read_text())["counts"]
    except Exception:
        rec["counts"] = None
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--num-tasks", type=int, default=8)
    ap.add_argument("--subset", default="lite")
    ap.add_argument("--split", default="test")
    ap.add_argument("--model", default=os.environ.get("MODEL", "Qwen/Qwen3-8B"))
    ap.add_argument("--run-id", default="default")
    ap.add_argument("--proxy-base-url",
                    default=os.environ.get("PROXY_BASE_URL", "http://127.0.0.1:9000"))
    ap.add_argument("--task-timeout", type=int, default=3600)
    ap.add_argument("--logdir", type=Path, default=DEFAULT_LOGDIR)
    ap.add_argument("--shuffle", action="store_true")

    # The two toggles. Both default OFF, so the default OpenHands run is the closest
    # available apples-to-apples with mini-swe-agent: linear, unmanaged, no fan-out.
    ap.add_argument("--delegation", action="store_true",
                    help="register DelegateTool: the agent can spawn sub-agents")
    ap.add_argument("--condenser", action="store_true",
                    help="enable LLMSummarizingCondenser (the SDK default; off here)")
    ap.add_argument("--condenser-max-size", type=int, default=80,
                    help="events before condensation fires")
    args = ap.parse_args()

    try:
        import openhands.sdk  # noqa: F401
    except ImportError:
        print("pip install openhands-sdk openhands-tools", file=sys.stderr)
        return 2
    if not shutil.which("git"):
        print("git not on PATH", file=sys.stderr)
        return 2

    args.logdir.mkdir(parents=True, exist_ok=True)
    insts = swebench_instances(args.subset, args.split, args.num_tasks, args.shuffle)
    print(f"[openhands] {len(insts)} tasks, concurrency={args.concurrency}, "
          f"delegation={'ON' if args.delegation else 'off'}, "
          f"condenser={'ON' if args.condenser else 'off'}, model={args.model}")

    q: queue.Queue = queue.Queue()
    for i in insts:
        q.put(i)
    results, lock = [], threading.Lock()

    def worker():
        while True:
            try:
                inst = q.get_nowait()
            except queue.Empty:
                return
            try:
                rec = run_one(inst, args, args.logdir)
            except Exception as exc:
                rec = {"instance_id": inst["instance_id"], "returncode": "error",
                       "error": repr(exc), "counts": None}
            with lock:
                results.append(rec)
                c = rec.get("counts") or {}
                print(f"[openhands] {len(results)}/{len(insts)} {rec['instance_id']} "
                      f"rc={rec['returncode']} delegations={c.get('delegations')} "
                      f"condensations={c.get('condensations')}", flush=True)

    threads = [threading.Thread(target=worker, daemon=True)
               for _ in range(min(args.concurrency, len(insts)))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    (args.logdir / "summary.json").write_text(json.dumps(
        {"delegation": args.delegation, "condenser": args.condenser,
         "results": results}, indent=2))

    dels = sum((r.get("counts") or {}).get("delegations", 0) for r in results)
    cons = sum((r.get("counts") or {}).get("condensations", 0) for r in results)
    print(f"[openhands] done. {dels} delegate calls, {cons} condensation events "
          f"across {len(results)} tasks")
    if args.delegation and dels == 0:
        print("[openhands] WARNING: delegation was ON and the model never used it. "
              "The two arms of your experiment are identical. Do not sweep on this.")
    if args.condenser and cons == 0:
        print("[openhands] WARNING: condenser was ON and never fired. Tasks are too "
              "short, or --condenser-max-size is too high.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
