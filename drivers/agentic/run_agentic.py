"""Agentic workload driver: SWE-bench Lite via mini-swe-agent, through the proxy.

Why one process per task instead of `mini-extra swebench --workers N`
---------------------------------------------------------------------
The batch runner has its own worker pool, which would be the obvious thing to
use. But mini-swe-agent passes `model.model_kwargs` straight through to
`litellm.completion`, and those kwargs are fixed for the life of the process.
Any `extra_headers` we set there would therefore stamp *every* task in the batch
with the same X-Session-Id, which destroys the per-session grouping the whole
measurement depends on.

So: one `mini-extra swebench --filter '^(instance_id)$'` process per task, with
a generated config carrying that task's session identity, and our own pool at
--concurrency. The scaffold is unmodified; we only write config files and set
`api_base`. Concurrency semantics are unchanged (N tasks in flight).

Session identity is carried two ways, belt and braces:
  - `extra_headers` in model_kwargs   (the intended path)
  - the api_base URL path             (works even if a provider shim drops headers)
The proxy prefers the header and falls back to the path.
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

import yaml

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LOGDIR = ROOT / "logs" / "agentic"


def load_instances(subset: str, split: str, num_tasks: int, shuffle: bool) -> list[str]:
    from datasets import load_dataset  # imported lazily: heavy, and only needed here

    name = {"lite": "princeton-nlp/SWE-bench_Lite",
            "verified": "princeton-nlp/SWE-bench_Verified",
            "full": "princeton-nlp/SWE-bench"}.get(subset, subset)
    ds = load_dataset(name, split=split)
    ids = sorted(ds["instance_id"])
    if shuffle:
        import random
        random.Random(42).shuffle(ids)
    return ids[:num_tasks] if num_tasks > 0 else ids


def write_task_config(instance_id: str, args, cfgdir: Path) -> Path:
    """A per-task config layered on top of mini-swe-agent's builtin swebench.yaml.

    Only the `model` block is set. Everything about how the agent thinks, when it
    stops, and how it calls tools is left as shipped -- we are measuring the
    scaffold, not tuning it.
    """
    base = args.proxy_base_url.rstrip("/")
    api_base = f"{base}/sess/agentic/{instance_id}/v1"
    cfg = {
        "model": {
            "model_name": f"hosted_vllm/{args.model}",
            "model_kwargs": {
                "api_base": api_base,
                "api_key": os.environ.get("PROXY_API_KEY", "dummy"),
                "drop_params": True,
                "temperature": args.temperature,
                "extra_headers": {
                    "X-Session-Id": instance_id,
                    "X-Session-Type": "agentic",
                },
            },
        },
    }
    path = cfgdir / f"{instance_id}.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return path


def run_one(instance_id: str, args, cfgdir: Path, logdir: Path) -> dict:
    cfg = write_task_config(instance_id, args, cfgdir)
    out = logdir / instance_id
    out.mkdir(parents=True, exist_ok=True)

    cmd = [
        "mini-extra", "swebench",
        "--subset", args.subset,
        "--split", args.split,
        "--filter", f"^({instance_id})$",
        "--workers", "1",
        "--output", str(out),
        "--environment-class", args.environment_class,
        "-c", str(cfg),
    ]
    env = {
        **os.environ,
        # Local models aren't in litellm's price registry; without this the run
        # dies on cost accounting, which we don't care about.
        "MSWEA_COST_TRACKING": "ignore_errors",
        "LITELLM_MODEL_REGISTRY_PATH": str(ROOT / "drivers" / "agentic" / "registry.json"),
    }

    t0 = time.time()
    with (out / "stdout.log").open("w") as fh:
        p = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env,
                           timeout=args.task_timeout)
    rec = {"instance_id": instance_id, "returncode": p.returncode,
           "wall_s": round(time.time() - t0, 2),
           "started": t0, "finished": time.time()}
    (out / "result.json").write_text(json.dumps(rec, indent=2))
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--concurrency", type=int, default=8,
                    help="tasks in flight; tune freely, no code changes needed")
    ap.add_argument("--num-tasks", type=int, default=8, help="0 = whole subset")
    ap.add_argument("--subset", default="lite")
    ap.add_argument("--split", default="test")
    ap.add_argument("--model", default=os.environ.get("MODEL", "Qwen/Qwen3-8B"),
                    help="model id as served by vLLM (no provider prefix)")
    ap.add_argument("--proxy-base-url", default=os.environ.get("PROXY_BASE_URL",
                                                               "http://127.0.0.1:9000"))
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--task-timeout", type=int, default=3600)
    ap.add_argument("--logdir", type=Path, default=DEFAULT_LOGDIR)
    ap.add_argument("--shuffle", action="store_true")
    ap.add_argument("--environment-class", default="docker", choices=["docker", "local"],
                    help="'docker' is the real SWE-bench harness. 'local' executes the "
                         "agent's bash DIRECTLY on this machine -- no Docker needed, which "
                         "matters on RunPod where docker-in-docker usually isn't available. "
                         "Only use it on a disposable pod: the agent runs arbitrary commands.")
    args = ap.parse_args()

    if not shutil.which("mini-extra"):
        print("mini-extra not on PATH. `pip install mini-swe-agent`.", file=sys.stderr)
        return 2
    if args.environment_class == "docker" and not shutil.which("docker"):
        print("docker not on PATH. Either use a RunPod template with Docker enabled, or "
              "pass --environment-class local (agent bash runs on THIS machine -- "
              "disposable pods only).", file=sys.stderr)
        return 2
    if args.environment_class == "local":
        print("[agentic] WARNING: --environment-class local. The agent will execute "
              "arbitrary bash on this machine. Disposable pods only.", file=sys.stderr)

    args.logdir.mkdir(parents=True, exist_ok=True)
    cfgdir = args.logdir / "_configs"
    cfgdir.mkdir(exist_ok=True)

    instances = load_instances(args.subset, args.split, args.num_tasks, args.shuffle)
    print(f"[agentic] {len(instances)} instances, concurrency={args.concurrency}, "
          f"proxy={args.proxy_base_url}, model={args.model}")

    q: queue.Queue[str] = queue.Queue()
    for i in instances:
        q.put(i)
    results: list[dict] = []
    lock = threading.Lock()

    def worker():
        while True:
            try:
                iid = q.get_nowait()
            except queue.Empty:
                return
            try:
                rec = run_one(iid, args, cfgdir, args.logdir)
            except subprocess.TimeoutExpired:
                rec = {"instance_id": iid, "returncode": "timeout", "wall_s": args.task_timeout}
            except Exception as exc:  # a broken task must not take the run down
                rec = {"instance_id": iid, "returncode": "error", "error": repr(exc)}
            with lock:
                results.append(rec)
                print(f"[agentic] {len(results)}/{len(instances)} {iid} "
                      f"rc={rec['returncode']} {rec.get('wall_s')}s", flush=True)
            q.task_done()

    threads = [threading.Thread(target=worker, daemon=True)
               for _ in range(min(args.concurrency, len(instances)))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    (args.logdir / "summary.json").write_text(json.dumps(results, indent=2))
    ok = sum(1 for r in results if r["returncode"] == 0)
    print(f"[agentic] done: {ok}/{len(results)} exited 0 "
          f"(task correctness is not the metric; completion is)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
