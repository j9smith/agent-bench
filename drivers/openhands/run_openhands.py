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
import hashlib, json, os, sys, threading, time, traceback
from pathlib import Path

from openhands.sdk import LLM, Agent, Conversation
from openhands.sdk.context.condenser import LLMSummarizingCondenser, NoOpCondenser
from openhands.sdk.event.condenser import Condensation

cfg = json.loads(sys.argv[1])
task_id, workdir = cfg["task_id"], cfg["workdir"]

# Per-run opt-in for proxy-side gap hinting. Sent as a header so an arm switches
# without restarting the long-lived proxy. The proxy is what actually predicts and
# emits; the worker only declares that this run wants it.
_extra_headers = {
    "X-Task-Id": task_id,
    "X-Session-Id": task_id,
    "X-Session-Type": "agentic",
    "X-Run-Id": cfg["run_id"],
}
if cfg.get("kv_hint_gap"):
    _extra_headers["X-KV-Hint-Gap"] = "1"

if cfg.get("plas"):
    _extra_headers["X-PLAS"] = "1"

llm = LLM(
    usage_id="agent",
    model=f"hosted_vllm/{cfg['model']}",
    base_url=cfg["api_base"],
    api_key=cfg["api_key"],
    # Static for the process -- which is exactly why task_id lives here and
    # sequence_id is derived by the proxy from the conversation root instead.
    extra_headers=_extra_headers,
)

ADD_EST = bool(cfg.get("estimate_durations"))
FEEDBACK = bool(cfg.get("feedback_durations"))

# Counters. Ground truth for "did the toggle actually do anything", independent of
# anything inferred from the proxy log. Declared here (rather than after the agent,
# as before) only because the timing sink below writes into it.
#
# spawn and delegate are separate commands on the SAME action type, so counting them
# together -- as the old substring match did -- inflates the number and says nothing
# about WIDTH. Width is the whole point of the delegation arm, so it is counted.
counts = {
    "condensations": 0,
    "llm_messages": 0,
    "spawns": 0,
    "children_spawned": 0,
    "max_fanout_seen": 0,
    "delegations": 0,
    "tasks_delegated": 0,
    "delegate_events_raw": 0,
    "tool_calls_timed": 0,
    "estimates_given": 0,
    "estimates_missing": 0,
}
_counts_lock = threading.Lock()

# Unique per worker process, so re-running the same task under one run-id keeps its
# timing rows separable (timings.jsonl appends; idx restarts each process).
_SESSION = os.urandom(4).hex()

try:
    from openhands.sdk.llm import TextContent
except ImportError:
    from openhands.sdk.llm.message import TextContent

from openhands.sdk.tool.tool import ToolExecutor

# Duration-estimate field, added by dynamic subclassing. This mirrors the SDK's own
# _create_action_type_with_summary (which it applies to EVERY tool, unconditionally),
# including the subclass rescan that recovers a lost cache entry.
#
# Two SDK properties make it work: _get_tool_schema does
# `action_type = action_type or self.action_type`, so the schema is derived LAZILY and
# model_copy() changes what the model sees; and parse_arguments does
# `self.action_type.model_validate(arguments)`, so the value arrives as a typed
# attribute rather than scraped prose.
#
# Field order is post-commitment for free: _prioritize_schema_fields hardcodes
# ("security_risk", "summary") and pydantic appends subclass fields after inherited
# ones, so this lands at the END of the schema. hermes parsing is not constrained
# decoding, so that is a nudge rather than a guarantee.
#
# The description is deliberately NEUTRAL -- no mention of caching, offload or
# retention. The model reports a duration; classification against a transfer RTT is
# downstream. Leaking the downstream use into the prompt would contaminate it.
_EST_FIELD = "estimated_duration_ms"
_EST_DESC = (
    "Your best estimate, in whole milliseconds, of how long THIS specific call "
    "will take to execute -- wall-clock time from dispatch until the result "
    "returns. Choose the other arguments first, then estimate based on what you "
    "have actually decided to run. An integer number of milliseconds."
)

_est_cache = {}
_est_lock = threading.Lock()


def _with_estimate(action_type):
    from pydantic import Field as _PField

    if _EST_FIELD in action_type.model_fields:
        return action_type
    with _est_lock:
        cached = _est_cache.get(action_type)
        if cached is not None:
            return cached
        target = f"{action_type.__name__}WithDurationEstimate"
        for sub in action_type.__subclasses__():
            if sub.__name__ == target:
                _est_cache[action_type] = sub
                return sub
        made = type(
            target,
            (action_type,),
            {
                _EST_FIELD: _PField(description=_EST_DESC),
                "__annotations__": {_EST_FIELD: int},
            },
        )
        _est_cache[action_type] = made
        return made


class _Sink:
    """Thread-safe per-call timing log.

    Thread safety is required: with delegation on, sub-agents execute tools
    concurrently inside this same process.
    """

    def __init__(self, path, task_id, run_id):
        self._path = path
        self._task_id = task_id
        self._run_id = run_id
        self._lock = threading.Lock()
        self._n = 0

    def record(self, action, estimated, actual_ms, is_error):
        try:
            args = action.model_dump()
        except Exception:
            args = {}
        args.pop(_EST_FIELD, None)
        # Args are kept (truncated) because the heuristic/regex control arm scores
        # against exactly these strings, and it is the fallback when the model
        # omits an estimate.
        args_json = json.dumps(args, default=str, sort_keys=True)
        row = {
            "run_id": self._run_id,
            "task_id": self._task_id,
            "session": _SESSION,
            "action_type": type(action).__name__,
            # DelegateExecutor names its threads "Task-<agent_id>", so this
            # separates parent rows from per-sub-agent rows.
            "thread": threading.current_thread().name,
            "estimated_ms": estimated,
            "actual_ms": round(actual_ms, 3),
            "is_error": bool(is_error),
            "args_sha1": hashlib.sha1(args_json.encode()).hexdigest()[:12],
            "args": args_json[:2000],
            "ts": round(time.time(), 3),
        }
        with self._lock:
            self._n += 1
            row["idx"] = self._n
            try:
                with open(self._path, "a") as fh:
                    fh.write(json.dumps(row, default=str) + "\n")
            except Exception:
                pass
        with _counts_lock:
            counts["tool_calls_timed"] += 1
            if estimated is None:
                counts["estimates_missing"] += 1
            else:
                counts["estimates_given"] += 1


_SINK = _Sink(cfg.get("timings_path", os.devnull), task_id, cfg["run_id"])


class _TimingExecutor(ToolExecutor):
    """Measure each tool call, optionally feed the duration back to the model.

    close() and interrupt() are CONCRETE no-ops on ToolExecutor, not abstract, so a
    naive wrapper silently swallows both -- leaking the terminal subprocess and
    breaking interrupt delivery to a running command. Both are forwarded.
    """

    def __init__(self, inner):
        self._inner = inner

    def __call__(self, action, conversation=None):
        estimated = getattr(action, _EST_FIELD, None)
        if isinstance(estimated, bool):
            estimated = None
        t0 = time.perf_counter()
        try:
            obs = self._inner(action, conversation)
        except Exception:
            _SINK.record(action, estimated, (time.perf_counter() - t0) * 1000.0, True)
            raise
        actual_ms = (time.perf_counter() - t0) * 1000.0
        _SINK.record(action, estimated, actual_ms,
                     bool(getattr(obs, "is_error", False)))

        if FEEDBACK:
            # PREPEND: TerminalObservation.to_llm_content runs maybe_truncate() over
            # the assembled text, so a trailing line can be clipped on large output.
            # Mutating the list in place is fine on a frozen model (frozen blocks
            # attribute assignment, not list mutation). This reaches the LLM because
            # TerminalObservation builds from self.text (which concatenates
            # self.content) and FileEditorObservation does not override
            # to_llm_content at all.
            try:
                content = getattr(obs, "content", None)
                if isinstance(content, list):
                    line = "[timing] this call took %d ms" % round(actual_ms)
                    if estimated is not None:
                        line += " (you estimated %d ms)" % int(estimated)
                    content.insert(0, TextContent(text=line + "\n"))
            except Exception:
                pass
        return obs

    def close(self):
        fn = getattr(self._inner, "close", None)
        if callable(fn):
            return fn()

    def interrupt(self):
        fn = getattr(self._inner, "interrupt", None)
        if callable(fn):
            return fn()

    def __getattr__(self, item):
        # Forward anything else to the real executor. Safe from recursion: _inner
        # lives in __dict__.
        return getattr(self._inner, item)


def _instrument(tool_def):
    """Add the estimate field and/or the timing wrapper to one tool definition."""
    upd = {}
    if ADD_EST:
        upd["action_type"] = _with_estimate(tool_def.action_type)
    if tool_def.executor is not None:
        upd["executor"] = _TimingExecutor(tool_def.executor)
    return tool_def.model_copy(update=upd) if upd else tool_def


def _instrumented_class(base_cls):
    """Replacement tool class registered under the SAME name.

    Agent resolves Tool name specs through the global registry, so overwriting the
    registry entry is the only interception point -- and it means sub-agents are
    instrumented too, since their tools resolve through that same registry. Built
    with type() rather than a class statement so each replacement gets a distinct
    __name__ (three classes sharing one __name__ risks collisions in the
    discriminated-union subclass registry).
    """

    def _create(cls, *args, **kwargs):
        return [_instrument(d) for d in base_cls.create(*args, **kwargs)]

    return type(
        "Instrumented" + base_cls.__name__,
        (base_cls,),
        {"name": base_cls.name, "create": classmethod(_create)},
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

# ---------------------------------------------------------------------------
# Tool set.
#
# The old wiring here was a silent no-op: register_builtins_agents() only populates
# the sub-agent FACTORY registry (so get_agent_factory(name=...) resolves) and never
# touches `tools`, so `tools` was byte-identical in both arms. Nor is
# get_default_tools(enable_sub_agents=True) the fix -- that binds TaskToolSet, whose
# children run SEQUENTIALLY, one at a time, blocking: no concurrency multiplier.
#
# openhands-tools 1.36.1 ships DelegateExecutor (real threaded fan-out, enforces
# max_children) but NO ToolDefinition exposing it -- `grep -rn DelegateTool` in the
# package returns nothing. So it is wrapped locally below. Verified working:
# probe gave a fan-out of 2 on an explicit instruction.
# ---------------------------------------------------------------------------
from openhands.tools.preset.default import get_default_tools, register_builtins_agents

# Call this FIRST: it imports the tool modules, which auto-register them. Our
# replacements must overwrite those entries afterwards, not before.
tools = get_default_tools(enable_browser=False)

if ADD_EST or FEEDBACK:
    try:
        from openhands.sdk.tool import register_tool
    except ImportError:
        from openhands.sdk.tool.registry import register_tool
    _instrumented = []
    for _mod, _cls in (
        ("openhands.tools.terminal", "TerminalTool"),
        ("openhands.tools.file_editor", "FileEditorTool"),
        ("openhands.tools.task_tracker", "TaskTrackerTool"),
    ):
        try:
            _base = getattr(__import__(_mod, fromlist=[_cls]), _cls)
        except Exception as _exc:
            print("[worker] WARN could not instrument %s: %r" % (_cls, _exc),
                  file=sys.stderr)
            continue
        register_tool(_base.name, _instrumented_class(_base))
        _instrumented.append(_base.name)
    print("[worker] instrumented tools=%r estimate=%s feedback=%s"
          % (_instrumented, ADD_EST, FEEDBACK), file=sys.stderr, flush=True)

if cfg["delegation"]:
    try:
        from openhands.sdk.tool import (
            Tool, ToolAnnotations, ToolDefinition, register_tool,
        )
    except ImportError:
        from openhands.sdk.tool.tool import ToolAnnotations, ToolDefinition
        from openhands.sdk.tool.spec import Tool
        from openhands.sdk.tool.registry import register_tool

    from openhands.sdk.subagent import get_factory_info
    from openhands.tools.delegate.definition import DelegateAction, DelegateObservation
    from openhands.tools.delegate.impl import DelegateExecutor

    # REQUIRED, not optional: populates the agent factory registry so _spawn_agents
    # can resolve types via get_agent_factory(name=...). On 1.36.1 this registers
    # bash-runner, code-explorer and general-purpose, plus a working "default" alias,
    # so the model may omit agent_types.
    _registered_agents = register_builtins_agents(enable_browser=False)
    _MAXC = int(cfg.get("max_fanout", 3))

    # Prefer an SDK-provided delegate tool if a future version ships one.
    try:
        from openhands.tools.delegate import DelegateTool as _SDKDelegate
    except ImportError:
        _SDKDelegate = None

    if _SDKDelegate is not None:
        _delegate_tool_name = _SDKDelegate.name
        _delegate_source = "sdk"
        tools.append(Tool(name=_delegate_tool_name, params={"max_children": _MAXC}))
    else:
        try:
            _agent_info = get_factory_info()
        except Exception:
            _agent_info = ""

        # The description does real work. The model must learn a two-step protocol,
        # and _spawn_agents ERRORS (rather than clamping) when the cap is exceeded,
        # so an under-specified description burns turns on failed spawns -- or yields
        # a fan-out of one, which looks like a successful run and has no signal.
        _DESC = (
            "Delegate independent subtasks to sub-agents that run IN PARALLEL.\n"
            "\n"
            "Two-step protocol:\n"
            '  1. {"command": "spawn", "ids": ["explore", "test"]}\n'
            '  2. {"command": "delegate", "tasks": {"explore": "...", "test": "..."}}\n'
            "\n"
            "Limits and semantics:\n"
            "  - At most " + str(_MAXC) + " sub-agents may be alive at once. "
            "Spawning beyond that returns an error and wastes a turn, so decide "
            "your fan-out width before calling spawn.\n"
            "  - Spawn 2 or more at once when the work allows it. A single "
            "sub-agent gains nothing over doing the work yourself.\n"
            "  - delegate BLOCKS until every sub-agent finishes, then returns "
            "their combined final responses.\n"
            "  - Each sub-agent starts with a FRESH context and cannot see this "
            "conversation. Every task string must be fully self-contained: name "
            "files, paths and commands explicitly.\n"
            "  - Sub-agents share one working directory. Do not give two of them "
            "edits to the same file.\n"
            "  - agent_types is optional; omit it for the general-purpose agent.\n"
            "\n"
            "Available agent types:\n" + str(_agent_info) + "\n"
            "\n"
            "Use this only when the work splits into genuinely independent pieces "
            "(e.g. explore an unfamiliar module while the test suite runs). Do not "
            "use it for steps that must happen in order."
        )

        try:
            from openhands.sdk.tool import DeclaredResources as _DeclaredResources
        except ImportError:
            try:
                from openhands.sdk.tool.tool import (
                    DeclaredResources as _DeclaredResources,
                )
            except ImportError:
                _DeclaredResources = None

        def _delegate_create(cls, conv_state=None, max_children=None):
            mc = _MAXC if max_children is None else int(max_children)
            action_type = DelegateAction
            if ADD_EST:
                # The fan-out window is itself a tool-call duration, and a
                # right-skewed one (delegate() joins over all children).
                action_type = _with_estimate(action_type)
            executor = DelegateExecutor(max_children=mc)
            if ADD_EST or FEEDBACK:
                executor = _TimingExecutor(executor)
            return [
                cls(
                    action_type=action_type,
                    observation_type=DelegateObservation,
                    description=_DESC,
                    annotations=ToolAnnotations(
                        title="delegate",
                        readOnlyHint=False,
                        destructiveHint=True,
                        idempotentHint=False,
                        openWorldHint=True,
                    ),
                    executor=executor,
                )
            ]

        _attrs = {"create": classmethod(_delegate_create)}
        if _DeclaredResources is not None:
            _attrs["declared_resources"] = (
                lambda self, action: _DeclaredResources(keys=(), declared=True)
            )
        # __init_subclass__ derives
        #   name = _camel_to_snake(cls.__name__).removesuffix("_tool")
        # so "DelegateTool" -> "delegate".
        DelegateTool = type("DelegateTool", (ToolDefinition,), _attrs)
        register_tool(DelegateTool.name, DelegateTool)
        _delegate_tool_name = DelegateTool.name
        _delegate_source = "local-wrapper"
        tools.append(Tool(name=_delegate_tool_name, params={"max_children": _MAXC}))

    # Fail loudly rather than silently producing a run identical to delegation=off.
    if not any(t.name == _delegate_tool_name for t in tools):
        raise RuntimeError(
            "delegation=True but no delegate tool is bound. tools=%r"
            % ([t.name for t in tools],)
        )
    print("[worker] delegation ON via %s, tool=%r, max_fanout=%d, agent_types=%r"
          % (_delegate_source, _delegate_tool_name, _MAXC, _registered_agents),
          file=sys.stderr, flush=True)

agent = Agent(llm=llm, tools=tools, condenser=condenser)

def _delegate_action(event):
    """Return the DelegateAction carried by this event, or None.

    startswith() rather than == because with --estimate-durations the action type is
    the dynamic subclass DelegateActionWithDurationEstimate.
    """
    a = getattr(event, "action", None)
    if a is not None and type(a).__name__.startswith("DelegateAction"):
        return a
    return None


def on_event(event):
    if isinstance(event, Condensation):
        counts["condensations"] += 1

    action = _delegate_action(event)
    if action is not None:
        cmd = getattr(action, "command", None)
        if cmd == "spawn":
            ids = getattr(action, "ids", None) or []
            counts["spawns"] += 1
            counts["children_spawned"] += len(ids)
            if len(ids) > counts["max_fanout_seen"]:
                counts["max_fanout_seen"] = len(ids)
        elif cmd == "delegate":
            tasks = getattr(action, "tasks", None) or {}
            counts["delegations"] += 1
            counts["tasks_delegated"] += len(tasks)
    else:
        # Belt-and-braces: if this SDK version nests the action elsewhere, keep a raw
        # signal so the arm never reports a flat zero by accident. Note this also
        # trips on non-delegation events whose payload merely contains the word, so
        # treat it as an upper bound, not a count.
        try:
            blob = json.dumps(
                getattr(event, "model_dump", lambda: {})(), default=str
            ).lower()
        except Exception:
            blob = ""
        if "delegateaction" in blob or '"delegate' in blob:
            counts["delegate_events_raw"] += 1

    # NB: this increments on EVERY event, not per LLM call. Name kept for continuity
    # with existing result.json consumers; it is an event count.
    counts["llm_messages"] += 1

def _kv_hint_done():
    """Tell the proxy this trajectory is over, so it can mark the KV dead.

    The proxy holds the mapping to vLLM's request_id (which the worker never sees)
    and does the actual POST to /kv_hint. Best-effort by design: a hint is
    advisory, so a failure here must never affect the measured run.
    """
    if not cfg.get("kv_hint_done"):
        return
    import urllib.error
    import urllib.request

    url = cfg["proxy_base"].rstrip("/") + "/_hint/session_done"
    data = json.dumps({"task_id": task_id, "run_id": cfg["run_id"]}).encode()
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            print("[worker] kv_hint done -> %s" % resp.read().decode()[:300],
                  file=sys.stderr, flush=True)
    except Exception as exc:
        print("[worker] kv_hint done POST failed: %r" % exc,
              file=sys.stderr, flush=True)


result = {"task_id": task_id, "exit": "ok"}
try:
    conversation = Conversation(agent=agent, workspace=workdir, callbacks=[on_event])
    conversation.send_message(cfg["task_text"])
    conversation.run()
except Exception as exc:
    result.update(exit="error", error=repr(exc), traceback=traceback.format_exc())
finally:
    # finally, not else: a FAILED trajectory's blocks are corpses too, and leaving
    # them protected is exactly the inversion this hint exists to correct.
    _kv_hint_done()

result["counts"] = counts
Path(cfg["result_path"]).write_text(json.dumps(result, indent=2))
print(json.dumps(result["counts"]))
# Previously the worker exited 0 even after a total failure, so a dead run was
# indistinguishable from a clean one -- and under a sweep the point would be recorded
# as OK in the manifest. Surface it.
if result["exit"] != "ok":
    print(result.get("traceback", result.get("error", "")), file=sys.stderr, flush=True)
    sys.exit(1)
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
        # Fallback for servers that won't fetch an arbitrary SHA directly. Was
        # --depth 50 of the default branch, which drags 50 commits of history --
        # for repos like matplotlib/sphinx that's GBs of test-baseline binaries, and
        # under continuous arrival load with no cleanup it blew the disk quota.
        # Now: depth-1 clone (one commit, not 50) with --filter=blob:none so large
        # blobs come lazily only if touched. Then best-effort checkout of the base
        # commit; if it isn't reachable at depth 1 we stay on the default-branch tip
        # (the workspace only needs to be a plausible repo checkout for the agent to
        # explore -- exact commit fidelity is secondary to not filling the disk).
        shutil.rmtree(wd, ignore_errors=True)
        subprocess.run(["git", "clone", "--quiet", "--depth", "1",
                        "--filter=blob:none", url, str(wd)],
                       check=True, timeout=900)
        try:
            subprocess.run(["git", "-C", str(wd), "fetch", "--quiet", "--depth", "1",
                            "origin", commit], check=True, timeout=300)
            subprocess.run(["git", "-C", str(wd), "checkout", "--quiet", "FETCH_HEAD"],
                           check=True, timeout=120)
        except subprocess.CalledProcessError:
            pass  # keep the shallow default-branch checkout
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
        "max_fanout": args.max_fanout,
        "estimate_durations": args.estimate_durations,
        "feedback_durations": args.feedback_durations,
        # Proxy base WITHOUT the /sess/... suffix: the hint endpoint is proxy-level,
        # not per-session.
        "proxy_base": base,
        "kv_hint_done": args.kv_hint_done,
        "kv_hint_gap": args.kv_hint_gap,
        "task_text": inst["problem_statement"],
        "result_path": str(out / "result.json"),
        "timings_path": str(out / "timings.jsonl"),
    }

    t0 = time.time()
    try:
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
    finally:
        # Remove the repo checkout once the agent is done with it. The measurement
        # data (requests.jsonl, result.json, stdout.log) lives elsewhere and is
        # untouched -- only the multi-hundred-MB working tree is deleted. This is what
        # keeps continuous ARRIVAL runs from accumulating checkouts until the disk
        # fills (the batch path cleaned between sweep points; arrivals had no such
        # boundary). Opt out with args.keep_workspace = True.
        if not getattr(args, "keep_workspace", False):
            shutil.rmtree(wd, ignore_errors=True)


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
                    help="bind a PARALLEL delegate tool: the agent can spawn "
                         "sub-agents that run concurrently")
    ap.add_argument("--max-fanout", type=int, default=3,
                    help="max sub-agents alive at once (DelegateExecutor "
                         "max_children). Exceeding it is an error, not a clamp.")
    ap.add_argument("--estimate-durations", action="store_true",
                    help="ask the model to predict each tool call's duration in ms; "
                         "logs predicted vs measured to <task>/timings.jsonl. NOTE: "
                         "enlarges the tool schema, so token counts are NOT "
                         "comparable with arms collected without it.")
    ap.add_argument("--feedback-durations", action="store_true",
                    help="show the measured duration back to the model in the tool "
                         "observation. With --estimate-durations this means "
                         "estimates are no longer cold priors.")
    ap.add_argument("--condenser", action="store_true",
                    help="enable LLMSummarizingCondenser (the SDK default; off here)")
    ap.add_argument("--condenser-max-size", type=int, default=80,
                    help="events before condensation fires")

    # KV retention hints. Both need a hint-enabled vLLM build; against stock vLLM
    # the proxy's POST 404s, is counted, and changes nothing about the run.
    ap.add_argument("--kv-hint-done", action="store_true",
                    help="POST a dontneed hint when each trajectory finishes. Zero "
                         "estimation error: the client knows this as a fact. Fixes "
                         "LRU's inversion, where a finished trajectory's freed "
                         "blocks become the MOST protected in the pool.")
    ap.add_argument("--kv-hint-gap", action="store_true",
                    help="ask the proxy to send expect_return_ms before each tool "
                         "call, predicted from its own per-tool EWMA of observed "
                         "dispatch gaps. UNTESTED; run --kv-hint-done first.")
    ap.add_argument("--plas", action="store_true",
                help="stamp PLAS program-level attained service as request priority "
                     "(requires vLLM --scheduling-policy priority)")
    args = ap.parse_args()

    try:
        import openhands.sdk  # noqa: F401
    except ImportError:
        print("pip install openhands-sdk openhands-tools", file=sys.stderr)
        return 2
    if not shutil.which("git"):
        print("git not on PATH", file=sys.stderr)
        return 2

    if args.max_fanout < 1:
        print("--max-fanout must be >= 1", file=sys.stderr)
        return 2
    if args.delegation and args.max_fanout < 2:
        print("[openhands] WARNING: --max-fanout 1 permits no parallelism. This arm "
              "measures sequential delegation, not fan-out.", file=sys.stderr)
    if args.estimate_durations and args.feedback_durations:
        print("[openhands] NOTE: estimates are conditioned on earlier measured "
              "durations. This is the feedback arm, not the cold-prior arm. Do not "
              "pool it with --estimate-durations alone.", file=sys.stderr)

    args.logdir.mkdir(parents=True, exist_ok=True)
    insts = swebench_instances(args.subset, args.split, args.num_tasks, args.shuffle)
    print(f"[openhands] {len(insts)} tasks, concurrency={args.concurrency}, "
          f"delegation={'ON' if args.delegation else 'off'}, "
          f"condenser={'ON' if args.condenser else 'off'}, model={args.model}")
    if args.kv_hint_done or args.kv_hint_gap:
        print(f"[openhands] kv hints: done={'ON' if args.kv_hint_done else 'off'} "
              f"gap={'ON' if args.kv_hint_gap else 'off'} "
              f"(via proxy {args.proxy_base_url})")

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
                      f"rc={rec['returncode']} spawns={c.get('spawns')} "
                      f"kids={c.get('children_spawned')} "
                      f"widest={c.get('max_fanout_seen')} "
                      f"timed={c.get('tool_calls_timed')} "
                      f"est={c.get('estimates_given')}/{c.get('estimates_missing')} "
                      f"condensations={c.get('condensations')}", flush=True)

    threads = [threading.Thread(target=worker, daemon=True)
               for _ in range(min(args.concurrency, len(insts)))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    (args.logdir / "summary.json").write_text(json.dumps(
        {"delegation": args.delegation, "max_fanout": args.max_fanout,
         "condenser": args.condenser,
         "estimate_durations": args.estimate_durations,
         "feedback_durations": args.feedback_durations,
         "kv_hint_done": args.kv_hint_done,
         "kv_hint_gap": args.kv_hint_gap,
         "concurrency": args.concurrency, "results": results}, indent=2))

    def total(key):
        return sum((r.get("counts") or {}).get(key, 0) for r in results)

    widest = max([(r.get("counts") or {}).get("max_fanout_seen", 0)
                  for r in results] or [0])
    cons = total("condensations")
    print(f"[openhands] done. {total('spawns')} spawn calls, "
          f"{total('children_spawned')} children spawned, "
          f"{total('delegations')} delegate calls "
          f"({total('tasks_delegated')} tasks), widest single spawn={widest}, "
          f"{cons} condensation events across {len(results)} tasks")

    timed, given = total("tool_calls_timed"), total("estimates_given")
    if args.estimate_durations or args.feedback_durations:
        rate = f"{100 * given / timed:.0f}%" if timed else "n/a"
        print(f"[openhands] timing: {timed} tool calls measured, {given} with an "
              f"estimate (compliance {rate}). rows in "
              f"logs/openhands/<task>/timings.jsonl")

    # Failure modes that LOOK like successful runs.
    if args.delegation and total("children_spawned") == 0:
        print("[openhands] WARNING: delegation was ON and no sub-agent was ever "
              "spawned. The two arms of your experiment are identical. Do not sweep "
              f"on this. (raw delegate-ish events: {total('delegate_events_raw')} -- "
              "an upper bound; >0 does not by itself mean a real call.)")
    elif args.delegation and widest <= 1:
        print("[openhands] WARNING: no spawn was wider than 1. Sequential "
              "delegation, not fan-out: no concurrency multiplier, no signal.")
    if args.estimate_durations and timed == 0:
        print("[openhands] WARNING: --estimate-durations was ON but no tool call was "
              "measured. The instrumented classes are not being resolved. Check the "
              "'[worker] instrumented tools=' line in stdout.log.")
    elif args.estimate_durations and given == 0 and timed > 0:
        print("[openhands] WARNING: tool calls were measured but the model never "
              "supplied an estimate. Inspect a raw tool call before sweeping.")
    if args.condenser and cons == 0:
        print("[openhands] WARNING: condenser was ON and never fired. Tasks are too "
              "short, or --condenser-max-size is too high.")
    return 0


if __name__ == "__main__":
    sys.exit(main())