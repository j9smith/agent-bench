"""Phase-7 + cross-engine acceptance, runnable without a GPU or a real Prometheus.

Drives multi-turn agentic + chat sessions through the proxy against mock_vllm, then
runs export_metrics.py TWICE -- once against a vLLM-shaped fake Prometheus, once
against an SGLang-shaped one -- and asserts:

  * each engine's join lands server-side state on the rows;
  * the SGLang histogram (num_retractions_sum) and prefix drift are handled;
  * `--engine auto` identifies each engine unaided;
  * THE TWO TABLES HAVE IDENTICAL SCHEMAS -- the new acceptance criterion.
"""
import json, os, subprocess, sys, time
from pathlib import Path
import httpx, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "logs" / "e2e_requests.jsonl"
UPSTREAM, PROXY = 8110, 8111
PROM = {"vllm": 8112, "sglang": 8113}
failures = []

def check(name, cond, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}{'  ' + detail if detail and not cond else ''}")
    if not cond: failures.append(name)

def wait(url, timeout=25):
    end = time.time() + timeout
    while time.time() < end:
        try:
            if httpx.get(url, timeout=1).status_code < 500: return True
        except Exception: time.sleep(0.2)
    return False

def turn(sid, stype, msgs):
    url = f"http://127.0.0.1:{PROXY}/sess/{stype}/{sid}/v1/chat/completions"
    with httpx.Client(timeout=30) as c:
        with c.stream("POST", url, json={"model": "mock", "stream": True, "messages": msgs}) as r:
            for _ in r.iter_raw(): pass

def main():
    LOG.parent.mkdir(parents=True, exist_ok=True); LOG.unlink(missing_ok=True)
    base = {**os.environ, "UPSTREAM_BASE_URL": f"http://127.0.0.1:{UPSTREAM}",
            "PROXY_LOG_PATH": str(LOG), "PYTHONPATH": str(ROOT)}
    procs = [
        subprocess.Popen([sys.executable, "-m", "uvicorn", "tools.mock_vllm:app",
                          "--port", str(UPSTREAM), "--log-level", "error"], cwd=ROOT, env=base),
        subprocess.Popen([sys.executable, "-m", "uvicorn", "proxy.main:app",
                          "--port", str(PROXY), "--log-level", "error"], cwd=ROOT, env=base),
    ] + [
        subprocess.Popen([sys.executable, "-m", "uvicorn", "tools.fake_prom:app",
                          "--port", str(PROM[e]), "--log-level", "error"],
                         cwd=ROOT, env={**base, "FAKE_ENGINE": e})
        for e in PROM
    ]
    try:
        assert wait(f"http://127.0.0.1:{PROXY}/healthz")
        for e, p in PROM.items():
            assert wait(f"http://127.0.0.1:{p}/api/v1/query_range?query=x&start=0&end=1&step=15")

        msgs = [{"role": "system", "content": "solve the issue"},
                {"role": "user", "content": "django__django-11099: fix the regex"}]
        for k in range(4):
            turn("django__django-11099", "agentic", msgs)
            msgs = msgs + [{"role": "assistant", "content": f"step {k}"},
                           {"role": "user", "content": "<output>" + "x" * 400 * (k + 1) + "</output>"}]
            time.sleep(0.4)  # stands in for tool execution: this is the gap
        cmsgs = [{"role": "user", "content": "hello"}]
        for k in range(3):
            turn("chat-00001", "chat", cmsgs)
            cmsgs = cmsgs + [{"role": "assistant", "content": "hi"},
                             {"role": "user", "content": f"follow up {k}"}]
            time.sleep(0.2)
        time.sleep(0.6)

        tables, outs = {}, {}
        print()
        for engine in ("vllm", "sglang"):
            out = ROOT / "analysis" / f"e2e_{engine}"
            outs[engine] = out
            # --engine auto: the exporter must work out which engine it's looking at.
            r = subprocess.run(
                [sys.executable, "scripts/export_metrics.py", "--engine", "auto",
                 "--requests", str(LOG), "--prometheus", f"http://127.0.0.1:{PROM[engine]}",
                 "--out", str(out), "--step", "15"],
                cwd=ROOT, env=base, capture_output=True, text=True)
            print(f"--- {engine}\n{r.stdout.strip()}")
            if r.returncode != 0: print(r.stderr, file=sys.stderr)
            check(f"{engine}: export exited 0", r.returncode == 0)
            check(f"{engine}: --engine auto detected it", f"detected engine: {engine}" in r.stdout)
            tables[engine] = pd.read_csv(out.with_suffix(".csv"))

        v, s = tables["vllm"], tables["sglang"]
        agent = v[v.session_type == "agentic"].sort_values("turn_number")

        print("\nacceptance:")
        check("one row per request, both engines", len(v) == 7 and len(s) == 7)
        check("SCHEMAS IDENTICAL across engines", list(v.columns) == list(s.columns),
              f"vllm-only={set(v.columns)-set(s.columns)} sglang-only={set(s.columns)-set(v.columns)}")
        for col in ["kv_cache_utilization", "queue_depth", "running_requests",
                    "eviction_events", "prefix_cache_hit_rate",
                    "prefix_cache_hit_rate_reported", "engine"]:
            check(f"logical column present in both: {col}",
                  col in v.columns and col in s.columns)
        check("engine column tags rows correctly",
              set(v.engine) == {"vllm"} and set(s.engine) == {"sglang"})
        check("vllm: kv utilisation joined onto every row", v.kv_cache_utilization.notna().all())
        check("sglang: kv utilisation joined onto every row (token_usage)",
              s.kv_cache_utilization.notna().all())
        check("sglang: retraction HISTOGRAM resolved via _sum, not a bare counter",
              s.eviction_events.notna().all())
        check("comparable hit rate computed for both from counter deltas",
              v.prefix_cache_hit_rate.notna().any() and s.prefix_cache_hit_rate.notna().any(),
              f"vllm={v.prefix_cache_hit_rate.dropna().unique()} sglang={s.prefix_cache_hit_rate.dropna().unique()}")
        check("comparable hit rate agrees across engines on identical fixtures (0.7)",
              abs(v.prefix_cache_hit_rate.dropna().mean() - 0.7) < 0.01
              and abs(s.prefix_cache_hit_rate.dropna().mean() - 0.7) < 0.01)
        check("engine-reported rate kept separate (sglang gauge 0.68 != derived 0.70)",
              abs(s.prefix_cache_hit_rate_reported.dropna().mean() - 0.68) < 0.01)
        check("session shape unchanged by engine",
              (agent["context_growth_tokens"].dropna() > 0).all()
              and agent["is_resumption"].tolist() == [False, True, True, True])
        check("gap magnitude ~ tool exec time", agent["gap_s"].dropna().between(0.3, 1.5).all())

        # raw archive + provenance
        import pandas as _pd
        for engine in ("vllm", "sglang"):
            raw = ROOT / "analysis" / f"raw_series_{engine}.parquet"
            man = ROOT / "analysis" / f"manifest_{engine}.json"
            check(f"{engine}: raw series archived", raw.exists())
            if raw.exists():
                rf = _pd.read_parquet(raw)
                mapped = set(tables[engine].columns)
                check(f"{engine}: raw archive keeps series the mapping never asked for",
                      len(set(rf.metric)) > 0 and any(
                          m.split(":")[-1].split("_", 1)[-1] not in mapped for m in set(rf.metric)))
                check(f"{engine}: raw archive preserves labels (not summed away)",
                      rf.labels.str.contains("model_name").all())
            check(f"{engine}: run manifest written", man.exists())
            if man.exists():
                mf = json.loads(man.read_text())
                check(f"{engine}: manifest records resolved series names",
                      mf["engine"] == engine and mf["n_requests"] == 7
                      and len(mf["resolved_series"]) >= 6)
        sraw = _pd.read_parquet(ROOT / "analysis" / "raw_series_sglang.parquet")
        check("sglang: phase label survives in the raw archive",
              sraw[sraw.metric.str.contains("num_running_reqs")].labels
                  .str.contains("phase").any())
        check("sglang: histogram parts archived even though only _sum is mapped",
              {"sglang_num_retractions_sum", "sglang_num_retractions_count"} <= set(sraw.metric))
    finally:
        for p in procs: p.terminate()
        for p in procs: p.wait(timeout=10)
    print("\n" + ("ALL CHECKS PASSED" if not failures else f"FAILED: {failures}"))
    return 1 if failures else 0

if __name__ == "__main__":
    sys.exit(main())
