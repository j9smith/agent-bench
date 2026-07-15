"""Fake Prometheus query_range, for exercising the join without a real server.

ENGINE=vllm serves vLLM-shaped series with the _total suffix (proving the suffix
probe works). ENGINE=sglang serves SGLang-shaped series with the sglang_ prefix,
num_retractions as a HISTOGRAM (_sum/_count/_bucket, no bare counter), and
num_running_reqs split by phase -- i.e. all three of the shapes that would break a
naive hardcoded mapping.
"""
import os, re, time
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

app = FastAPI()
ENGINE = os.environ.get("FAKE_ENGINE", "vllm")

VLLM = {
    "vllm:kv_cache_usage_perc":        lambda i: 0.30 + 0.05 * i,
    "vllm:num_requests_running":       lambda i: 2 + i,
    "vllm:num_requests_waiting":       lambda i: 0,
    "vllm:num_preemptions_total":      lambda i: 3.0 * i,
    "vllm:prefix_cache_queries_total": lambda i: 1000.0 * (i + 1),
    "vllm:prefix_cache_hits_total":    lambda i: 700.0 * (i + 1),
    "vllm:prompt_tokens_total":        lambda i: 5000.0 * (i + 1),
    "vllm:generation_tokens_total":    lambda i: 500.0 * (i + 1),
}
# Deliberately: no bare `sglang_num_retractions` counter, only the histogram parts.
SGLANG = {
    "sglang_token_usage":             lambda i: 0.30 + 0.05 * i,
    "sglang_num_running_reqs":        lambda i: 2 + i,     # phase-labelled upstream; sum() folds it
    "sglang_num_queue_reqs":          lambda i: 0,
    "sglang_num_retractions_sum":     lambda i: 3.0 * i,
    "sglang_num_retractions_count":   lambda i: 1.0 * i,
    "sglang_cached_tokens_total":     lambda i: 700.0 * (i + 1),
    "sglang_prompt_tokens_total":     lambda i: 1000.0 * (i + 1),
    "sglang_generation_tokens_total": lambda i: 500.0 * (i + 1),
    "sglang_cache_hit_rate":          lambda i: 0.68,
}
SERIES = VLLM if ENGINE == "vllm" else SGLANG

@app.get("/api/v1/label/__name__/values")
async def label_values():
    return JSONResponse({"status": "success", "data": sorted(SERIES)})


@app.get("/api/v1/query_range")
async def query_range(request: Request):
    q = request.query_params.get("query", "")
    m = re.search(r'__name__="([^"]+)"', q)   # sum({__name__="..."}) or {__name__="..."}
    name = m.group(1) if m else q
    start = float(request.query_params.get("start", time.time()))
    end = float(request.query_params.get("end", time.time()))
    step = float(request.query_params.get("step", 15))
    if name not in SERIES:
        return JSONResponse({"status": "success", "data": {"resultType": "matrix", "result": []}})
    fn, values, i, t = SERIES[name], [], 0, start
    while t <= end + step:
        values.append([t, str(fn(i))]); i += 1; t += step
    labels = {"__name__": name, "model_name": "mock"}
    if name.endswith("num_running_reqs") and ENGINE == "sglang":
        labels["phase"] = "decode"          # SGLang label-splits this; sum() must fold it
    return JSONResponse({"status": "success",
                         "data": {"resultType": "matrix",
                                  "result": [{"metric": labels, "values": values}]}})
