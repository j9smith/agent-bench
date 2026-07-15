"""A tiny fake vLLM. Exists only so the proxy can be exercised on a laptop with
no GPU. Not part of the measurement path -- delete it and nothing breaks."""

import json
import time

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse

app = FastAPI()
WORDS = ["the", "kv", "cache", "does", "not", "amortize", "across", "a", "batch"]


def _usage(body):
    n = sum(len(str(m.get("content", ""))) for m in body.get("messages", [])) // 4
    return {"prompt_tokens": max(n, 1), "completion_tokens": len(WORDS), "total_tokens": max(n, 1) + len(WORDS)}


@app.post("/v1/chat/completions")
async def chat(request: Request):
    body = await request.json()
    include_usage = bool((body.get("stream_options") or {}).get("include_usage"))
    if not body.get("stream"):
        return JSONResponse({
            "id": "cmpl-mock", "object": "chat.completion", "created": int(time.time()),
            "model": body.get("model", "mock"),
            "choices": [{"index": 0, "message": {"role": "assistant", "content": " ".join(WORDS)},
                         "finish_reason": "stop"}],
            "usage": _usage(body),
        })

    async def gen():
        for w in WORDS:
            chunk = {"id": "cmpl-mock", "object": "chat.completion.chunk",
                     "model": body.get("model", "mock"),
                     "choices": [{"index": 0, "delta": {"content": w + " "}, "finish_reason": None}]}
            yield f"data: {json.dumps(chunk)}\n\n".encode()
        stop = {"id": "cmpl-mock", "object": "chat.completion.chunk",
                "model": body.get("model", "mock"),
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}
        yield f"data: {json.dumps(stop)}\n\n".encode()
        if include_usage:
            u = {"id": "cmpl-mock", "object": "chat.completion.chunk",
                 "model": body.get("model", "mock"), "choices": [], "usage": _usage(body)}
            yield f"data: {json.dumps(u)}\n\n".encode()
        yield b"data: [DONE]\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")


@app.get("/metrics")
async def metrics():
    return PlainTextResponse(
        "# TYPE vllm:prefix_cache_queries_total counter\n"
        'vllm:prefix_cache_queries_total{model_name="mock"} 1000.0\n'
        "# TYPE vllm:prefix_cache_hits_total counter\n"
        'vllm:prefix_cache_hits_total{model_name="mock"} 700.0\n'
        "# TYPE vllm:kv_cache_usage_perc gauge\n"
        'vllm:kv_cache_usage_perc{model_name="mock"} 0.42\n'
        "# TYPE vllm:num_preemptions_total counter\n"
        'vllm:num_preemptions_total{model_name="mock"} 0.0\n'
    )


@app.get("/v1/models")
async def models():
    return JSONResponse({"object": "list", "data": [{"id": "mock", "object": "model"}]})
