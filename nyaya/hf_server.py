"""A deliberately simple OpenAI-compatible chat server on Hugging Face transformers.

It runs one request at a time, with no batching, no paged KV cache and no prefix caching: the baseline
that shows what a serving engine such as vLLM adds, and a fallback when vLLM is not available.

  python -m nyaya.hf_server --model Qwen/Qwen3-4B-Instruct-2507 --port 8100

Endpoints: POST /v1/chat/completions (stream or not; `ignore_eos` for fixed-length benchmark
outputs), GET /v1/models, GET /health, GET /metrics.
"""
from __future__ import annotations

import argparse
import json
import os
import threading
import time
import uuid

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse, StreamingResponse
from pydantic import BaseModel, Field

from .metrics import REGISTRY

TTFT = REGISTRY.histogram("hf_time_to_first_token_seconds", "Server-side time to first token")
E2E = REGISTRY.histogram("hf_e2e_request_seconds", "Server-side request latency")
TOKENS = REGISTRY.counter("hf_generation_tokens_total", "Generated tokens")
PROMPT = REGISTRY.counter("hf_prompt_tokens_total", "Prompt tokens")

app = FastAPI(title="nyaya-hf-server")
STATE: dict = {}
LOCK = threading.Lock()                      # one generation at a time: the naive baseline


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str | None = None
    messages: list[Message] = Field(min_length=1, max_length=64)
    max_tokens: int = Field(default=256, ge=1, le=4096)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    stream: bool = False
    ignore_eos: bool = False
    stream_options: dict | None = None


def load(name: str, device: str | None = None, revision: str | None = None):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if torch.device(device).type == "cuda" else torch.float32
    revision = revision or os.environ.get("HF_MODEL_REVISION") or None      # a commit hash pins the weights
    tok = AutoTokenizer.from_pretrained(name, revision=revision)
    from .lora import dtype_kw
    model = AutoModelForCausalLM.from_pretrained(name, revision=revision, **dtype_kw(dtype)).to(device).eval()
    STATE.update(name=name, tok=tok, model=model, device=device)


def _prompt_ids(messages):
    tok = STATE["tok"]
    ids = tok.apply_chat_template([m.model_dump() for m in messages], add_generation_prompt=True,
                                  return_tensors="pt", return_dict=True)["input_ids"]
    return ids.to(STATE["device"])


def _generate(req: ChatRequest):
    """Yields text pieces; returns usage via the final yielded dict."""
    from transformers import StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer
    tok, model = STATE["tok"], STATE["model"]
    cancel = threading.Event()

    class Cancelled(StoppingCriteria):                   # a client that disconnects stops its generation
        def __call__(self, input_ids, scores, **kw):
            return cancel.is_set()
    with LOCK:
        t0 = time.perf_counter()
        ids = _prompt_ids(req.messages)
        streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True)
        kw = dict(input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=req.max_tokens,
                  streamer=streamer, do_sample=req.temperature > 0, pad_token_id=tok.pad_token_id or tok.eos_token_id,
                  stopping_criteria=StoppingCriteriaList([Cancelled()]))
        if req.temperature > 0:
            kw["temperature"] = req.temperature
        if req.ignore_eos:
            kw["min_new_tokens"] = req.max_tokens
        out = {}

        def run():
            with torch.no_grad():
                out["seq"] = model.generate(**kw)
        th = threading.Thread(target=run)
        th.start()
        first = None
        try:
            for piece in streamer:
                if piece:
                    if first is None:
                        first = time.perf_counter()
                        TTFT.observe(first - t0)
                    yield piece
        finally:                                          # also on disconnect: never leave a generation running
            cancel.set()
            th.join()
        n_out = int(out["seq"].shape[1] - ids.shape[1]) if "seq" in out else 0
        E2E.observe(time.perf_counter() - t0)
        TOKENS.inc(n_out)
        PROMPT.inc(int(ids.shape[1]))
        yield {"prompt_tokens": int(ids.shape[1]), "completion_tokens": n_out,
               "total_tokens": int(ids.shape[1]) + n_out}


@app.get("/health")
def health():
    return {"status": "ok", "model": STATE.get("name")}


@app.get("/v1/models")
def models():
    return {"object": "list", "data": [{"id": STATE.get("name"), "object": "model"}]}


@app.get("/metrics", response_class=PlainTextResponse)
def metrics():
    return REGISTRY.render()


@app.post("/v1/chat/completions")
def completions(req: ChatRequest):
    if "model" not in STATE:
        raise HTTPException(503, "model not loaded")
    rid, created = f"chatcmpl-{uuid.uuid4().hex[:12]}", int(time.time())

    def chunk(delta, finish=None, usage=None):
        c = {"id": rid, "object": "chat.completion.chunk", "created": created, "model": STATE["name"],
             "choices": [] if usage else [{"index": 0, "delta": delta, "finish_reason": finish}]}
        if usage:
            c["usage"] = usage
        return f"data: {json.dumps(c)}\n\n"

    if req.stream:
        def events():
            yield chunk({"role": "assistant"})
            usage = None
            for p in _generate(req):
                if isinstance(p, dict):
                    usage = p
                else:
                    yield chunk({"content": p})
            yield chunk({}, finish="length" if usage and usage["completion_tokens"] >= req.max_tokens else "stop")
            if usage and (req.stream_options or {}).get("include_usage"):
                yield chunk({}, usage=usage)
            yield "data: [DONE]\n\n"
        return StreamingResponse(events(), media_type="text/event-stream")
    parts, usage = [], {}
    for p in _generate(req):
        if isinstance(p, dict):
            usage = p
        else:
            parts.append(p)
    return {"id": rid, "object": "chat.completion", "created": created, "model": STATE["name"],
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "".join(parts)},
                         "finish_reason": "length" if usage.get("completion_tokens", 0) >= req.max_tokens else "stop"}],
            "usage": usage}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8100)
    a = ap.parse_args()
    load(a.model)
    import uvicorn
    uvicorn.run(app, host=a.host, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
