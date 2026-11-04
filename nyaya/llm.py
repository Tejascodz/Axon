"""Client for any OpenAI-compatible chat endpoint (vLLM, SGLang, nyaya.hf_server, a hosted API),
with streaming and per-request timing.

Environment
  NYAYA_LLM_BASE_URL   e.g. http://localhost:8000/v1 (vLLM) or a hosted endpoint
  NYAYA_LLM_API_KEY    API key if the endpoint needs one (only sent over https, or to localhost)
  NYAYA_LLM_MODEL      model name the endpoint expects

Timing returned by `timed_chat` (all in seconds, measured on the client):
  ttft      time to first token: request sent -> first non-empty content chunk
  itl       inter-token latency: gaps between consecutive content chunks
  tpot      time per output token after the first: (e2e - ttft) / (completion_tokens - 1)
  e2e       end-to-end request latency
"""
from __future__ import annotations

import json
import os
import time
import urllib.parse
import urllib.request


def endpoint(base: str | None = None) -> str:
    base = (base or os.environ.get("NYAYA_LLM_BASE_URL", "http://localhost:8000/v1")).rstrip("/")
    u = urllib.parse.urlparse(base)
    if u.scheme not in ("http", "https") or not u.hostname:
        raise ValueError("NYAYA_LLM_BASE_URL must be an http(s) URL")
    local = u.hostname in ("localhost", "127.0.0.1", "::1")
    if os.environ.get("NYAYA_LLM_API_KEY") and u.scheme != "https" and not local:
        raise ValueError("refusing to send NYAYA_LLM_API_KEY over plain http to a remote host; use https")
    return base


def _request(base, body, timeout):
    headers = {"Content-Type": "application/json"}
    if os.environ.get("NYAYA_LLM_API_KEY"):
        headers["Authorization"] = f"Bearer {os.environ['NYAYA_LLM_API_KEY']}"
    req = urllib.request.Request(f"{base}/chat/completions", data=json.dumps(body).encode(), headers=headers)
    return urllib.request.urlopen(req, timeout=timeout)


def _body(messages, model, temperature, max_tokens, stream, extra):
    b = {"model": model or os.environ.get("NYAYA_LLM_MODEL", "default"), "messages": messages,
         "temperature": temperature, "stream": stream}
    if max_tokens:
        b["max_tokens"] = max_tokens
    if stream:
        b["stream_options"] = {"include_usage": True}
    b.update(extra or {})
    return b


def chat_full(messages, temperature=0.1, max_tokens=None, model=None, base=None, extra=None, timeout=300) -> dict:
    """{"text", "finish_reason", "usage"}; finish_reason "length" means the reply hit max_tokens."""
    with _request(endpoint(base), _body(messages, model, temperature, max_tokens, False, extra), timeout) as r:
        d = json.loads(r.read())
    ch = d["choices"][0]
    return {"text": ch["message"]["content"] or "", "finish_reason": ch.get("finish_reason"), "usage": d.get("usage") or {}}


def chat(messages, temperature=0.1, max_tokens=None, model=None, base=None, extra=None, timeout=300) -> str:
    return chat_full(messages, temperature, max_tokens, model, base, extra, timeout)["text"]


def chat_stream(messages, temperature=0.1, max_tokens=None, model=None, base=None, extra=None, timeout=300):
    """Yields ("text", piece) for every content chunk, then ("usage", dict) if the server sends it."""
    with _request(endpoint(base), _body(messages, model, temperature, max_tokens, True, extra), timeout) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            ev = json.loads(data)
            for ch in ev.get("choices") or []:
                piece = (ch.get("delta") or {}).get("content")
                if piece:
                    yield "text", piece
            if ev.get("usage"):
                yield "usage", ev["usage"]


def timed_chat(messages, **kw) -> dict:
    t0 = time.perf_counter()
    first, last, gaps, parts, usage = None, None, [], [], {}
    for kind, val in chat_stream(messages, **kw):
        now = time.perf_counter()
        if kind == "usage":
            usage = val
            continue
        if first is None:
            first = now
        else:
            gaps.append(now - last)
        last = now
        parts.append(val)
    end = time.perf_counter()
    n_out = usage.get("completion_tokens") or len(parts)
    ttft = (first - t0) if first is not None else None
    tpot = ((end - first) / (n_out - 1)) if first is not None and n_out > 1 else None
    return {"text": "".join(parts), "ttft": ttft, "itl": gaps, "tpot": tpot, "e2e": end - t0,
            "completion_tokens": n_out, "prompt_tokens": usage.get("prompt_tokens"), "chunks": len(parts)}
