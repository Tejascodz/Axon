"""Client for any OpenAI-compatible chat endpoint (vLLM, SGLang, nyaya.hf_server, a hosted API),
with streaming and per-request timing.

Environment
  NYAYA_LLM_BASE_URL   e.g. http://localhost:8000/v1 (vLLM) or a hosted endpoint
  NYAYA_LLM_API_KEY    API key if the endpoint needs one (only sent over https, or to localhost)
  NYAYA_LLM_MODEL      model name the endpoint expects
  NYAYA_LLM_RETRIES    retries of a request that failed before any output (connection error, 429,
                       500/502/503/504), with exponential backoff and jitter (default 2)
  NYAYA_LLM_BREAKER_FAILURES / NYAYA_LLM_BREAKER_SECONDS
                       circuit breaker: after this many consecutive failed requests (default 5), calls
                       fail at once with CircuitOpen for this many seconds (default 30) instead of
                       piling up on a server that is down; then one trial request decides
  NYAYA_LLM_MAX_RESPONSE_MB
                       largest reply read from the endpoint (default 8): a broken or hostile endpoint
                       cannot exhaust memory

Redirects are never followed: an endpoint that answers 3xx is an error, so the API key is never sent
to a host other than the one configured.

Timing returned by `timed_chat` (all in seconds, measured on the client):
  ttft      time to first token: request sent -> first non-empty content chunk
  itl       inter-token latency: gaps between consecutive content chunks
  tpot      time per output token after the first: (e2e - ttft) / (completion_tokens - 1)
  e2e       end-to-end request latency
"""
from __future__ import annotations

import json
import os
import random
import socket
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

from .metrics import REGISTRY

RETRIES = REGISTRY.counter("llm_retries_total", "LLM requests retried after a transient failure")
FAILURES = REGISTRY.counter("llm_failures_total", "LLM requests that failed after all retries", ["kind"])
BREAKER_REJECTED = REGISTRY.counter("llm_breaker_rejected_total", "LLM calls refused at once while the breaker was open")
_RETRIABLE_HTTP = {429, 500, 502, 503, 504}


class CircuitOpen(RuntimeError):
    """The generator failed repeatedly; calls are refused for a short cool-down."""


class _Breaker:
    """closed: calls go through. open (after N consecutive failures): calls fail at once until the
    cool-down ends. half-open: exactly one trial call goes through; its outcome closes or re-opens."""

    def __init__(self):
        self.lock, self.fails, self.open_until, self.trial = threading.Lock(), 0, 0.0, False

    def _tripped(self):
        return self.fails >= int(os.environ.get("NYAYA_LLM_BREAKER_FAILURES", "5"))

    def check(self):
        with self.lock:
            now = time.monotonic()
            if now < self.open_until or (self._tripped() and self.trial):
                BREAKER_REJECTED.inc()
                raise CircuitOpen(f"generator unavailable; retry in {max(self.open_until - now, 1):.0f} s")
            if self._tripped():
                self.trial = True                          # half-open: this caller is the one trial

    def success(self):
        with self.lock:
            self.fails, self.trial = 0, False

    def failure(self):
        with self.lock:
            self.fails += 1
            self.trial = False
            if self._tripped():
                self.open_until = time.monotonic() + float(os.environ.get("NYAYA_LLM_BREAKER_SECONDS", "30"))

    def release(self):
        """The call ended without saying whether the generator works (e.g. a bad request)."""
        with self.lock:
            self.trial = False


BREAKER = _Breaker()
REGISTRY.gauge("llm_breaker_open", "1 while the circuit breaker refuses LLM calls",
               fn=lambda: 1.0 if time.monotonic() < BREAKER.open_until else 0.0)


def endpoint(base: str | None = None) -> str:
    base = (base or os.environ.get("NYAYA_LLM_BASE_URL", "http://localhost:8000/v1")).rstrip("/")
    u = urllib.parse.urlparse(base)
    if u.scheme not in ("http", "https") or not u.hostname:
        raise ValueError("NYAYA_LLM_BASE_URL must be an http(s) URL")
    local = u.hostname in ("localhost", "127.0.0.1", "::1")
    if os.environ.get("NYAYA_LLM_API_KEY") and u.scheme != "https" and not local:
        raise ValueError("refusing to send NYAYA_LLM_API_KEY over plain http to a remote host; use https")
    return base


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise urllib.error.HTTPError(req.full_url, code, f"redirect to another URL refused ({code})", headers, fp)


_OPENER = urllib.request.build_opener(_NoRedirect)


def _max_bytes() -> int:
    return int(float(os.environ.get("NYAYA_LLM_MAX_RESPONSE_MB", "8")) * 2 ** 20)


def _send(base, body, timeout):
    headers = {"Content-Type": "application/json"}
    if os.environ.get("NYAYA_LLM_API_KEY"):
        headers["Authorization"] = f"Bearer {os.environ['NYAYA_LLM_API_KEY']}"
    req = urllib.request.Request(f"{base}/chat/completions", data=json.dumps(body).encode(), headers=headers)
    return _OPENER.open(req, timeout=timeout)  # base comes from endpoint(), http(s) only  # nosec B310


def _request(base, body, timeout):
    """Open the response, retrying transient failures that happen before any output. A request that
    fails mid-stream is not retried (part of the answer was already used). Client errors (4xx other
    than 429) are neither retried nor counted against the generator."""
    BREAKER.check()
    attempts = 1 + max(0, int(os.environ.get("NYAYA_LLM_RETRIES", "2")))
    try:
        for i in range(attempts):
            try:
                r = _send(base, body, timeout)
                BREAKER.success()
                return r
            except urllib.error.HTTPError as e:
                if e.code not in _RETRIABLE_HTTP:
                    raise
                err, kind = e, f"http_{e.code}"
            except (urllib.error.URLError, ConnectionError, TimeoutError, socket.timeout) as e:
                err, kind = e, type(e).__name__
            if i == attempts - 1:
                FAILURES.inc(kind=kind if kind.startswith("http_") else "connection")
                BREAKER.failure()
                raise err
            RETRIES.inc()
            time.sleep(min(2.0, 0.25 * 2 ** i) * (0.5 + random.random()))  # jitter: retries do not arrive in step  # nosec B311
    finally:
        BREAKER.release()                                  # a 4xx or other error must not hold the trial slot


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
        raw = r.read(_max_bytes() + 1)
        if len(raw) > _max_bytes():
            raise ValueError("LLM reply larger than NYAYA_LLM_MAX_RESPONSE_MB")
        d = json.loads(raw)
    ch = d["choices"][0]
    return {"text": ch["message"]["content"] or "", "finish_reason": ch.get("finish_reason"), "usage": d.get("usage") or {}}


def chat(messages, temperature=0.1, max_tokens=None, model=None, base=None, extra=None, timeout=300) -> str:
    return chat_full(messages, temperature, max_tokens, model, base, extra, timeout)["text"]


def chat_stream(messages, temperature=0.1, max_tokens=None, model=None, base=None, extra=None, timeout=300):
    """Yields ("text", piece) for every content chunk, then ("usage", dict) if the server sends it."""
    with _request(endpoint(base), _body(messages, model, temperature, max_tokens, True, extra), timeout) as r:
        budget = _max_bytes()
        while True:
            raw = r.readline(budget + 1)
            if not raw:
                break
            budget -= len(raw)
            if budget < 0:
                raise ValueError("LLM reply larger than NYAYA_LLM_MAX_RESPONSE_MB")
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
