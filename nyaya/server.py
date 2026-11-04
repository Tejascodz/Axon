"""HTTP service: re-ranking and full RAG answers, with metrics, logs and basic protection.

  pip install "axon-retrieval[server]"
  AXON_MODEL=lightonai/LateOn uvicorn nyaya.server:app --host 127.0.0.1 --port 8000

  POST /rerank {"query": "...", "documents": ["...", ...], "top_k": 10,
                "conditions": null, "method": "grouped", "explain": true}
  POST /ask    {"question": "...", "k": 5}      needs AXON_INDEX (built with `python -m nyaya.rag build`)
               and an OpenAI-compatible generator (NYAYA_LLM_BASE_URL, e.g. vLLM or SGLang)
  GET  /health   liveness: the process is up
  GET  /ready    readiness: model (and index, if configured) loaded; 503 until then
  GET  /metrics  Prometheus

Protection
  * AXON_API_KEY: when set, every request except /health and /ready must send
    `Authorization: Bearer <key>` (compared in constant time).
  * AXON_TENANT_KEYS="tenantA:keyA,tenantB:keyB": one key per tenant of a multi-tenant index
    (built with `--tenants`). The key decides the tenant; /ask only ever retrieves that tenant's
    documents, and the filter is applied inside retrieval, not in the prompt. A malformed list
    refuses every request instead of turning authentication off.
  * AXON_REQUIRE_AUTH=1: refuse every request (503) if no key is configured, so a deployment
    with a missing secret fails closed instead of running open. The Docker image sets it.
  * AXON_METRICS_KEY: when set, /metrics accepts only this key (give it to Prometheus, not to
    clients); otherwise /metrics is protected like every other route.
  * Each key setting can instead name a file, e.g. AXON_API_KEY_FILE=/run/secrets/axon_api_key
    (Docker or Kubernetes secrets), so keys never sit in the environment. The file is re-read
    when it changes, so keys rotate without a restart.
  * AXON_RATE_PER_MIN: requests per minute per client address (default 120; 0 disables; /metrics exempt).
  * AXON_MAX_INFLIGHT: /ask requests served at once (default 16); more get 503 at once instead of
    queueing without bound (back-pressure).
  * Request sizes are capped (AXON_MAX_DOCS, AXON_MAX_CHARS) so one request cannot exhaust memory.
  * Metrics are labelled by route, never by the raw URL, so unknown paths cannot grow memory.
  * Binds to localhost by default; put it behind TLS before exposing it.

Operations
  * AXON_PRELOAD=1 (default) loads the model and index at start-up; /ready says when.
  * Every response carries X-Request-ID (taken from the request if it is a safe value, otherwise
    new), and one JSON log line per request records it with route, status, latency and tenant.
    Queries, documents and keys are never logged.
  * After `python -m nyaya.rag update`, the next /ask sees the new index; no restart.
"""
from __future__ import annotations

import contextlib
import hmac
import json
import logging
import os
import re
import sys
import threading
import time
import uuid
from collections import defaultdict, deque
from dataclasses import asdict

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel, Field

from .axon import METHODS, Axon
from .metrics import REGISTRY

MAX_DOCS = int(os.environ.get("AXON_MAX_DOCS", "500"))
MAX_CHARS = int(os.environ.get("AXON_MAX_CHARS", "20000"))
RATE = int(os.environ.get("AXON_RATE_PER_MIN", "120"))
MAX_INFLIGHT = int(os.environ.get("AXON_MAX_INFLIGHT", "16"))
ROUTES = {"/health", "/ready", "/metrics", "/rerank", "/ask", "/docs", "/openapi.json"}
OPEN = {"/health", "/ready"}                         # probes: no key needed, nothing sensitive returned

REQS = REGISTRY.counter("axon_requests_total", "HTTP requests", ["path", "status"])
LAT = REGISTRY.histogram("axon_request_seconds", "HTTP request latency", ["path"])
DOCS = REGISTRY.histogram("axon_documents_per_request", "Documents per re-rank request", [],
                          buckets=(1, 5, 10, 25, 50, 100, 250, 500))
INFLIGHT = REGISTRY.counter("axon_ask_rejected_total", "/ask requests refused because the server was full")
_START = time.time()
_inflight = [0]
_inflight_lock = threading.Lock()


def _rss_bytes():
    with open("/proc/self/statm") as f:                      # Linux; elsewhere the sample is left out
        return int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE")


def _index_value(fn):
    def read():
        rag = _state.get("rag")
        return None if rag is None else fn(rag)
    return read


REGISTRY.gauge("axon_ready", "1 once the model (and index, if configured) is loaded",
               fn=lambda: 1.0 if _state.get("ready") else 0.0)
REGISTRY.gauge("axon_ask_inflight", "/ask requests being served now", fn=lambda: _inflight[0])
REGISTRY.gauge("axon_ask_capacity", "AXON_MAX_INFLIGHT", fn=lambda: MAX_INFLIGHT)
REGISTRY.gauge("axon_index_version", "Version of the index snapshot being served",
               fn=_index_value(lambda r: r.cfg.get("version", 0)))
REGISTRY.gauge("axon_index_live_chunks", "Searchable chunks in the served index", fn=_index_value(lambda r: r.store.n_live))
REGISTRY.gauge("axon_index_dead_chunks", "Deleted chunks awaiting `rag compact`",
               fn=_index_value(lambda r: len(r.store) - r.store.n_live))
REGISTRY.gauge("process_resident_memory_bytes", "Resident memory of the server process", fn=_rss_bytes)
REGISTRY.gauge("process_start_time_seconds", "Start time of the process (Unix seconds)", fn=lambda: _START)

class _StdoutHandler(logging.Handler):
    """One JSON object per line on whatever stdout is now (process managers may swap it)."""

    def emit(self, record):
        try:
            sys.stdout.write(self.format(record) + "\n")
            sys.stdout.flush()
        except Exception:
            self.handleError(record)


log = logging.getLogger("axon.access")
if not log.handlers:
    log.addHandler(_StdoutHandler())
    log.setLevel(logging.INFO)
    log.propagate = False

@contextlib.asynccontextmanager
async def _lifespan(_app):
    if os.environ.get("AXON_PRELOAD", "1") == "1":
        threading.Thread(target=_preload, daemon=True).start()
    yield


app = FastAPI(title="Axon", version="1.0", lifespan=_lifespan)
_state: dict = {}
_load_lock = threading.Lock()
_hits: dict = defaultdict(deque)
_rate_lock = threading.Lock()
_ask_slots = threading.BoundedSemaphore(MAX_INFLIGHT)
_RID = re.compile(r"^[A-Za-z0-9._-]{1,64}$")


def model() -> Axon:
    with _load_lock:                                # loaded once, shared by every method and by /ask
        if "axon" not in _state:
            _state["axon"] = Axon(os.environ.get("AXON_MODEL", "lightonai/LateOn"))
        return _state["axon"]


def pipeline():
    with _load_lock:
        if "rag" not in _state:
            idx = os.environ.get("AXON_INDEX")
            if not idx:
                raise HTTPException(503, "AXON_INDEX is not configured")
            from .rag import AxonRAG
            _state["rag"] = AxonRAG(idx, axon=_state.get("axon"))
            _state.setdefault("axon", _state["rag"].ax)
    rag = _state["rag"]
    rag.refresh()                                   # picks up `rag update` / `rag compact`
    return rag


def _preload():
    try:
        model()
        if os.environ.get("AXON_INDEX"):
            pipeline()
        _state["ready"] = True
    except Exception as e:                          # stays not-ready; the reason is logged, not returned
        _state["load_error"] = type(e).__name__
        log.info(json.dumps({"event": "preload_failed", "error": type(e).__name__, "detail": str(e)[:300]}))




def _allowed(client: str) -> bool:
    if RATE <= 0:
        return True
    now = time.monotonic()
    with _rate_lock:
        q = _hits[client]
        while q and now - q[0] > 60:
            q.popleft()
        if len(q) >= RATE:
            return False
        q.append(now)
        if len(_hits) > 100_000:                    # forget idle clients: the table cannot grow forever
            for c in [c for c, d in _hits.items() if not d or now - d[-1] > 60][:50_000]:
                _hits.pop(c, None)
        return True


_secret_cache: dict = {}


def _secret(name: str):
    """NAME from the environment, or the contents of the file NAME_FILE (cached until it changes).
    An unreadable file raises: a missing secret must never mean "no authentication"."""
    if os.environ.get(name):
        return os.environ[name]
    path = os.environ.get(name + "_FILE")
    if not path:
        return None
    st = os.stat(path)
    key = (path, st.st_mtime_ns, st.st_size)
    if _secret_cache.get(name, (None,))[0] != key:
        with open(path) as f:
            _secret_cache[name] = (key, f.read().strip())
    return _secret_cache[name][1] or None


def _tenant_keys():
    """[(tenant, key)], or None when AXON_TENANT_KEYS is set but malformed (then every request is
    refused: a typo in the key list must never turn authentication off)."""
    raw = (_secret("AXON_TENANT_KEYS") or "").replace("\n", ",").strip().strip(",")
    if not raw:
        return []
    pairs = []
    for entry in raw.split(","):
        t, sep, k = entry.partition(":")
        if not sep or not t.strip() or len(k.strip()) < 16:
            return None
        pairs.append((t.strip(), k.strip()))
    return pairs


def _authorize(raw: str, got: bytes):
    """(None, tenant) when the request may go on, else ((status, detail), None). Every configured key
    is compared, in constant time, whatever matches first."""
    try:
        tenant_keys, key, metrics_key = _tenant_keys(), _secret("AXON_API_KEY"), _secret("AXON_METRICS_KEY")
    except OSError:
        return (503, "server misconfigured: a key file cannot be read"), None
    if raw == "/metrics" and metrics_key:                  # least privilege: the scraper's key reads metrics only
        return (None if hmac.compare_digest(got, f"Bearer {metrics_key}".encode()) else (401, "unauthorized")), None
    if tenant_keys is None:
        return (503, "server misconfigured: AXON_TENANT_KEYS needs tenant:key pairs, keys of 16+ characters"), None
    if tenant_keys:
        match = None
        for tenant, k in tenant_keys:                      # check every key: timing reveals nothing
            if hmac.compare_digest(got, f"Bearer {k}".encode()):
                match = tenant
        return (None, match) if match is not None else ((401, "unauthorized"), None)
    if key:
        return (None if hmac.compare_digest(got, f"Bearer {key}".encode()) else (401, "unauthorized")), None
    if os.environ.get("AXON_REQUIRE_AUTH") == "1":         # deployments set this: no key configured = refuse
        return (503, "server misconfigured: AXON_REQUIRE_AUTH=1 but no key is set"), None
    return None, None


def _deny(path, rid, status, detail):
    REQS.inc(path=path, status=str(status))
    return JSONResponse({"detail": detail}, status_code=status, headers={"X-Request-ID": rid})


@app.middleware("http")
async def guard(request: Request, call_next):
    raw = request.url.path
    path = raw if raw in ROUTES else "other"        # bounded metric labels
    rid = request.headers.get("x-request-id", "")
    rid = rid if _RID.match(rid) else uuid.uuid4().hex
    request.state.request_id, request.state.tenant = rid, None
    t0 = time.perf_counter()
    status = 500
    try:
        if raw not in OPEN:
            denied, request.state.tenant = _authorize(raw, request.headers.get("authorization", "").encode())
            if not denied and raw != "/metrics" and not _allowed(request.client.host if request.client else "unknown"):
                denied = (429, "rate limit exceeded")
            if denied:
                status = denied[0]
                return _deny(path, rid, *denied)
        resp = await call_next(request)
        status = resp.status_code
        resp.headers["X-Request-ID"] = rid
        REQS.inc(path=path, status=str(status))
        return resp
    finally:
        dt = time.perf_counter() - t0
        LAT.observe(dt, path=path)
        if raw not in ("/metrics", "/health", "/ready"):
            log.info(json.dumps({"ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "request_id": rid,
                                 "method": request.method, "route": path, "status": status, "ms": round(1000 * dt, 1),
                                 "tenant": request.state.tenant}))


class RerankRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    documents: list[str] = Field(min_length=1)
    top_k: int | None = Field(default=None, ge=1)
    conditions: list[str] | None = Field(default=None, max_length=12)
    method: str = "grouped"
    explain: bool = True


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    k: int = Field(default=5, ge=1, le=20)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/ready")
def ready():
    if _state.get("ready") or (os.environ.get("AXON_PRELOAD", "1") != "1" and "axon" in _state):
        return {"status": "ready", "index": bool(os.environ.get("AXON_INDEX")), "methods": list(METHODS)}
    return JSONResponse({"status": "loading" if "load_error" not in _state else "failed"}, status_code=503)


@app.get("/metrics", response_class=PlainTextResponse)
def metrics():
    return REGISTRY.render()


@app.post("/rerank")
def rerank(req: RerankRequest):
    if req.method not in METHODS:
        raise HTTPException(422, f"method must be one of {list(METHODS)}")
    if len(req.documents) > MAX_DOCS:
        raise HTTPException(413, f"at most {MAX_DOCS} documents per request")
    if any(len(d) > MAX_CHARS for d in req.documents):
        raise HTTPException(413, f"documents are limited to {MAX_CHARS} characters")
    if req.conditions and any(len(c) > 500 for c in req.conditions):
        raise HTTPException(413, "conditions are limited to 500 characters")
    DOCS.observe(len(req.documents))
    results = model().rerank(req.query, req.documents, top_k=req.top_k, conditions=req.conditions,
                             explain=req.explain, method=req.method)
    return {"results": [asdict(r) for r in results]}


@app.post("/ask")
def ask(req: AskRequest, request: Request):
    if not _ask_slots.acquire(blocking=False):
        INFLIGHT.inc()
        raise HTTPException(503, "server busy, retry shortly", headers={"Retry-After": "1"})
    with _inflight_lock:
        _inflight[0] += 1
    try:
        out = pipeline().answer(req.question, k=req.k, tenant=request.state.tenant)
        out["request_id"] = request.state.request_id
        return out
    except HTTPException:
        raise
    except PermissionError:
        raise HTTPException(403, "this index is multi-tenant: use a tenant key")
    except Exception as e:                       # generator unreachable, etc.: no internals in the reply
        log.info(json.dumps({"event": "ask_failed", "request_id": request.state.request_id,
                             "error": type(e).__name__}))
        raise HTTPException(502, f"generation failed: {type(e).__name__}")
    finally:
        with _inflight_lock:
            _inflight[0] -= 1
        _ask_slots.release()
