"""HTTP service: re-ranking and full RAG answers, with metrics and basic protection.

  pip install "axon-retrieval[server]"
  AXON_MODEL=lightonai/LateOn uvicorn nyaya.server:app --host 127.0.0.1 --port 8000

  POST /rerank {"query": "...", "documents": ["...", ...], "top_k": 10,
                "conditions": null, "method": "grouped", "explain": true}
  POST /ask    {"question": "...", "k": 5}      needs AXON_INDEX (built with `python -m nyaya.rag build`)
               and an OpenAI-compatible generator (NYAYA_LLM_BASE_URL, e.g. vLLM)
  GET  /health, GET /metrics (Prometheus)

Protection
  * AXON_API_KEY: when set, every request except /health must send `Authorization: Bearer <key>`
    (compared in constant time).
  * AXON_TENANT_KEYS="tenantA:keyA,tenantB:keyB": one key per tenant of a multi-tenant index
    (built with `--tenants`). The key decides the tenant; /ask only ever retrieves that tenant's
    documents, and the filter is applied inside retrieval, not in the prompt.
  * AXON_RATE_PER_MIN: requests per minute per client address (default 120; 0 disables; /metrics exempt).
  * Request sizes are capped (AXON_MAX_DOCS, AXON_MAX_CHARS) so one request cannot exhaust memory.
  * Binds to localhost by default; put it behind TLS before exposing it.
"""
from __future__ import annotations

import hmac
import os
import threading
import time
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

REQS = REGISTRY.counter("axon_requests_total", "HTTP requests", ["path", "status"])
LAT = REGISTRY.histogram("axon_request_seconds", "HTTP request latency", ["path"])
DOCS = REGISTRY.histogram("axon_documents_per_request", "Documents per re-rank request", [],
                          buckets=(1, 5, 10, 25, 50, 100, 250, 500))

app = FastAPI(title="Axon", version="1.0")
_state: dict = {}
_load_lock = threading.Lock()
_hits: dict = defaultdict(deque)
_rate_lock = threading.Lock()


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
        return _state["rag"]


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
        return True


def _tenant_keys():
    """[(tenant, key)], or None when AXON_TENANT_KEYS is set but malformed (then every request is
    refused: a typo in the key list must never turn authentication off)."""
    raw = os.environ.get("AXON_TENANT_KEYS", "").strip()
    if not raw:
        return []
    pairs = []
    for entry in raw.split(","):
        t, sep, k = entry.partition(":")
        if not sep or not t.strip() or len(k.strip()) < 16:
            return None
        pairs.append((t.strip(), k.strip()))
    return pairs


@app.middleware("http")
async def guard(request: Request, call_next):
    path = request.url.path
    t0 = time.perf_counter()
    request.state.tenant = None
    if path != "/health":
        got = request.headers.get("authorization", "").encode()
        tenant_keys = _tenant_keys()
        key = os.environ.get("AXON_API_KEY")
        if tenant_keys is None:
            REQS.inc(path=path, status="503")
            return JSONResponse({"detail": "server misconfigured: AXON_TENANT_KEYS needs tenant:key pairs, "
                                           "keys of 16+ characters"}, status_code=503)
        if tenant_keys:
            match = None
            for tenant, k in tenant_keys:                  # check every key: timing reveals nothing
                if hmac.compare_digest(got, f"Bearer {k}".encode()):
                    match = tenant
            if match is None:
                REQS.inc(path=path, status="401")
                return JSONResponse({"detail": "unauthorized"}, status_code=401)
            request.state.tenant = match
        elif key and not hmac.compare_digest(got, f"Bearer {key}".encode()):
            REQS.inc(path=path, status="401")
            return JSONResponse({"detail": "unauthorized"}, status_code=401)
        if path != "/metrics" and not _allowed(request.client.host if request.client else "unknown"):
            REQS.inc(path=path, status="429")
            return JSONResponse({"detail": "rate limit exceeded"}, status_code=429)
    resp = await call_next(request)
    REQS.inc(path=path, status=str(resp.status_code))
    LAT.observe(time.perf_counter() - t0, path=path)
    return resp


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
    return {"status": "ok", "model": os.environ.get("AXON_MODEL", "lightonai/LateOn"), "methods": list(METHODS)}


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
    try:
        return pipeline().answer(req.question, k=req.k, tenant=request.state.tenant)
    except HTTPException:
        raise
    except PermissionError:
        raise HTTPException(403, "this index is multi-tenant: use a tenant key")
    except Exception as e:                       # generator unreachable, etc.: no internals in the reply
        REQS.inc(path="/ask", status="502")
        raise HTTPException(502, f"generation failed: {type(e).__name__}")
