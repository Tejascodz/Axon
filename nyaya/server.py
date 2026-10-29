"""HTTP re-ranking service.

  pip install "axon-retrieval[server]"
  AXON_MODEL=lightonai/LateOn uvicorn nyaya.server:app --host 127.0.0.1 --port 8000

  POST /rerank {"query": "...", "documents": ["...", ...], "top_k": 10,
                "conditions": null, "method": "grouped", "explain": true}

The model loads once at start-up. Request sizes are capped (AXON_MAX_DOCS, AXON_MAX_CHARS) so a
single request cannot exhaust memory. The service binds to localhost by default and has no
authentication of its own: put it behind your own gateway before exposing it.
"""
from __future__ import annotations

import os
from dataclasses import asdict

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .axon import METHODS, Axon

MAX_DOCS = int(os.environ.get("AXON_MAX_DOCS", "500"))
MAX_CHARS = int(os.environ.get("AXON_MAX_CHARS", "20000"))

app = FastAPI(title="Axon", version="1.0")
_model: list[Axon] = []


def model() -> Axon:
    if not _model:                                  # loaded once, shared by every method
        _model.append(Axon(os.environ.get("AXON_MODEL", "lightonai/LateOn")))
    return _model[0]


class RerankRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    documents: list[str] = Field(min_length=1)
    top_k: int | None = Field(default=None, ge=1)
    conditions: list[str] | None = Field(default=None, max_length=12)
    method: str = "grouped"
    explain: bool = True


@app.get("/health")
def health():
    return {"status": "ok", "model": os.environ.get("AXON_MODEL", "lightonai/LateOn"), "methods": list(METHODS)}


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
    results = model().rerank(req.query, req.documents, top_k=req.top_k, conditions=req.conditions,
                             explain=req.explain, method=req.method)
    return {"results": [asdict(r) for r in results]}
