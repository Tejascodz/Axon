"""End-to-end RAG: chunking, indexing, hybrid retrieval, Axon re-ranking, guarded generation.

  build : chunk a folder of .txt / .md / .pdf files; index with BM25 + ColBERT token vectors and,
          optionally, a PLAID index (approximate search for large collections)
  ask   : BM25 + ColBERT first stage (fused with RRF) -> Axon re-ranking with per-condition
          evidence -> injection scan -> grounded, cited answer from any OpenAI-compatible endpoint
          (vLLM, SGLang, nyaya.hf_server, a hosted API) -> citation check. Time to first token and
          inter-token latency are recorded for every answer.

  python -m nyaya.rag build --src my_docs/ --out index/ --model lightonai/LateOn --plaid
  NYAYA_LLM_BASE_URL=http://localhost:8000/v1 NYAYA_LLM_MODEL=Qwen/Qwen3-4B-Instruct-2507 \
      python -m nyaya.rag ask --index index/ "Which jackets are waterproof and have a hood?"

Checkpoints trained with nyaya.train still load through `--ckpt` (legacy HybridSearch).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import threading
import time
from collections import defaultdict

import torch

from .bm25 import BM25
from .evaluate import encode_corpus
from . import guards
from .llm import chat, timed_chat
from .metrics import REGISTRY
from .model import HFBackbone, NyayaRetriever, PyLateBackbone, TinyBackbone, WordTokenizer, split_clauses

__all__ = ["AxonRAG", "build_axon", "chat", "chunk", "HybridSearch", "load_retriever"]

# --------------------------------------------------------------------------- loading


def load_retriever(path: str, device: str | None = None) -> NyayaRetriever:
    # weights_only=True: a checkpoint can only contain tensors and plain data, never executable objects
    ck = torch.load(path, map_location="cpu", weights_only=True)
    a, c = ck["args"], ck["config"]
    if a["backbone"] == "tiny":
        tok = WordTokenizer()
        tok.itos = ck["itos"]
        tok.stoi = {w: i for i, w in enumerate(tok.itos)}
        bb, tokenize = TinyBackbone(len(tok.itos), a["hidden"], a["layers"]), tok
    elif a["backbone"].startswith("pylate:"):
        bb, tokenize = PyLateBackbone(a["backbone"][len("pylate:"):], query_length=a.get("query_length")), None
    else:
        bb = HFBackbone(a["backbone"])
        tokenize = bb.tokenize
    m = NyayaRetriever(bb, tokenize, c["q_mode"], c["d_mode"], c["agg"], c["dim"], c["q_slots"],
                       c["d_slots"], c["gated"], c["tau"], within=c.get("within", "mean"),
                       calibrate=c.get("calibrate", False), compete=c.get("compete"),
                       strip_leadin=c.get("strip_leadin", False))
    m.load_state_dict(ck["state"])
    device = device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    return m.to(device).eval()


# --------------------------------------------------------------------------- ingestion


MAX_FILE_MB = float(os.environ.get("AXON_MAX_FILE_MB", "50"))
MAX_PDF_PAGES = int(os.environ.get("AXON_MAX_PDF_PAGES", "500"))


def check_file(path: str) -> str | None:
    """Why a file must not be ingested (None = fine): links, oversized files, and PDFs whose content
    is not a PDF (the extension is not trusted; the file signature is checked)."""
    if os.path.islink(path):
        return "symbolic link"
    size = os.path.getsize(path)
    if size > MAX_FILE_MB * 2 ** 20:
        return f"larger than {MAX_FILE_MB:g} MB"
    if path.lower().endswith(".pdf"):
        with open(path, "rb") as f:
            if f.read(5) != b"%PDF-":
                return "not a PDF (wrong file signature)"
    return None


def read_file(path: str, max_pages: int = MAX_PDF_PAGES) -> str:
    if path.lower().endswith(".pdf"):
        from pypdf import PdfReader  # optional dependency
        pages = PdfReader(path).pages
        if len(pages) > max_pages:
            raise ValueError(f"more than {max_pages} pages")
        return "\n\n".join((p.extract_text() or "") for p in pages)
    return open(path, encoding="utf-8", errors="ignore").read()


def chunk(text: str, max_words: int = 220, overlap: int = 40) -> list[str]:
    """Paragraph-first splitting; long paragraphs fall back to overlapping word windows."""
    out, buf = [], []
    for para in re.split(r"\n\s*\n", text):
        words = para.split()
        if not words:
            continue
        if len(buf) + len(words) <= max_words:
            buf += words
            continue
        if buf:
            out.append(" ".join(buf))
        while len(words) > max_words:
            out.append(" ".join(words[:max_words]))
            words = words[max_words - overlap:]
        buf = words
    if buf:
        out.append(" ".join(buf))
    return out


def build(src: str, ckpt: str, out: str):
    chunks = []
    for root, _, files in os.walk(src):
        for f in sorted(files):
            p = os.path.join(root, f)
            if f.lower().endswith((".txt", ".md", ".pdf")) and check_file(p) is None:     # same limits as build_axon
                for i, c in enumerate(chunk(read_file(p))):
                    chunks.append({"id": f"c{len(chunks)}", "source": os.path.relpath(p, src), "part": i, "text": c})
    model = load_retriever(ckpt)
    D, M = encode_corpus(model, [c["text"] for c in chunks], max_len=256)
    os.makedirs(out, exist_ok=True)
    torch.save({"D": D, "M": M}, f"{out}/vectors.pt")
    with open(f"{out}/chunks.jsonl", "w") as f:
        for c in chunks:
            f.write(json.dumps(c) + "\n")
    print(f"indexed {len(chunks)} chunks from {src}")


# --------------------------------------------------------------------------- retrieval


class HybridSearch:
    def __init__(self, index_dir: str, model: NyayaRetriever):
        self.chunks = [json.loads(l) for l in open(f"{index_dir}/chunks.jsonl")]
        v = torch.load(f"{index_dir}/vectors.pt", map_location="cpu", weights_only=True)
        self.D, self.M = v["D"], v["M"]
        self.bm25 = BM25([c["text"] for c in self.chunks])          # rebuilt from text: no pickled objects on disk
        self.model = model

    @torch.no_grad()
    def nyaya(self, q: str, k: int = 100) -> list[int]:
        dev = next(self.model.parameters()).device
        qv = self.model.encode([q], "q")
        s = self.model.score(qv, (self.D.to(dev).float(), self.M.to(dev), None, None))[0]
        return s.topk(min(k, len(s))).indices.tolist()

    def search(self, q: str, k: int = 8, depth: int = 100, rrf_k: int = 60, w_bm25=1.0, w_nyaya=1.0):
        fused = defaultdict(float)
        for w, ranked in ((w_bm25, self.bm25.search(q, depth)), (w_nyaya, self.nyaya(q, depth))):
            for r, i in enumerate(ranked, 1):
                fused[i] += w / (rrf_k + r)
        return [self.chunks[i] for i in sorted(fused, key=fused.get, reverse=True)[:k]]


# --------------------------------------------------------------------------- generation (legacy)


def answer(question: str, hits: list[dict]) -> str:
    order = hits[0::2] + hits[1::2][::-1]            # strongest evidence at both ends of the context
    sources = [{"id": h["id"], "title": h["source"], "text": h["text"]} for h in order]
    return chat(guards.build_messages(question, sources))


# --------------------------------------------------------------------------- Axon pipeline

RAG_TTFT = REGISTRY.histogram("rag_time_to_first_token_seconds", "Time to first token of RAG answers")
RAG_ITL = REGISTRY.histogram("rag_inter_token_latency_seconds", "Gaps between streamed chunks of RAG answers")
RAG_STAGE = REGISTRY.histogram("rag_stage_seconds", "Time per pipeline stage", ["stage"])
RAG_EVENTS = REGISTRY.counter("rag_events_total", "Guardrail and outcome events", ["event"])


def build_axon(src_or_chunks, out: str, model: str = "lightonai/LateOn", plaid: bool = False, axon=None,
               tenants_from_dirs: bool = False, log=None):
    """Index a folder (or a list of {"source", "text"[, "tenant"]} dicts) with Axon's ColBERT model.
    With `tenants_from_dirs`, each top-level folder is a tenant: its documents are only ever
    retrieved for that tenant. Returns (axon, build report, chunks)."""
    from . import index as ix
    from .axon import Axon
    ax = axon or Axon(model)
    report = ix.build(src_or_chunks, out, ax, plaid=plaid, tenants_from_dirs=tenants_from_dirs, log=log)
    return ax, report, ix.load(out)[1]


def _stamp(index_dir):
    """Identifies the committed version: each commit renames a new config.json into place, so the
    inode changes even where file times are coarse."""
    st = os.stat(f"{index_dir}/config.json")
    return st.st_ino, st.st_mtime_ns


class _Loaded:
    """One consistent snapshot of an index folder: config.json first, then exactly the files it names."""

    def __init__(self, index_dir, gpu_budget_gb):
        from . import index as ix
        for attempt in range(5):
            try:
                mtime = _stamp(index_dir)
                cfg, chunks, store, _ = ix.load(index_dir)
                break
            except (FileNotFoundError, ValueError):                  # a commit landed mid-read: read the new one
                if attempt == 4:
                    raise
                time.sleep(0.5)
        self.cfg, self.chunks, self.store, self.mtime = cfg, chunks, store, mtime
        self.live = store.live.numpy().astype(bool)
        self.bm25 = BM25([c["text"] for c in chunks])               # deleted chunks are masked, not removed
        self.plaid = None
        if cfg.get("plaid"):
            if not ix.plaid_current(index_dir, cfg):
                RAG_EVENTS.inc(event="plaid_stale_exact_search")
            else:
                try:
                    from pylate import indexes
                    self.plaid = indexes.PLAID(index_folder=index_dir, index_name=ix.plaid_dir(cfg), override=False,
                                               show_progress=False)
                except Exception:
                    RAG_EVENTS.inc(event="plaid_unavailable_exact_search")
        self.pos = {c["id"]: n for n, c in enumerate(chunks)}
        self.masks = {}
        self.gpu_budget_gb = gpu_budget_gb


class AxonRAG:
    def __init__(self, index_dir: str, axon=None, device: str | None = None, gpu_budget_gb: float = 6.0):
        from .axon import Axon
        self.dir = index_dir
        cfg = json.load(open(f"{index_dir}/config.json"))
        self.ax = axon or Axon(cfg["model"], device=device, revision=cfg.get("model_revision"))
        self._reload_lock = threading.Lock()
        self._reloader = None
        from .index import check_model
        check_model(cfg, self.ax)                                     # e.g. the server's AXON_MODEL differs
        self._s = _Loaded(index_dir, gpu_budget_gb)

    # the current snapshot; `refresh` swaps it in one assignment, so a request in flight keeps its own
    cfg = property(lambda self: self._s.cfg)
    chunks = property(lambda self: self._s.chunks)
    store = property(lambda self: self._s.store)
    bm25 = property(lambda self: self._s.bm25)
    plaid = property(lambda self: self._s.plaid)
    _pos = property(lambda self: self._s.pos)
    _masks = property(lambda self: self._s.masks)

    def changed(self) -> bool:
        try:
            return _stamp(self.dir) != self._s.mtime
        except OSError:
            return False

    def refresh(self) -> bool:
        """Reload now if the index changed on disk; cheap when it has not. One reload at a time; the
        current snapshot is served until it is replaced."""
        if not self.changed():
            return False
        with self._reload_lock:
            if not self.changed():
                return False
            try:
                self._s = _Loaded(self.dir, self._s.gpu_budget_gb)
            except Exception:
                RAG_EVENTS.inc(event="index_reload_failed")          # keep serving the snapshot we have
                return False
        RAG_EVENTS.inc(event="index_reloaded")
        return True

    def refresh_async(self):
        """For servers: reload in one background thread and return at once, so no request waits."""
        if not self.changed() or (self._reloader is not None and self._reloader.is_alive()):
            return None
        with self._reload_lock:
            if self._reloader is None or not self._reloader.is_alive():
                self._reloader = threading.Thread(target=self.refresh, daemon=True, name="index-reload")
                self._reloader.start()
        return self._reloader

    @property
    def multi_tenant(self) -> bool:
        return bool(self.cfg.get("multi_tenant")) or any("tenant" in c for c in self.chunks)

    def tenant_mask(self, tenant):
        """Which chunks a query may see: live chunks, of that tenant. In a multi-tenant index a tenant
        is required: there is no "see everything" default."""
        import numpy as np
        s = self._s
        if tenant is None:
            if self.multi_tenant:
                raise PermissionError("this index is multi-tenant: a tenant is required")
            return s.live
        if tenant not in s.masks:
            s.masks[tenant] = np.array([c.get("tenant") == tenant for c in s.chunks], dtype=bool) & s.live
        return s.masks[tenant]

    def first_stage(self, q: str, depth: int = 100, approximate: bool | None = None, allowed=None) -> list[int]:
        s = self._s
        allowed = s.live if allowed is None else allowed
        use_plaid = s.plaid is not None if approximate is None else (approximate and s.plaid is not None)
        if use_plaid:
            restricted = not bool(allowed.all())
            sub = [c["id"] for c, ok in zip(s.chunks, allowed) if ok] if restricted and self.multi_tenant else None
            res = s.plaid(self.ax.query_vectors(q)[None], k=depth, subset=sub)[0] if sub != [] else []
            ids = [s.pos[r["id"]] for r in res if r["id"] in s.pos and allowed[s.pos[r["id"]]]]
            if ids:
                return ids
            RAG_EVENTS.inc(event="plaid_empty_fallback_exact")          # never return nothing silently
        return self.ax.search_store(q, s.store, k=depth, allowed=allowed, gpu_budget_gb=s.gpu_budget_gb)

    def retrieve(self, q: str, k: int = 5, depth: int = 100, rrf_k: int = 60, conditions=None,
                 method: str | None = None, approximate: bool | None = None, rerank: bool = True,
                 tenant: str | None = None) -> list[dict]:
        """Tenant and deletion filters apply inside both searches, before fusion and re-ranking, and
        every result is checked again before it is returned."""
        s = self._s                                                   # one snapshot for the whole request
        allowed = self.tenant_mask(tenant)
        t0 = time.perf_counter()
        fused = defaultdict(float)
        for ranked in (s.bm25.search(q, depth, allowed=allowed), self.first_stage(q, depth, approximate, allowed)):
            for r, i in enumerate(ranked, 1):
                fused[i] += 1.0 / (rrf_k + r)
        cand = [i for i in sorted(fused, key=fused.get, reverse=True)[:depth] if allowed[i]]
        t1 = time.perf_counter()
        if not cand:
            return []
        if not rerank:                                                  # ablation: fused first stage only
            return [{**s.chunks[i], "score": fused[i], "weakest": None, "conditions": []} for i in cand[:k]]
        sub = self.ax.store_subset(s.store, cand, [s.chunks[i]["text"] for i in cand])
        results = self.ax.rerank(q, sub, top_k=k, conditions=conditions, method=method)
        t2 = time.perf_counter()
        RAG_STAGE.observe(t1 - t0, stage="first_stage")
        RAG_STAGE.observe(t2 - t1, stage="rerank")
        hits = []
        for r in results:
            if not allowed[cand[r.index]]:                               # defence in depth
                RAG_EVENTS.inc(event="tenant_filter_violation_blocked")
                continue
            c = s.chunks[cand[r.index]]
            hits.append({**c, "score": r.score, "weakest": r.weakest,
                         "conditions": [(m.condition, m.relative, [e[2] for e in m.evidence]) for m in r.conditions]})
        return hits

    def answer(self, q: str, k: int = 5, scan: bool = True, certificate: bool = True, max_tokens: int = 256,
               tenant: str | None = None, require_citation: bool = True, verifier=None, **llm_kw) -> dict:
        """`verifier` (nyaya.verify.Verifier): also withhold answers whose claims the cited sources do
        not support (claim-level check), not only answers without citations."""
        hits = self.retrieve(q, k=k, tenant=tenant)
        sources = [{"id": h["id"], "title": h.get("source", ""), "text": h["text"]} for h in hits]
        withheld = []
        if scan:
            sources, held = guards.quarantine(sources)
            RAG_EVENTS.inc(len(held), event="source_withheld")
            # say which source was withheld and why; its (possibly hostile) text is not sent back
            withheld = [{"id": h["id"], "title": h.get("title", ""), "flags": h["flags"]} for h in held]
        cert = None
        if certificate and len(split_clauses(q)) > 1:
            keep = {s["id"] for s in sources}
            cert = {h["id"]: h["conditions"] for h in hits if h["id"] in keep and h["conditions"]}
        if not sources:
            RAG_EVENTS.inc(event="no_sources")
            return {"answer": guards.INSUFFICIENT, "abstained": True, "sources": [], "withheld": withheld,
                    "text": "", "check": None, "timing": {}}
        t = timed_chat(guards.build_messages(q, sources, cert), max_tokens=max_tokens, **llm_kw)
        if t.get("e2e") is not None:
            RAG_STAGE.observe(t["e2e"], stage="generate")
        if t["ttft"] is not None:
            RAG_TTFT.observe(t["ttft"])
        for g in t["itl"]:
            RAG_ITL.observe(g)
        chk = guards.check_answer(t["text"], [s["id"] for s in sources])
        if require_citation and not chk.abstained and not chk.cited:
            chk = guards.require_citation(chk)
            RAG_EVENTS.inc(event="uncited_answer_withheld")
        verdict = None
        if verifier is not None and not chk.abstained:
            t0 = time.perf_counter()
            verdict = verifier.check(chk.text, sources)
            RAG_STAGE.observe(time.perf_counter() - t0, stage="verify")
            if not verdict.supported:
                chk = guards.withhold(guards.UNSUPPORTED_TEXT)
                RAG_EVENTS.inc(event="unsupported_answer_withheld")
        RAG_EVENTS.inc(event="abstained" if chk.abstained else "answered")
        if chk.invalid_citations:
            RAG_EVENTS.inc(len(chk.invalid_citations), event="invalid_citation_removed")
        return {"answer": chk.answer, "abstained": chk.abstained, "text": chk.text, "check": chk.__dict__,
                "sources": [{k2: h[k2] for k2 in ("id", "source", "text", "weakest", "conditions") if k2 in h}
                            for h in hits if h["id"] in {s["id"] for s in sources}],
                "withheld": withheld, "certificate": cert,
                "verification": None if verdict is None else {"supported": verdict.supported, "reason": verdict.reason,
                                                              "claims": [(c.text, c.cited, c.score) for c in verdict.claims]},
                "timing": {k2: t[k2] for k2 in ("ttft", "tpot", "e2e", "completion_tokens", "prompt_tokens")}}


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--src", required=True)
    b.add_argument("--out", required=True)
    b.add_argument("--model", default="lightonai/LateOn", help="released ColBERT model (Axon)")
    b.add_argument("--ckpt", default=None, help="legacy: a checkpoint trained with nyaya.train")
    b.add_argument("--plaid", action="store_true", help="also build a PLAID index for large collections")
    b.add_argument("--tenants", action="store_true", help="each top-level folder of --src is a separate tenant")
    u = sub.add_parser("update", help="re-index only files that were added, changed or removed")
    u.add_argument("--src", required=True)
    u.add_argument("--index", required=True)
    c = sub.add_parser("compact", help="drop deleted chunks for good (no re-encoding)")
    c.add_argument("--index", required=True)
    st = sub.add_parser("stats", help="what is in an index")
    st.add_argument("--index", required=True)
    a = sub.add_parser("ask")
    a.add_argument("question")
    a.add_argument("--index", required=True)
    a.add_argument("--ckpt", default=None)
    a.add_argument("--k", type=int, default=5)
    a.add_argument("--no_llm", action="store_true", help="only print retrieved chunks")
    a.add_argument("--tenant", default=None, help="required for a multi-tenant index")
    args = ap.parse_args()
    if args.cmd == "build":
        if args.ckpt:
            build(args.src, args.ckpt, args.out)
        else:
            _, rep, ch = build_axon(args.src, args.out, args.model, plaid=args.plaid, tenants_from_dirs=args.tenants,
                                    log=print)
            print(f"indexed {len(ch)} chunks from {args.src} in {rep['seconds']} s ({rep['vector_mb']} MB of vectors)")
            for sk in rep["refused"]:
                print(f"  skipped {sk['source']}: {sk['reason']}")
        return
    if args.cmd in ("update", "compact", "stats"):
        from . import index as ix
        if args.cmd == "stats":
            cfg = ix.read_config(args.index)
            print(json.dumps({k: cfg.get(k) for k in ("model", "version", "n", "n_live", "plaid", "multi_tenant",
                                                       "updated", "plaid_error") if k in cfg}, indent=1))
            return
        if args.cmd == "compact":
            print(json.dumps(ix.compact(args.index)))
            return
        from .axon import Axon
        cfg = ix.read_config(args.index)
        print(json.dumps(ix.update(args.src, args.index, Axon(cfg["model"], revision=cfg.get("model_revision")), log=print)))
        return
    if args.ckpt:
        hs = HybridSearch(args.index, load_retriever(args.ckpt))
        hits = hs.search(args.question, k=args.k)
        for h in hits:
            print(f"[{h['id']}] {h['source']}: {h['text'][:160]}...")
        if not args.no_llm:
            print("\n" + answer(args.question, hits))
        return
    rag = AxonRAG(args.index)
    if rag.multi_tenant and not args.tenant:
        raise SystemExit("this index is multi-tenant: pass --tenant <name>")
    if args.no_llm:
        for h in rag.retrieve(args.question, k=args.k, tenant=args.tenant):
            print(f"[{h['id']}] {h.get('source', '')}: {h['text'][:160]}...")
            for c, rel, ev in h["conditions"]:
                print(f"     {c}: {rel:.2f} {ev[:2]}")
        return
    out = rag.answer(args.question, k=args.k, tenant=args.tenant)
    print(out["text"] or out["answer"])
    for s in out["sources"]:
        print(f"  [{s['id']}] {s.get('source', '')}")
    if out["withheld"]:
        print(f"  withheld {len(out['withheld'])} source(s) that looked like instructions to the model")
    if out["timing"].get("ttft") is not None:
        print(f"  time to first token {out['timing']['ttft'] * 1000:.0f} ms, total {out['timing']['e2e']:.2f} s")


if __name__ == "__main__":
    main()
