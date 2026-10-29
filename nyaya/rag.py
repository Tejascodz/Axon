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


def read_file(path: str) -> str:
    if path.lower().endswith(".pdf"):
        from pypdf import PdfReader  # optional dependency
        return "\n\n".join((p.extract_text() or "") for p in PdfReader(path).pages)
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
            if f.lower().endswith((".txt", ".md", ".pdf")) and not os.path.islink(p):   # never follow links out of src
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


def build_axon(src_or_chunks, out: str, model: str = "lightonai/LateOn", plaid: bool = False, axon=None):
    """Index a folder (or a list of {"source", "text"} dicts) with Axon's ColBERT model."""
    from .axon import Axon
    if isinstance(src_or_chunks, str):
        chunks = []
        for root, _, files in os.walk(src_or_chunks):
            for f in sorted(files):
                p = os.path.join(root, f)
                if f.lower().endswith((".txt", ".md", ".pdf")) and not os.path.islink(p):
                    for i, c in enumerate(chunk(read_file(p))):
                        chunks.append({"source": os.path.relpath(p, src_or_chunks), "part": i, "text": c})
    else:
        chunks = [dict(c) for c in src_or_chunks]
    for n, c in enumerate(chunks):
        c["id"] = f"c{n}"
    ax = axon or Axon(model)
    idx = ax.index([c["text"] for c in chunks])
    os.makedirs(out, exist_ok=True)
    torch.save({"D": idx.D.half().cpu(), "M": idx.M.cpu(), "offsets": idx.offsets}, f"{out}/vectors.pt")
    with open(f"{out}/chunks.jsonl", "w") as f:
        for c in chunks:
            f.write(json.dumps(c) + "\n")
    cfg = {"model": model, "plaid": False, "n": len(chunks)}
    if plaid:
        from pylate import indexes
        emb = [idx.D[i][idx.M[i]].float().cpu().numpy() for i in range(len(chunks))]
        pl = indexes.PLAID(index_folder=out, index_name="plaid", override=True, show_progress=False)
        pl.add_documents(documents_ids=[c["id"] for c in chunks], documents_embeddings=emb)
        cfg["plaid"] = True
    json.dump(cfg, open(f"{out}/config.json", "w"))
    return ax, idx, chunks


class AxonRAG:
    def __init__(self, index_dir: str, axon=None, device: str | None = None):
        from .axon import Axon, Index
        self.cfg = json.load(open(f"{index_dir}/config.json"))
        self.chunks = [json.loads(line) for line in open(f"{index_dir}/chunks.jsonl")]
        self.ax = axon or Axon(self.cfg["model"], device=device)
        v = torch.load(f"{index_dir}/vectors.pt", map_location="cpu", weights_only=True)
        dt = torch.float16 if self.ax.fp16 else torch.float32
        self.index = Index([c["text"] for c in self.chunks], v["D"].to(self.ax.device, dt), v["M"].to(self.ax.device),
                           v.get("offsets"))
        self.bm25 = BM25([c["text"] for c in self.chunks])
        self.plaid = None
        if self.cfg.get("plaid"):
            from pylate import indexes
            self.plaid = indexes.PLAID(index_folder=index_dir, index_name="plaid", override=False, show_progress=False)
        self._pos = {c["id"]: n for n, c in enumerate(self.chunks)}

    def first_stage(self, q: str, depth: int = 100, approximate: bool | None = None) -> list[int]:
        use_plaid = self.plaid is not None if approximate is None else approximate
        if use_plaid:
            res = self.plaid(self.ax.query_vectors(q)[None], k=depth)[0]
            if res:
                return [self._pos[r["id"]] for r in res]
            RAG_EVENTS.inc(event="plaid_empty_fallback_exact")          # never return nothing silently
        return self.ax.search(q, self.index, k=depth)

    def retrieve(self, q: str, k: int = 5, depth: int = 100, rrf_k: int = 60, conditions=None,
                 method: str | None = None, approximate: bool | None = None, rerank: bool = True) -> list[dict]:
        t0 = time.perf_counter()
        fused = defaultdict(float)
        for ranked in (self.bm25.search(q, depth), self.first_stage(q, depth, approximate)):
            for r, i in enumerate(ranked, 1):
                fused[i] += 1.0 / (rrf_k + r)
        cand = sorted(fused, key=fused.get, reverse=True)[:depth]
        t1 = time.perf_counter()
        if not rerank:                                                  # ablation: fused first stage only
            return [{**self.chunks[i], "score": fused[i], "weakest": None, "conditions": []} for i in cand[:k]]
        results = self.ax.rerank(q, self.ax.subset(self.index, cand), top_k=k, conditions=conditions, method=method)
        t2 = time.perf_counter()
        RAG_STAGE.observe(t1 - t0, stage="first_stage")
        RAG_STAGE.observe(t2 - t1, stage="rerank")
        hits = []
        for r in results:
            c = self.chunks[cand[r.index]]
            hits.append({**c, "score": r.score, "weakest": r.weakest,
                         "conditions": [(m.condition, m.relative, [e[2] for e in m.evidence]) for m in r.conditions]})
        return hits

    def answer(self, q: str, k: int = 5, scan: bool = True, certificate: bool = True, max_tokens: int = 256,
               **llm_kw) -> dict:
        hits = self.retrieve(q, k=k)
        sources = [{"id": h["id"], "title": h.get("source", ""), "text": h["text"]} for h in hits]
        withheld = []
        if scan:
            sources, withheld = guards.quarantine(sources)
            RAG_EVENTS.inc(len(withheld), event="source_withheld")
        cert = None
        if certificate and len(split_clauses(q)) > 1:
            keep = {s["id"] for s in sources}
            cert = {h["id"]: h["conditions"] for h in hits if h["id"] in keep and h["conditions"]}
        if not sources:
            RAG_EVENTS.inc(event="no_sources")
            return {"answer": guards.INSUFFICIENT, "abstained": True, "sources": [], "withheld": withheld,
                    "text": "", "check": None, "timing": {}}
        t = timed_chat(guards.build_messages(q, sources, cert), max_tokens=max_tokens, **llm_kw)
        if t["ttft"] is not None:
            RAG_TTFT.observe(t["ttft"])
        for g in t["itl"]:
            RAG_ITL.observe(g)
        chk = guards.check_answer(t["text"], [s["id"] for s in sources])
        RAG_EVENTS.inc(event="abstained" if chk.abstained else "answered")
        if chk.invalid_citations:
            RAG_EVENTS.inc(len(chk.invalid_citations), event="invalid_citation_removed")
        return {"answer": chk.answer, "abstained": chk.abstained, "text": chk.text, "check": chk.__dict__,
                "sources": [{k2: h[k2] for k2 in ("id", "source", "text", "weakest", "conditions") if k2 in h}
                            for h in hits if h["id"] in {s["id"] for s in sources}],
                "withheld": withheld, "certificate": cert,
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
    a = sub.add_parser("ask")
    a.add_argument("question")
    a.add_argument("--index", required=True)
    a.add_argument("--ckpt", default=None)
    a.add_argument("--k", type=int, default=5)
    a.add_argument("--no_llm", action="store_true", help="only print retrieved chunks")
    args = ap.parse_args()
    if args.cmd == "build":
        if args.ckpt:
            build(args.src, args.ckpt, args.out)
        else:
            _, _, ch = build_axon(args.src, args.out, args.model, plaid=args.plaid)
            print(f"indexed {len(ch)} chunks from {args.src}")
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
    if args.no_llm:
        for h in rag.retrieve(args.question, k=args.k):
            print(f"[{h['id']}] {h.get('source', '')}: {h['text'][:160]}...")
            for c, rel, ev in h["conditions"]:
                print(f"     {c}: {rel:.2f} {ev[:2]}")
        return
    out = rag.answer(args.question, k=args.k)
    print(out["text"] or out["answer"])
    for s in out["sources"]:
        print(f"  [{s['id']}] {s.get('source', '')}")
    if out["withheld"]:
        print(f"  withheld {len(out['withheld'])} source(s) that looked like instructions to the model")
    if out["timing"].get("ttft") is not None:
        print(f"  time to first token {out['timing']['ttft'] * 1000:.0f} ms, total {out['timing']['e2e']:.2f} s")


if __name__ == "__main__":
    main()
