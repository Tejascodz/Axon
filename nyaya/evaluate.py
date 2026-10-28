"""Ranking metrics and corpus-level evaluation."""
from __future__ import annotations

import json
import math
from collections import defaultdict

import torch


def recall_at_k(ranked: list[str], gold: set[str], k: int) -> float:
    if not gold:
        return 0.0
    return len(set(ranked[:k]) & gold) / min(len(gold), k)


def ndcg_at_k(ranked: list[str], gold: set[str], k: int = 10) -> float:
    dcg = sum(1.0 / math.log2(i + 2) for i, d in enumerate(ranked[:k]) if d in gold)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(gold), k)))
    return dcg / idcg if idcg else 0.0


def mrr(ranked: list[str], gold: set[str]) -> float:
    for i, d in enumerate(ranked):
        if d in gold:
            return 1.0 / (i + 1)
    return 0.0


@torch.no_grad()
def encode_corpus(model, texts: list[str], batch: int = 64, max_len: int = 128):
    vecs, masks = [], []
    for i in range(0, len(texts), batch):
        D, m, _, _ = model.encode(texts[i:i + batch], "d", max_len=max_len)
        vecs.append(D.half().cpu())
        masks.append(m.cpu())
    # pad to a common number of vectors (token mode has variable length)
    K = max(v.shape[1] for v in vecs)
    D = torch.zeros(len(texts), K, vecs[0].shape[-1], dtype=torch.float16)
    M = torch.zeros(len(texts), K, dtype=torch.bool)
    o = 0
    for v, m in zip(vecs, masks):
        D[o:o + len(v), : v.shape[1]] = v
        M[o:o + len(v), : m.shape[1]] = m
        o += len(v)
    return D, M


@torch.no_grad()
def rank(model, queries: list[str], D, M, top: int = 100, batch: int = 32, chunk: int = 8192):
    """Exact brute-force late-interaction search.

    On a GPU the whole corpus is kept resident in fp16 (tensor cores), which is ~10x faster
    than streaming fp32 chunks from the CPU; MaxSim values are cast back to fp32 before
    aggregation, so soft-min / mean are computed at full precision.
    """
    device = next(model.parameters()).device
    on_gpu = device.type == "cuda"
    if on_gpu and D.numel() * 2 < 6e9:
        D, M = D.to(device), M.to(device)
    out = []
    for i in range(0, len(queries), batch):
        q = model.encode(queries[i:i + batch], "q")
        if on_gpu:
            q = (q[0].half(),) + tuple(q[1:])
        n_q = q[0].shape[0] * q[0].shape[1] * (q[0].shape[2] if q[0].dim() == 4 else 1)
        step = max(64, min(chunk, int(3e8 // max(1, n_q * D.shape[1]))))   # bound the [B, N, Cq, K] tensor
        best_s, best_i = None, None
        for j in range(0, D.shape[0], step):
            dj = D[j:j + step].to(device)
            d = (dj if on_gpu else dj.float(), M[j:j + step].to(device), None, None)
            s = model.score(q, d)
            k = min(top, s.shape[1])
            v, idx = s.topk(k, dim=1)
            idx = idx + j
            if best_s is None:
                best_s, best_i = v, idx
            else:
                cs, ci = torch.cat([best_s, v], 1), torch.cat([best_i, idx], 1)
                v2, pos = cs.topk(min(top, cs.shape[1]), dim=1)
                best_s, best_i = v2, ci.gather(1, pos)
        out.extend(best_i.cpu().tolist())
    return out


def evaluate(model, docs, queries, ks=(10, 100), group_by_width=True, max_len=128):
    """docs: list with .doc_id/.text ; queries: list with .text/.positives/.clauses."""
    model.eval()
    D, M = encode_corpus(model, [d.text for d in docs], max_len=max_len)
    ranked_idx = rank(model, [q.text for q in queries], D, M, top=max(ks))
    ids = [d.doc_id for d in docs]
    per = defaultdict(lambda: defaultdict(list))
    for q, r in zip(queries, ranked_idx):
        ranked = [ids[i] for i in r]
        gold = set(q.positives)
        keys = ["all"] + ([f"w{len(q.clauses)}"] if group_by_width else [])
        for key in keys:
            for k in ks:
                per[key][f"recall@{k}"].append(recall_at_k(ranked, gold, k))
            per[key]["ndcg@10"].append(ndcg_at_k(ranked, gold, 10))
            per[key]["mrr"].append(mrr(ranked, gold))
    return {g: {m: round(sum(v) / len(v), 4) for m, v in ms.items()} for g, ms in per.items()}


def load_beir(folder: str):
    """Load a BEIR-format dataset (corpus.jsonl, queries.jsonl, qrels/test.tsv or qrels.tsv)."""
    import os
    from types import SimpleNamespace as NS
    docs = [NS(doc_id=j["_id"], text=(j.get("title", "") + " " + j["text"]).strip())
            for j in map(json.loads, open(f"{folder}/corpus.jsonl"))]
    qtext = {j["_id"]: j["text"] for j in map(json.loads, open(f"{folder}/queries.jsonl"))}
    gold = defaultdict(set)
    if os.path.exists(f"{folder}/qrels.jsonl"):                    # LIMIT release format
        for j in map(json.loads, open(f"{folder}/qrels.jsonl")):
            if float(j.get("score", 1)) > 0:
                gold[j["query-id"]].add(j["corpus-id"])
    else:
        qrel_path = next(p for p in (f"{folder}/qrels/test.tsv", f"{folder}/qrels.tsv") if os.path.exists(p))
        for line in open(qrel_path).read().splitlines()[1:]:
            qid, did, s = line.split("\t")[:3]
            if float(s) > 0:
                gold[qid].add(did)
    queries = [NS(query_id=q, text=qtext[q], positives=sorted(g), clauses={}) for q, g in gold.items()]
    return docs, queries
