"""Real benchmarks used alongside the synthetic AND-of-OR set.

MultiConIR (EMNLP Findings 2025; github.com/EIT-NLP/MultiConIR, Apache-2.0)
  Five domains (books, movies, people, medical cases, legal documents). For each document,
  queries with 1..10 conditions and hard negatives that break some conditions. Metrics follow
  the official scripts:
    task1  win rate of Positive over HN_k for Query_k (k = 1..10), averaged over domains
    task2  Query10 vs Positive, HN1..HN10: Positive > HN1, and HN_i > HN_(i+1) (monotonicity)
    task3  task2 with the descriptive (natural-language) Query10
NanoBEIR (sentence-transformers/NanoBEIR-en): small versions of BEIR datasets, used to check
  that ordinary search is not hurt.
"""
from __future__ import annotations

import os
import re
from types import SimpleNamespace as NS

import torch

from .evaluate import evaluate
from .model import split_clauses

DOMAINS = ["Books", "Movies", "People", "Medical Case", "Legal Document"]
NANO_DEFAULT = ["scifact", "nfcorpus", "fiqa2018", "scidocs", "nq"]   # short queries; argument-style sets skipped
NANO_NAMES = {"climatefever": "ClimateFEVER", "dbpedia": "DBPedia", "fever": "FEVER", "fiqa2018": "FiQA2018",
              "hotpotqa": "HotpotQA", "msmarco": "MSMARCO", "nfcorpus": "NFCorpus", "nq": "NQ",
              "quoraretrieval": "QuoraRetrieval", "scidocs": "SCIDOCS", "arguana": "ArguAna",
              "scifact": "SciFact", "touche2020": "Touche2020"}


# --------------------------------------------------------------------------- MultiConIR

def _slice(q, a, b):
    return tuple(x[a:b] if torch.is_tensor(x) else x for x in q)


@torch.no_grad()
def _encode_docs(model, texts, bs=64):
    """Encode documents once; returns padded [N, K, d] vectors and [N, K] mask."""
    vs, ms = [], []
    for i in range(0, len(texts), bs):
        D, M, _, _ = model.encode(texts[i:i + bs], "d", max_len=512)
        vs.append(D)
        ms.append(M)
    K = max(v.shape[1] for v in vs)
    D = vs[0].new_zeros(len(texts), K, vs[0].shape[-1])
    M = torch.zeros(len(texts), K, dtype=torch.bool, device=D.device)
    o = 0
    for v, m in zip(vs, ms):
        D[o:o + len(v), : v.shape[1]] = v
        M[o:o + len(v), : m.shape[1]] = m
        o += len(v)
    return D, M


@torch.no_grad()
def _row_scores(model, query_sets, docs_per_row, rows_per_chunk=32):
    """For each row: score its queries against its own documents only. Every document is
    encoded exactly once, however many query sets use it.

    query_sets : list of (queries_per_row, clause_mode); returns one list per query set,
                 each holding one [n_queries, n_docs] score list per row.
    """
    outs = [[] for _ in query_sets]
    for i in range(0, len(docs_per_row), rows_per_chunk):
        drows = docs_per_row[i:i + rows_per_chunk]
        nd = len(drows[0])
        D, M = _encode_docs(model, [x for r in drows for x in r])
        for out, (qpr, mode) in zip(outs, query_sets):
            qrows = qpr[i:i + rows_per_chunk]
            nq = len(qrows[0])
            prev = getattr(model, "clause_mode", "auto")
            model.clause_mode = mode
            q = model.encode([x for r in qrows for x in r], "q")
            model.clause_mode = prev
            for r in range(len(drows)):
                S = model.score(_slice(q, r * nq, (r + 1) * nq),
                                (D[r * nd:(r + 1) * nd], M[r * nd:(r + 1) * nd], None, None))
                out.append(S.float().cpu().tolist())
    return outs


def multiconir(model, root: str, domains=DOMAINS, limit_rows: int | None = None, return_pairs: bool = False):
    """Official MultiConIR win rates (macro-averaged over domains, as in the paper's scripts).
    With return_pairs, also returns every individual comparison for significance testing."""
    import pandas as pd
    model.eval()
    t1 = {k: [] for k in range(1, 11)}
    t2, t3 = [], []
    pairs = {"domain": [], "task1": [], "task2": [], "task3": []}
    for dom in domains:
        f1 = os.path.join(root, "datasets", "Task1", f"{dom}_Task1.csv")
        f2 = os.path.join(root, "datasets", "Task2_&_3", f"{dom}_Task2_&_3.csv")
        if not (os.path.exists(f1) and os.path.exists(f2)):
            continue
        a, b = pd.read_csv(f1), pd.read_csv(f2)
        if limit_rows:
            a, b = a.head(limit_rows), b.head(limit_rows)
        # task 1: Query_k vs (Positive, HN_k), k = 1..10
        qs = [[str(r[f"Query{k}"]) for k in range(1, 11)] for _, r in a.iterrows()]
        ds = [[str(r["Positive"])] + [str(r[f"HN{k}"]) for k in range(1, 11)] for _, r in a.iterrows()]
        rows, = _row_scores(model, [(qs, "auto")], ds)
        wins = [[int(S[k - 1][0] > S[k - 1][k]) for k in range(1, 11)] for S in rows]
        for k in range(1, 11):
            t1[k].append(sum(w[k - 1] for w in wins) / len(wins))
        pairs["task1"] += wins
        pairs["domain"] += [dom] * len(wins)
        # tasks 2 and 3: Query10 (instruction / descriptive) vs Positive, HN1..HN10 (same documents)
        cands = [[str(r["Positive"])] + [str(r[f"HN{i}"]) for i in range(1, 11)] for _, r in b.iterrows()]
        r2, r3 = _row_scores(model, [([[str(x)] for x in b["Query10"]], "auto"),
                                     ([[str(x)] for x in b["Natural_Query10"]], "sentences")], cands)
        for rows, store, key in ((r2, t2, "task2"), (r3, t3, "task3")):
            per = [(int(S[0][0] > S[0][1]), sum(float(S[0][i] > S[0][i + 1]) for i in range(1, 10)) / 9) for S in rows]
            store.append((sum(x[0] for x in per) / len(per), sum(x[1] for x in per) / len(per)))
            pairs[key] += per
    if not t2:
        raise FileNotFoundError(f"MultiConIR csv files not found under {root}")
    res = {"task1": {f"wr{k}": round(sum(v) / len(v), 4) for k, v in t1.items()}}
    res["task1"]["mean"] = round(sum(res["task1"].values()) / 10, 4)
    for name, store in (("task2", t2), ("task3", t3)):
        res[name] = {"pos_vs_hn1": round(sum(x[0] for x in store) / len(store), 4),
                     "monotonicity": round(sum(x[1] for x in store) / len(store), 4)}
    return (res, pairs) if return_pairs else res


# --------------------------------------------------------------------------- NanoBEIR

def load_nanobeir(name: str, dataset_id: str = "sentence-transformers/NanoBEIR-en"):
    from datasets import load_dataset
    split = f"Nano{NANO_NAMES[name]}"
    corpus = load_dataset(dataset_id, "corpus", split=split)
    queries = load_dataset(dataset_id, "queries", split=split)
    qrels = load_dataset(dataset_id, "qrels", split=split)
    docs = [NS(doc_id=str(r["_id"]), text=r["text"]) for r in corpus if r["text"]]
    qtext = {str(r["_id"]): r["text"] for r in queries if r["text"]}
    gold: dict[str, set] = {}
    for r in qrels:
        ids = r["corpus-id"] if isinstance(r["corpus-id"], list) else [r["corpus-id"]]
        gold.setdefault(str(r["query-id"]), set()).update(str(x) for x in ids)
    qs = [NS(query_id=q, text=qtext[q], positives=sorted(g), clauses={}) for q, g in gold.items() if q in qtext]
    return docs, qs


def nanobeir(model, names=NANO_DEFAULT) -> dict:
    out = {}
    for n in names:
        docs, qs = load_nanobeir(n)
        out[n] = evaluate(model, docs, qs, ks=(10, 100), group_by_width=False, max_len=300)["all"]
    out["mean_ndcg@10"] = round(sum(out[n]["ndcg@10"] for n in names) / len(names), 4)
    return out


# --------------------------------------------------------------------------- violated-condition localization

_SENT = re.compile(r"(?<=[.!?])\s+")
_STOP = set("the a an and or of to in for on with by from at as is are was were be been this that these those it its "
            "into about over under their his her them they he she which who whom whose than then also has have had "
            "not no".split())


def _content(t):
    return {w[:5] for w in re.findall(r"[a-z0-9]+", t.lower()) if len(w) > 2 and w not in _STOP}


def _overlap(cond, text):
    c = _content(cond)
    return len(c & _content(text)) / max(1, len(c))


def localization_items(root, domains=DOMAINS, limit_rows=None):
    """MultiConIR task-2 rows where the near-miss HN1 differs from the positive in exactly one
    sentence, and that sentence clearly belongs to one of the query's ten conditions.

    Ground truth (fixed before any model was run, and independent of every model): the condition
    with the highest content-word overlap with the removed sentence, kept only when that overlap
    is >= 0.25 and >= 1.5x the second best."""
    import pandas as pd
    items = []
    for dom in domains:
        f = os.path.join(root, "datasets", "Task2_&_3", f"{dom}_Task2_&_3.csv")
        if not os.path.exists(f):
            continue
        b = pd.read_csv(f)
        if limit_rows:
            b = b.head(limit_rows)
        for _, r in b.iterrows():
            pos, hn = str(r["Positive"]), str(r["HN1"])
            P = [s.strip() for s in _SENT.split(pos) if s.strip()]
            H = [s.strip() for s in _SENT.split(hn) if s.strip()]
            removed, added = [s for s in P if s not in H], [s for s in H if s not in P]
            conds = split_clauses(str(r["Query10"]))
            if len(removed) != 1 or len(added) != 1 or len(conds) != 10:
                continue
            ov = [_overlap(c, removed[0]) for c in conds]
            order = sorted(range(10), key=lambda k: -ov[k])
            if ov[order[0]] >= 0.25 and ov[order[0]] >= 1.5 * ov[order[1]] + 1e-9:
                items.append(dict(domain=dom, query=str(r["Query10"]), conditions=conds, positive=pos,
                                  negative=hn, truth=order[0]))
    return items


@torch.no_grad()
def localization(model, root, domains=DOMAINS, limit_rows=None, batch=16):
    """How often each method names the condition the near-miss document actually breaks.

    comparative : condition whose score drops most from the positive to the near-miss
    absolute     : weakest condition of the near-miss alone, each condition's score divided by
                   its score against its own text (what a live system can do: it never sees the positive)
    lexical      : absolute baseline without a model: condition with least word overlap with the near-miss
    Chance is 1/10."""
    if model.config.get("q_mode") != "clause_tokens" or model.config.get("within") == "flat":
        return {"skipped": "needs per-condition scores (a clause-grouped model)"}
    items = localization_items(root, domains, limit_rows)
    hits = {"comparative": [], "absolute": [], "lexical": []}
    for i in range(0, len(items), batch):
        chunk = items[i:i + batch]
        q = model.encode([it["query"] for it in chunk], "q")
        D, M = _encode_docs(model, [t for it in chunk for t in (it["positive"], it["negative"])])
        C, Mc = _encode_docs(model, [c for it in chunk for c in it["conditions"]])
        for r, it in enumerate(chunk):
            qr = _slice(q, r, r + 1)
            s = model.clause_scores(qr, (D[2 * r:2 * r + 2], M[2 * r:2 * r + 2], None, None))[0].float()   # [2, 10]
            own = model.clause_scores(qr, (C[10 * r:10 * r + 10], Mc[10 * r:10 * r + 10], None, None))[0].float()
            ceil = own.diagonal().clamp(min=1e-3)                    # each condition against its own text
            t = it["truth"]
            hits["comparative"].append(int(int((s[0] - s[1]).argmax()) == t))
            hits["absolute"].append(int(int((s[1] / ceil).argmin()) == t))
            lex = [_overlap(c, it["negative"]) for c in it["conditions"]]
            hits["lexical"].append(int(min(range(10), key=lambda k: lex[k]) == t))
    out = {"n": len(items), "chance": 0.1}
    for k, v in hits.items():
        out[k] = round(sum(v) / max(1, len(v)), 4)
    out["hits"] = hits
    return out


# --------------------------------------------------------------------------- latency

@torch.no_grad()
def latency(model, root, n_queries=50, n_docs=100, domain="Books"):
    """Re-ranking cost: documents are encoded once (an index), then each query is encoded and
    scored against `n_docs` candidates. Reports docs/s for encoding and per-query milliseconds."""
    import time

    import pandas as pd
    b = pd.read_csv(os.path.join(root, "datasets", "Task2_&_3", f"{domain}_Task2_&_3.csv"))
    docs = [str(x) for x in b["Positive"].head(n_docs)]
    queries = [str(x) for x in b["Query10"].head(n_queries)]
    dev = next(model.parameters()).device
    sync = (lambda: torch.cuda.synchronize()) if dev.type == "cuda" else (lambda: None)  # noqa: E731
    model.encode(queries[:2], "q")                                   # warm-up
    sync()
    t0 = time.perf_counter()
    D, M = _encode_docs(model, docs)
    sync()
    enc = time.perf_counter() - t0
    times = []
    for q in queries:
        t0 = time.perf_counter()
        model.score(model.encode([q], "q"), (D, M, None, None))
        sync()
        times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    return {"device": str(dev), "docs_per_s": round(len(docs) / enc, 1), "candidates": len(docs),
            "query_ms_median": round(times[len(times) // 2], 2), "query_ms_p95": round(times[int(0.95 * len(times)) - 1], 2)}
