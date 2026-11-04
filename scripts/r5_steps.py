"""Round 5 steps (each runs as its own process; scripts/round5.py runs them in order).

  retrieve   GPU   index HotpotQA paragraphs, retrieval metrics, build every test prompt,
                   Axon certificates for the MultiConIR condition check, scanner false positives
  traindata  CPU   fine-tuning examples from the HotpotQA training split
  generate   HTTP  run every test prompt against a chat server; per-item records and metrics
  bench      HTTP  serving benchmark (TTFT, inter-token latency, throughput)
"""
import argparse
import json
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nyaya import guards, rag_eval  # noqa: E402

R = "results"


def rank_metrics(ranked, gold):
    """Recall@k (share of the gold paragraphs found), both-gold@5, MRR@10, nDCG@10 (binary relevance)."""
    import math
    out = {f"recall@{k}": len(gold & set(ranked[:k])) / len(gold) for k in (5, 10, 20)}
    out["both@5"] = float(gold <= set(ranked[:5]))
    first = next((r for r, i in enumerate(ranked[:10], 1) if i in gold), None)
    out["mrr@10"] = 1.0 / first if first else 0.0
    dcg = sum(1 / math.log2(r + 1) for r, i in enumerate(ranked[:10], 1) if i in gold)
    out["ndcg@10"] = dcg / sum(1 / math.log2(r + 1) for r in range(1, min(len(gold), 10) + 1))
    return out


def _dump(obj, path):
    tmp = path + ".tmp"
    json.dump(obj, open(tmp, "w"))
    os.replace(tmp, path)


# --------------------------------------------------------------------------- retrieve

def retrieve(a):
    import torch
    from nyaya.axon import Axon
    from nyaya.benchmarks import localization_items
    from nyaya.rag import AxonRAG, build_axon
    items = rag_eval.load_hotpot("validation", n=a.n_answer + a.n_unanswerable, seed=0, cache=os.path.join(a.data, "hotpot"))
    A, U = items[:a.n_answer], items[a.n_answer:]
    chunks, pos = rag_eval.corpus(items)
    ax = Axon(a.retriever)
    t0 = time.time()
    plaid_error = None
    try:
        build_axon(chunks, a.index, a.retriever, plaid=True, axon=ax)
    except Exception as e:                                   # approximate index is optional
        plaid_error = repr(e)[:300]
        build_axon(chunks, a.index, a.retriever, plaid=False, axon=ax)
    index_s = time.time() - t0
    rag = AxonRAG(a.index, axon=ax)
    sync = torch.cuda.synchronize if torch.cuda.is_available() else (lambda: None)
    per_q, lat = [], {"exact": [], "plaid": [], "bm25": [], "pipeline": []}
    for it in items:
        gold = {pos[t] for t in it["gold"]}
        rec = {"id": it["id"]}
        for name, fn in (("bm25", lambda q: rag.bm25.search(q, 100)),
                         ("exact", lambda q: rag.first_stage(q, 100, approximate=False)),
                         ("plaid", (lambda q: rag.first_stage(q, 100, approximate=True)) if rag.plaid else None)):
            if fn is None:
                continue
            sync()
            t = time.perf_counter()
            ranked = fn(it["question"])
            sync()
            lat[name].append(time.perf_counter() - t)
            rec[f"{name}_r100"] = len(gold & set(ranked)) / 2
        t = time.perf_counter()
        hits = rag.retrieve(it["question"], k=100, depth=100, approximate=False)
        sync()
        lat["pipeline"].append(time.perf_counter() - t)
        axon100 = [int(h["id"][1:]) for h in hits]
        ranked = axon100[:20]
        rec["r5_both"] = float(gold <= set(ranked[:5]))
        rec["r5_any"] = float(bool(gold & set(ranked[:5])))
        fused = [int(h["id"][1:]) for h in rag.retrieve(it["question"], k=20, depth=100, approximate=False, rerank=False)]
        rec["r5_both_fused"] = float(gold <= set(fused[:5]))
        # BM25 fused again with the re-ranker's order (keeps the keyword signal the re-ranker drops)
        sc = {}
        for lst in (rag.bm25.search(it["question"], 100), axon100):
            for r, i in enumerate(lst, 1):
                sc[i] = sc.get(i, 0.0) + 1.0 / (60 + r)
        hybrid = sorted(sc, key=sc.get, reverse=True)[:20]
        rec["lists"] = {"fused": fused, "reranked": ranked, "reranked_bm25": hybrid}
        rec["metrics"] = {k: rank_metrics(v, gold) for k, v in rec["lists"].items()}
        per_q.append(rec)

    def ms(x):
        x = sorted(x)
        return {"median_ms": round(1000 * x[len(x) // 2], 2), "p95_ms": round(1000 * x[int(0.95 * (len(x) - 1))], 2),
                "p99_ms": round(1000 * x[int(0.99 * (len(x) - 1))], 2)} if x else None
    summary = {"questions": len(items), "corpus_paragraphs": len(chunks), "index_seconds": round(index_s, 1),
               "plaid": bool(rag.plaid), "plaid_error": plaid_error,
               "recall@5_both_gold": sum(r["r5_both"] for r in per_q) / len(per_q),
               "recall@5_any_gold": sum(r["r5_any"] for r in per_q) / len(per_q),
               "recall@5_both_gold_fused_only": sum(r["r5_both_fused"] for r in per_q) / len(per_q),
               **{f"{k}_recall@100": sum(r[f"{k}_r100"] for r in per_q) / len(per_q) for k in ("bm25", "exact", "plaid")
                  if f"{k}_r100" in per_q[0]},
               "latency": {k: ms(v) for k, v in lat.items() if v},
               "ranking": {stage: {m: sum(r["metrics"][stage][m] for r in per_q) / len(per_q)
                                   for m in per_q[0]["metrics"][stage]} for stage in ("fused", "reranked", "reranked_bm25")}}
    # which ranking feeds the generator is chosen on the 200 held-out questions (never on the 500 graded ones)
    held = [r for r in per_q if r["id"] in {it["id"] for it in U}] or per_q
    choice = max(("fused", "reranked", "reranked_bm25"),
                 key=lambda st: sum(r["metrics"][st]["both@5"] + 1e-3 * r["metrics"][st]["mrr@10"] for r in held))
    summary["generator_ranking"] = choice
    for r in per_q:
        r["ranked"] = r["lists"][choice]
    # tenant isolation on the real index: half the paragraphs belong to each of two tenants
    import zlib
    for c in rag.chunks:
        c["tenant"] = f"t{zlib.crc32(c['source'].encode()) % 2}"
    rag._masks.clear()
    leaks, returned, refused = 0, 0, 0
    for it in items[:200]:
        hits_t = rag.retrieve(it["question"], k=10, depth=100, tenant="t0", approximate=bool(rag.plaid))
        returned += len(hits_t)
        leaks += sum(1 for h in hits_t if h.get("tenant") != "t0")
    try:
        rag.retrieve(items[0]["question"], k=5)
    except PermissionError:
        refused = 1
    for c in rag.chunks:
        c.pop("tenant", None)
    rag._masks.clear()
    summary["tenant_isolation"] = {"queries": min(200, len(items)), "results_returned": returned,
                                   "cross_tenant_results": leaks, "query_without_tenant_refused": bool(refused)}
    by_id = {r["id"]: r for r in per_q}

    prompts = {"answer": [], "unanswerable": [], "injection": [], "condition": [], "condition_calib": []}
    for it in A:
        srcs = rag_eval.sources_from(chunks, by_id[it["id"]]["ranked"][:5])
        prompts["answer"].append({"item": it, "sources": srcs, "messages": guards.build_messages(it["question"], srcs)})
    for it in U:
        gold = {pos[t] for t in it["gold"]}
        ids = [i for i in by_id[it["id"]]["ranked"] if i not in gold][:5]
        srcs = rag_eval.sources_from(chunks, ids)
        prompts["unanswerable"].append({"item": it, "sources": srcs, "messages": guards.build_messages(it["question"], srcs)})
    flagged_clean = sum(1 for c in chunks if guards.scan_source(c["text"]))
    detected = 0
    for i, p in enumerate(prompts["answer"][:a.n_injection]):
        it, srcs = p["item"], [dict(s) for s in p["sources"]]
        j = next((k for k, s in enumerate(srcs) if s["title"] not in it["gold"]), len(srcs) - 1)
        c, tmpl = rag_eval.canary(i), i % len(rag_eval.TEST_INJECTIONS)
        srcs[j]["text"] = srcs[j]["text"] + " " + rag_eval.TEST_INJECTIONS[tmpl].format(c=c)
        kept, withheld = guards.quarantine(srcs)
        detected += int(any(s["id"] == srcs[j]["id"] for s in withheld))
        prompts["injection"].append({"item": it, "canary": c, "template": tmpl, "sources": srcs, "kept": kept,
                                     "withheld": [s["id"] for s in withheld],
                                     "undefended": guards.build_messages(it["question"], srcs, defended=False),
                                     "defended": guards.build_messages(it["question"], srcs),
                                     "scanned": guards.build_messages(it["question"], kept)})
    summary["scanner"] = {"clean_paragraphs": len(chunks), "false_positives": flagged_clean,
                          "false_positive_rate": flagged_clean / len(chunks),
                          "injected": len(prompts["injection"]), "detected": detected,
                          "detection_rate": detected / max(1, len(prompts["injection"])),
                          "detected_by_template": [sum(1 for p in prompts["injection"] if p["template"] == t and p["withheld"])
                                                   for t in range(len(rag_eval.TEST_INJECTIONS))]}

    loc = localization_items(os.path.join(a.data, "MultiConIR"))
    random.Random(0).shuffle(loc)
    calib, test = loc[:a.n_calib], loc[a.n_calib:a.n_calib + a.n_condition]
    for split, rows in (("condition_calib", calib), ("condition", test)):
        for k, it in enumerate(rows):
            use_pos = k % 2 == 0
            doc = it["positive"] if use_pos else it["negative"]
            r = ax.rerank(it["query"], [doc], conditions=it["conditions"], method="grouped")[0]
            cert = [(m.condition, round(m.relative, 3), [e[2] for e in m.evidence]) for m in r.conditions]
            rec = {"truth": "ALL" if use_pos else it["truth"], "domain": it["domain"],
                   "relative": [c[1] for c in cert], "weakest": r.weakest}
            if split == "condition":
                rec["alone"] = rag_eval.condition_messages(it["conditions"], doc)
                rec["cert"] = rag_eval.condition_messages(it["conditions"], doc, cert)
            prompts[split].append(rec)
    summary["condition_items"] = {"calibration": len(calib), "test": len(test), "pool": len(loc)}
    _dump(summary, f"{R}/r5_retrieval.json")
    _dump(prompts, f"{R}/r5_prompts.json")
    print(json.dumps(summary, indent=1), flush=True)


# --------------------------------------------------------------------------- traindata

def traindata(a):
    # load extra questions: short contexts are skipped, and exactly n_train examples are kept
    items = rag_eval.load_hotpot("train", n=int(a.n_train * 1.25) + 50, seed=1, cache=os.path.join(a.data, "hotpot"))
    ex = rag_eval.training_examples(items, seed=1)[:a.n_train]
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        for e in ex:
            f.write(json.dumps({"messages": e["messages"], "target": e["target"]}) + "\n")
    kinds = {}
    for e in ex:
        kinds[e["kind"]] = kinds.get(e["kind"], 0) + 1
    print(json.dumps({"examples": len(ex), **kinds}), flush=True)


# --------------------------------------------------------------------------- generate

def generate(a):
    from nyaya.llm import chat_full
    P = json.load(open(f"{R}/r5_prompts.json"))
    mt = 192                       # enough for "Answer:" + an evidence sentence + citations; cut-offs are recorded
    lim = a.limit or None
    jobs = []                                                       # (arm, index, messages, max_tokens)
    for i, p in enumerate(P["answer"][:lim]):
        jobs.append(("answer", i, p["messages"], mt))
    for i, p in enumerate(P["unanswerable"][:lim]):
        jobs.append(("unanswerable", i, p["messages"], mt))
    for i, p in enumerate(P["injection"][:lim]):
        for arm in ("undefended", "defended", "scanned"):
            jobs.append((f"injection_{arm}", i, p[arm], mt))
    for i, p in enumerate(P["condition"][:lim]):
        jobs.append(("condition_alone", i, p["alone"], 16))
        jobs.append(("condition_cert", i, p["cert"], 16))

    cut = {}

    def run(job):
        arm, i, msgs, mt = job
        err = "unknown"
        for attempt in range(3):
            try:
                out = chat_full(msgs, temperature=0.0, max_tokens=mt, model=a.model, base=a.base, timeout=600)
                if out["finish_reason"] == "length":
                    cut[(arm, i)] = True
                return job, out["text"]
            except Exception as e:
                err = repr(e)[:200]
                time.sleep(2 * (attempt + 1))
        return job, f"<error {err}>"
    t0 = time.time()
    with ThreadPoolExecutor(a.concurrency) as ex:
        out = list(ex.map(run, jobs))
    replies = {}
    for (arm, i, _, _), rep in out:
        replies.setdefault(arm, {})[i] = rep
    rec = {"tag": a.tag, "model": a.model, "minutes": round((time.time() - t0) / 60, 1), "arms": {}}
    for arm in ("answer", "unanswerable"):
        rec["arms"][arm] = [rag_eval.answer_record(P[arm][i]["item"], P[arm][i]["sources"], r)
                            for i, r in sorted(replies.get(arm, {}).items())]
    for arm in ("undefended", "defended", "scanned"):
        rows = []
        for i, r in sorted(replies.get(f"injection_{arm}", {}).items()):
            p = P["injection"][i]
            srcs = p["kept"] if arm == "scanned" else p["sources"]
            row = rag_eval.answer_record(p["item"], srcs, r)
            row.update(attacked=rag_eval.attacked(r, p["canary"]), template=p["template"])
            rows.append(row)
        rec["arms"][f"injection_{arm}"] = rows
    for arm in ("alone", "cert"):
        rows = []
        for i, r in sorted(replies.get(f"condition_{arm}", {}).items()):
            p = P["condition"][i]
            v = rag_eval.parse_verdict(r)
            rows.append({"truth": p["truth"], "verdict": v, "correct": float(v == p["truth"]), "reply": r[:200]})
        rec["arms"][f"condition_{arm}"] = rows
    for arm, rows in rec["arms"].items():                       # replies that hit the token limit
        for i, row in enumerate(rows):                          # rows are in index order 0..n-1
            row["truncated"] = bool(cut.get((arm, i)))
    rec["truncated"] = {arm: sum(r["truncated"] for r in rows) for arm, rows in rec["arms"].items()}
    rec["errors"] = sum(1 for _, r in out if r.startswith("<error"))
    rec["errors_by_arm"] = {}
    for (arm, _, _, _), r in out:
        if r.startswith("<error"):
            rec["errors_by_arm"][arm] = rec["errors_by_arm"].get(arm, 0) + 1
    _dump(rec, f"{R}/r5_gen_{a.tag}.json")
    print(f"{a.tag}: {len(jobs)} requests in {rec['minutes']} min, {rec['errors']} errors", flush=True)


# --------------------------------------------------------------------------- bench

def bench(a):
    from nyaya.llm_bench import benchmark
    P = json.load(open(f"{R}/r5_prompts.json"))
    prompts = [p["messages"] for p in P["answer"]]               # a pool of 500: no prompt is sent twice
    levels = tuple(int(x) for x in a.levels.split(","))
    rows = benchmark(a.base, a.model, prompts, levels=levels, max_tokens=a.max_tokens,
                     requests_per_level=(lambda c: a.requests or max(16, 4 * c)), gpu=a.gpu)
    _dump({"tag": a.tag, "model": a.model, "max_tokens": a.max_tokens, "rows": rows}, f"{R}/r5_bench_{a.tag}.json")
    for r in rows:
        print({k: r[k] for k in ("concurrency", "ttft_ms_p50", "itl_ms_p50", "tpot_ms_mean", "output_tokens_per_s", "failed")},
              flush=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("retrieve")
    r.add_argument("--retriever", default="lightonai/LateOn")
    r.add_argument("--data", default="data")
    r.add_argument("--index", default="/tmp/r5_index")  # single-user Kaggle VM scratch  # nosec B108
    r.add_argument("--n_answer", type=int, default=500)
    r.add_argument("--n_unanswerable", type=int, default=200)
    r.add_argument("--n_injection", type=int, default=200)
    r.add_argument("--n_condition", type=int, default=600)
    r.add_argument("--n_calib", type=int, default=200)
    t = sub.add_parser("traindata")
    t.add_argument("--data", default="data")
    t.add_argument("--n_train", type=int, default=1600)
    t.add_argument("--out", default="data/r5_train.jsonl")
    g = sub.add_parser("generate")
    g.add_argument("--base", required=True)
    g.add_argument("--model", required=True)
    g.add_argument("--tag", required=True)
    g.add_argument("--concurrency", type=int, default=16)
    g.add_argument("--limit", type=int, default=0)
    b = sub.add_parser("bench")
    b.add_argument("--base", required=True)
    b.add_argument("--model", required=True)
    b.add_argument("--tag", required=True)
    b.add_argument("--levels", default="1,4,16,32")
    b.add_argument("--requests", type=int, default=0)
    b.add_argument("--max_tokens", type=int, default=128)
    b.add_argument("--gpu", type=int, default=None)
    a = ap.parse_args()
    os.makedirs(R, exist_ok=True)
    {"retrieve": retrieve, "traindata": traindata, "generate": generate, "bench": bench}[a.cmd](a)


if __name__ == "__main__":
    main()
