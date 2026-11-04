"""Round 6: Axon's index at scale. Kaggle T4 (one GPU is enough), about 1.5-2 h, unattended.

  python scripts/round6.py            # full run; results/r6.json and results/ROUND6.md
  python scripts/round6.py --smoke    # a few minutes on CPU with a stand-in encoder: checks the script only

Each part writes its results into results/r6.json as soon as it finishes, and a re-run skips parts
already there.

Fixed before running:
  evidence  Re-ranking from the index's stored vectors gives the same scores, order and evidence as
            encoding the candidates afresh (100 HotpotQA questions x 20 candidates, real LateOn).
  scale     HotpotQA paragraphs (the 6,852 of round 5 first, then the rest of validation, then train)
            indexed at 100,000 and 300,000 passages with the production code (nyaya.index, PLAID on).
            Recall@100 of the gold paragraphs (700 round-5 questions): exact search vs PLAID.
            Latency (200 questions): exact search with the vectors resident on the GPU and streamed
            from host memory, PLAID, and the whole retrieval pipeline. Build time, load time,
            peak memory. 300,000 is reduced if the 100,000 build shows it would not finish in 90 min.
  update    On the 100,000 index: change 500 files, delete 250, add 250 (1%). Update time against
            the full build. Checks: changed text is found, deleted files never come back, added files
            are found, recall@100 on the 700 questions is unchanged; then compaction time.
  deps      The image's dependency overrides (deploy/overrides.txt: newer sentence-transformers,
            transformers, ujson) give the same vectors as PyLate's own pins: 3,000 passages and 200
            questions encoded in both environments; token counts, cosine of every vector, and
            top-10 overlap of exact search. PLAID also builds in the overridden environment.
"""
import argparse
import json
import os
import random
import resource
import shutil
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
OUT = os.path.join(ROOT, "results", "r6.json")
RETRIEVER, REVISION = "lightonai/LateOn", "62911e105059585d244384c7d17826e35f669c17"   # the snapshot of round 5


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def save(part, value):
    r = json.load(open(OUT)) if os.path.exists(OUT) else {}
    r[part] = value
    tmp = OUT + ".tmp"
    json.dump(r, open(tmp, "w"), indent=1)
    os.replace(tmp, OUT)


def done(part):
    return os.path.exists(OUT) and part in json.load(open(OUT))


def rss_peak_gb():
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20, 2)       # Linux: KiB


def ms(xs):
    xs = sorted(xs)
    if not xs:
        return None
    return {"p50_ms": round(1000 * xs[len(xs) // 2], 1), "p95_ms": round(1000 * xs[int(0.95 * (len(xs) - 1))], 1),
            "p99_ms": round(1000 * xs[int(0.99 * (len(xs) - 1))], 1), "n": len(xs)}


def make_axon(smoke):
    if smoke:
        sys.path.insert(0, os.path.join(ROOT, "tests"))
        from test_system import _FakeAxon
        ax = _FakeAxon()
        ax.revision = None
        return ax
    from nyaya.axon import Axon
    try:
        return Axon(RETRIEVER, revision=REVISION)
    except Exception as e:                                  # pinned snapshot unavailable: record it, carry on
        log(f"revision {REVISION} failed ({type(e).__name__}); using the latest snapshot")
        save("revision_fallback", repr(e)[:300])
        return Axon(RETRIEVER)


def sync():
    import torch
    if torch.cuda.is_available():
        torch.cuda.synchronize()


# --------------------------------------------------------------------------- data

def corpus(a):
    """(passages [(title, text)], eval questions [(question, gold positions)]) in a fixed order."""
    from nyaya import rag_eval
    if a.smoke:
        rng = random.Random(0)
        words = [f"w{i}" for i in range(3000)]
        paras = [(f"T{i}", " ".join(rng.choice(words) for _ in range(30))) for i in range(a.sizes[-1] + 400)]
        evals = [(" ".join(paras[i][1].split()[:6]), {i, i + 1}) for i in range(0, 60, 2)]
        return paras, evals
    cache = os.path.join(a.data, "hotpot")
    val = rag_eval.load_hotpot("validation", n=None, seed=0, cache=cache)
    pos, paras = {}, []

    def add(items):
        for it in items:
            for t, p in it["paragraphs"]:
                if t not in pos:
                    pos[t] = len(paras)
                    paras.append((t, p))
    add(val[:700])                                                  # exactly round 5's corpus first
    evals = [(it["question"], {pos[t] for t in it["gold"]}) for it in val[:700]]
    add(val[700:])
    if len(paras) < max(a.sizes) + 400:
        add(rag_eval.load_hotpot("train", n=None, seed=0, cache=cache))
    log(f"{len(paras)} unique paragraphs available, {len(evals)} evaluation questions")
    return paras, evals


def text(title, body):
    from nyaya import rag_eval
    return rag_eval.passage(title, body)


# --------------------------------------------------------------------------- evidence

def part_evidence(a, ax, paras, evals):
    import torch
    rng = random.Random(1)
    n_q, n_c = (10, 8) if a.smoke else (100, 20)
    score_diff, same_order, same_evidence, total = 0.0, 0, 0, 0
    for q, gold in evals[:n_q]:
        ids = sorted(gold) + rng.sample(range(len(paras)), n_c - len(gold))
        texts = [text(*paras[i]) for i in ids]
        fresh = ax.rerank(q, ax.index(texts), explain=True)
        store = ax.encode_store(texts)
        stored = ax.rerank(q, ax.store_subset(store, list(range(len(ids))), texts), explain=True)
        score_diff = max(score_diff, max(abs(x.score - y.score) for x, y in
                                         zip(sorted(fresh, key=lambda r: r.index), sorted(stored, key=lambda r: r.index))))
        same_order += [r.index for r in fresh] == [r.index for r in stored]
        for x in fresh:
            y = next(r for r in stored if r.index == x.index)
            total += 1
            same_evidence += [[e[2] for e in m.evidence] for m in x.conditions] == \
                             [[e[2] for e in m.evidence] for m in y.conditions]
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    return {"questions": n_q, "candidates": n_c, "max_score_diff": score_diff,
            "same_order": same_order / n_q, "same_evidence": same_evidence / total}


# --------------------------------------------------------------------------- scale

def recall(rag, evals, approximate, k=100):
    found, lat = [], []
    for q, gold in evals:
        sync()
        t = time.perf_counter()
        ranked = rag.first_stage(q, k, approximate=approximate)
        sync()
        lat.append(time.perf_counter() - t)
        found.append(len(gold & set(ranked[:k])) / len(gold))
    return sum(found) / len(found), lat


def measure(rag, evals, a, has_plaid):
    import torch
    out = {}
    n_lat = 20 if a.smoke else 200
    for q, _ in evals[:5]:                                               # warm up
        rag.first_stage(q, 100, approximate=False)
    rag._s.gpu_budget_gb = 14.0
    rag.store._resident = None
    try:
        r, lat = recall(rag, evals, approximate=False)
        out["exact"] = {"recall@100": round(r, 4), "latency": ms(lat[:n_lat]),
                        "vectors_resident_on_gpu": rag.store._resident is not None}
    except torch.cuda.OutOfMemoryError:
        out["exact"] = {"note": "vectors did not fit on the GPU"}
    rag.store._resident = None
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    rag._s.gpu_budget_gb = 0.0                                           # stream from host memory every query
    r2, lat2 = recall(rag, evals[:n_lat], approximate=False)
    out["exact_streamed"] = {"latency": ms(lat2)}
    if "recall@100" in out["exact"]:
        r_res, _ = recall(rag, evals[:n_lat], approximate=False)
        out["exact_streamed"]["same_results_as_resident"] = abs(r_res - r2) < 1e-9
    else:
        r_full, _ = recall(rag, evals, approximate=False)
        out["exact"]["recall@100"] = round(r_full, 4)
    rag._s.gpu_budget_gb = 14.0
    if has_plaid:
        rp, latp = recall(rag, evals, approximate=True)
        out["plaid"] = {"recall@100": round(rp, 4), "latency": ms(latp[:n_lat])}
    for name, approx in (("pipeline_exact", False), ("pipeline_plaid", True)):
        if approx and not has_plaid:
            continue
        lat = []
        for q, _ in evals[:n_lat]:
            sync()
            t = time.perf_counter()
            rag.retrieve(q, k=5, approximate=approx)
            sync()
            lat.append(time.perf_counter() - t)
        out[name] = {"latency": ms(lat)}
    return out


def write_files(src, paras, n):
    os.makedirs(src, exist_ok=True)
    for i in range(n):
        p = os.path.join(src, f"p{i:07d}.txt")
        if not os.path.exists(p):
            with open(p, "w") as f:
                f.write(text(*paras[i]))


def part_scale(a, ax, paras, evals, size, files):
    import torch
    from nyaya import index as ix
    from nyaya.rag import AxonRAG
    out_dir = os.path.join(a.work, f"index_{size}")
    shutil.rmtree(out_dir, ignore_errors=True)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    if files:
        src = os.path.join(a.work, f"src_{size}")
        write_files(src, paras, size)
        rep = ix.build(src, out_dir, ax, plaid=not a.no_plaid, log=log)
    else:
        rep = ix.build([{"source": f"p{i:07d}.txt", "text": text(*paras[i])} for i in range(size)], out_dir, ax,
                       plaid=not a.no_plaid, log=log)
    cfg = ix.read_config(out_dir)
    res = {"passages": size, "build_seconds": rep["seconds"], "vector_gb": round(rep["vector_mb"] / 1024, 2),
           "plaid": rep["plaid"], "plaid_error": cfg.get("plaid_error"),
           "gpu_peak_build_gb": round(torch.cuda.max_memory_allocated() / 2 ** 30, 2) if torch.cuda.is_available() else None,
           "rss_peak_gb_after_build": rss_peak_gb()}
    t = time.time()
    rag = AxonRAG(out_dir, axon=ax)
    res["load_seconds"] = round(time.time() - t, 1)
    res.update(measure(rag, evals, a, bool(rag.plaid)))
    res["rss_peak_gb"] = rss_peak_gb()
    log(f"scale {size}: {json.dumps(res)}")
    return res, rag, out_dir


def part_update(a, ax, paras, evals, size, out_dir, build_seconds):
    from nyaya import index as ix
    from nyaya.rag import AxonRAG
    src = os.path.join(a.work, f"src_{size}")
    rng = random.Random(2)
    n_eval = max(p for _, g in evals for p in g) + 1                     # never touch the evaluation paragraphs
    k = max(4, size // 100)
    pick = rng.sample(range(n_eval, size), k // 2 + k // 4)
    changed, deleted = pick[:k // 2], pick[k // 2:]
    added = list(range(size, size + k // 4))
    rag = AxonRAG(out_dir, axon=ax)
    before, _ = recall(rag, evals, approximate=False)
    for i in changed:
        with open(os.path.join(src, f"p{i:07d}.txt"), "a") as f:
            f.write(f" Revised entry: catalogue number zq{i}x applies to this record.")
    for i in deleted:
        os.remove(os.path.join(src, f"p{i:07d}.txt"))
    for i in added:
        with open(os.path.join(src, f"p{i:07d}.txt"), "w") as f:
            f.write(text(*paras[i]))
    up = ix.update(src, out_dir, ax, log=log)
    rag.refresh()
    checks = {}
    pos = {c["source"]: n for n, c in enumerate(rag.chunks) if not c.get("deleted")}
    hit = [pos[f"p{i:07d}.txt"] in [int(h["id"][1:]) for h in rag.retrieve(f"catalogue number zq{i}x", k=5)]
           for i in changed[:50]]
    checks["changed_found_top5"] = sum(hit) / len(hit)
    dead = {f"p{i:07d}.txt" for i in deleted}
    back = 0
    for i in deleted[:100]:
        back += any(h["source"] in dead for h in rag.retrieve(paras[i][1][:200], k=20))
    checks["deleted_returned"] = back
    found = [any(h["source"] == f"p{i:07d}.txt" for h in rag.retrieve(paras[i][1][:200], k=5)) for i in added[:50]]
    checks["added_found_top5"] = sum(found) / len(found)
    after, _ = recall(rag, evals, approximate=False)
    checks["recall@100_before"], checks["recall@100_after"] = round(before, 4), round(after, 4)
    if rag.plaid:
        checks["plaid_recall@100_after"] = round(recall(rag, evals, approximate=True)[0], 4)
    c = ix.compact(out_dir)
    rag.refresh()
    checks["recall@100_after_compact"] = round(recall(rag, evals, approximate=False)[0], 4)
    res = {"files_changed": len(changed), "files_deleted": len(deleted), "files_added": len(added),
           "update": up, "full_build_seconds": build_seconds,
           "update_speedup": round(build_seconds / max(up["seconds"], 1e-3), 1), "compact": c, "checks": checks}
    log(f"update: {json.dumps(res)}")
    return res


# --------------------------------------------------------------------------- deps

def encode_sample(path, n_docs, n_q, data, smoke):
    """Runs in either environment: encodes a fixed sample and saves the vectors."""
    import torch
    a = argparse.Namespace(smoke=smoke, data=data, sizes=[n_docs])
    paras, evals = corpus(a)
    ax = make_axon(smoke)
    store = ax.encode_store([text(*p) for p in paras[:n_docs]])
    qs = [torch.as_tensor(ax.query_vectors(q)).half().cpu() for q, _ in evals[:n_q]]
    import importlib.metadata as md
    vers = {p: md.version(p) for p in ("torch", "transformers", "sentence-transformers", "pylate", "ujson", "fast-plaid")}
    plaid = None
    if not smoke:
        try:
            import tempfile

            from pylate import indexes
            with tempfile.TemporaryDirectory() as d:
                pl = indexes.PLAID(index_folder=d, index_name="t", override=True, show_progress=False)
                pl.add_documents(documents_ids=[str(i) for i in range(len(store))],
                                 documents_embeddings=[store.doc(i).float().numpy() for i in range(len(store))])
                plaid = [r["id"] for r in pl(qs[0].float().numpy()[None], k=10)[0]]
        except Exception as e:
            plaid = f"failed: {type(e).__name__}: {str(e)[:200]}"
    torch.save({"flat": store.flat.clone(), "ptr": store.ptr, "q": qs, "versions": vers, "plaid_top10_q0": plaid}, path)


def part_deps(a):
    import torch
    from nyaya.store import VectorStore
    n_docs, n_q = (200, 20) if a.smoke else (3000, 200)
    env = os.path.join(a.work, "deps_env")
    py = os.path.join(env, "bin", "python")
    logf = open(os.path.join(ROOT, "logs", "r6_deps_env.log"), "a")
    if not os.path.exists(py):
        run = lambda cmd: subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT, check=True)   # noqa: E731
        run([sys.executable, "-m", "pip", "install", "-q", "uv"])
        run([sys.executable, "-m", "uv", "venv", "--python", sys.executable, env])
        run([sys.executable, "-m", "uv", "pip", "install", "--python", py, "--torch-backend=auto", "torch==2.11.*"])
        run([sys.executable, "-m", "uv", "pip", "install", "--python", py, "--override",
             os.path.join(ROOT, "deploy", "overrides.txt"), ROOT, "pypdf>=5", "datasets"])
    files = {}
    for name, python in (("pinned", sys.executable), ("overridden", py)):
        files[name] = os.path.join(a.work, f"deps_{name}.pt")
        cmd = [python, os.path.abspath(__file__), "--encode_sample", files[name], "--n_docs", str(n_docs),
               "--n_q", str(n_q), "--data", a.data] + (["--smoke"] if a.smoke else [])
        if subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT, cwd=ROOT).returncode != 0:
            return {"error": f"encoding failed in the {name} environment (logs/r6_deps_env.log)"}
    A, B = (torch.load(files[k], weights_only=True) for k in ("pinned", "overridden"))
    res = {"versions": {"pinned": A["versions"], "overridden": B["versions"]}, "passages": n_docs, "questions": n_q,
           "same_token_counts": bool(torch.equal(A["ptr"], B["ptr"])), "plaid_in_overridden_env": B["plaid_top10_q0"]}
    if res["same_token_counts"]:
        fa, fb = A["flat"].float(), B["flat"].float()
        cos = torch.nn.functional.cosine_similarity(fa, fb, dim=1)
        res.update(vector_cosine_min=round(float(cos.min()), 6), vector_cosine_mean=round(float(cos.mean()), 6),
                   max_abs_diff=round(float((fa - fb).abs().max()), 5))
    sa, sb = VectorStore(A["flat"], A["ptr"]), VectorStore(B["flat"], B["ptr"])
    overlap = [len(set(sa.search(qa.float(), k=10)) & set(sb.search(qb.float(), k=10))) / 10
               for qa, qb in zip(A["q"], B["q"])]
    res["top10_overlap_mean"] = round(sum(overlap) / len(overlap), 4)
    res["top10_identical_share"] = round(sum(o == 1.0 for o in overlap) / len(overlap), 4)
    log(f"deps: {json.dumps(res)}")
    return res


# --------------------------------------------------------------------------- summary

def summary():
    r = json.load(open(OUT)) if os.path.exists(OUT) else {}
    L = ["# Round 6: the index at scale", ""]
    env = r.get("environment", {})
    if env:
        L += [f"Hardware: {', '.join(env.get('gpus', [])) or 'no GPU'}; host memory {env.get('mem_total_gb')} GB.", ""]
    if "evidence" in r:
        e = r["evidence"]
        L += ["## Stored vectors vs fresh encoding (re-ranking)", "",
              f"{e['questions']} questions x {e['candidates']} candidates: max score difference {e['max_score_diff']:.2e}, "
              f"same order {e['same_order']:.0%}, same evidence {e['same_evidence']:.1%}.", ""]
    rows = [(k, v) for k, v in r.items() if k.startswith("scale_")]
    if rows:
        L += ["## Scale", "", "| passages | build | load | vectors | exact recall@100 | exact p50 / p95 (GPU) | exact p50 (streamed) "
              "| PLAID recall@100 | PLAID p50 / p95 | pipeline p50 / p95 (exact) | pipeline p50 / p95 (PLAID) | peak RAM |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|"]

        def lat(d, key, both=True):
            x = (d.get(key) or {}).get("latency")
            return "-" if not x else (f"{x['p50_ms']} / {x['p95_ms']} ms" if both else f"{x['p50_ms']} ms")
        for _, s in rows:
            ex, pl = s.get("exact", {}), s.get("plaid", {})
            L.append(f"| {s['passages']:,} | {s['build_seconds'] / 60:.1f} min | {s['load_seconds']} s | {s['vector_gb']} GB "
                     f"| {ex.get('recall@100', '-')} | {'did not fit' if ex.get('note') else lat(s, 'exact')} "
                     f"| {lat(s, 'exact_streamed', False)} | {pl.get('recall@100', '-')} | {lat(s, 'plaid')} "
                     f"| {lat(s, 'pipeline_exact')} | {lat(s, 'pipeline_plaid')} | {s['rss_peak_gb']} GB |")
        notes = [s.get("plaid_error") for _, s in rows if s.get("plaid_error")]
        L += [""] + [f"PLAID error: {n}" for n in notes] + ([""] if notes else [])
        if r.get("size_note"):
            L += [r["size_note"], ""]
    if "update" in r:
        u = r["update"]
        c = u["checks"]
        L += ["## Incremental update (1% of the 100k index)", "",
              f"{u['files_changed']} changed, {u['files_deleted']} deleted, {u['files_added']} added: update "
              f"{u['update']['seconds']} s against a full build of {u['full_build_seconds']:.0f} s ({u['update_speedup']}x). "
              f"Compaction {u['compact']['seconds']} s.", "",
              f"* changed text found in the top 5: {c['changed_found_top5']:.0%}",
              f"* deleted files returned in the top 20 of 100 queries for their own text: {c['deleted_returned']}",
              f"* added files found in the top 5: {c['added_found_top5']:.0%}",
              f"* recall@100 before / after update / after compaction: {c['recall@100_before']} / {c['recall@100_after']} / "
              f"{c['recall@100_after_compact']}" + (f"; PLAID after update {c['plaid_recall@100_after']}"
                                                    if "plaid_recall@100_after" in c else ""), ""]
    if "deps" in r:
        d = r["deps"]
        L += ["## Dependency overrides (deploy/overrides.txt) vs PyLate's pins", ""]
        if "error" in d:
            L += [d["error"], ""]
        else:
            L += [f"Pinned: {d['versions']['pinned']}", "", f"Overridden: {d['versions']['overridden']}", "",
                  f"{d['passages']} passages, {d['questions']} questions: same token counts {d['same_token_counts']}; "
                  f"vector cosine min {d.get('vector_cosine_min')}, mean {d.get('vector_cosine_mean')}; "
                  f"top-10 overlap {d['top10_overlap_mean']} (identical for {d['top10_identical_share']:.0%} of questions). "
                  f"PLAID in the overridden environment: {d['plaid_in_overridden_env']}.", ""]
    L += ["Not run in this round: the 30-minute load test through the server with SGLang (left out to save GPU quota)."]
    return "\n".join(L) + "\n"


# --------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--sizes", type=int, nargs=2, default=None)
    ap.add_argument("--data", default=os.path.join(ROOT, "data"))
    ap.add_argument("--work", default=None)
    ap.add_argument("--no_plaid", action="store_true")
    ap.add_argument("--skip", nargs="*", default=[], help="parts to skip: evidence scale update deps")
    ap.add_argument("--summary_only", action="store_true")
    ap.add_argument("--encode_sample", default=None)
    ap.add_argument("--n_docs", type=int, default=3000)
    ap.add_argument("--n_q", type=int, default=200)
    a = ap.parse_args()
    if a.encode_sample:
        return encode_sample(a.encode_sample, a.n_docs, a.n_q, a.data, a.smoke)
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    os.makedirs(os.path.join(ROOT, "logs"), exist_ok=True)
    if a.summary_only:
        print(summary())
        return
    a.sizes = a.sizes or ([300, 900] if a.smoke else [100_000, 300_000])
    a.work = a.work or ("/tmp/r6_smoke" if a.smoke else "/tmp/r6")  # single-user Kaggle VM scratch  # nosec B108
    os.makedirs(a.work, exist_ok=True)
    env = {"python": sys.version.split()[0], "smoke": a.smoke}
    try:
        env["gpus"] = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
                                     capture_output=True, text=True, timeout=30).stdout.strip().splitlines()
    except Exception:
        env["gpus"] = []
    env["mem_total_gb"] = round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2 ** 30, 1)
    save("environment", env)
    with open(os.path.join(ROOT, "results", "r6_pip_freeze.txt"), "w") as f:
        subprocess.run([sys.executable, "-m", "pip", "freeze"], stdout=f, stderr=subprocess.DEVNULL)

    paras, evals = corpus(a)
    ax = make_axon(a.smoke)
    if "evidence" not in a.skip and not done("evidence") and not a.smoke:
        save("evidence", part_evidence(a, ax, paras, evals))
    if "scale" not in a.skip:
        small, large = a.sizes
        key = f"scale_{small}"
        if not done(key) or not done("update"):
            res, rag, out_dir = part_scale(a, ax, paras, evals, small, files=True)
            save(key, res)
            del rag
            if "update" not in a.skip and not done("update"):
                save("update", part_update(a, ax, paras, evals, small, out_dir, res["build_seconds"]))
            shutil.rmtree(out_dir, ignore_errors=True)
        b100 = json.load(open(OUT))[key]["build_seconds"]
        budget = 5400                                               # 90 minutes for the large build
        fit = int(small * budget / max(b100, 1))
        if fit < large and not a.smoke:
            save("size_note", f"Large size reduced from {large:,} to {fit:,}: the {small:,} build took {b100 / 60:.0f} min.")
            large = max(small, fit // 10_000 * 10_000)
        large = min(large, len(paras) - 400)
        if large > small and not done(f"scale_{large}"):
            res, rag, out_dir = part_scale(a, ax, paras, evals, large, files=False)
            save(f"scale_{large}", res)
            del rag
            shutil.rmtree(out_dir, ignore_errors=True)
    del ax
    if "deps" not in a.skip and not done("deps"):
        try:
            save("deps", part_deps(a))
        except Exception as e:
            save("deps", {"error": f"{type(e).__name__}: {str(e)[:300]}"})
    md = summary()
    with open(os.path.join(ROOT, "results", "ROUND6.md"), "w") as f:
        f.write(md)
    print("\n" + md, flush=True)


if __name__ == "__main__":
    main()
