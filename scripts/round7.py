"""Round 6b + 7. Kaggle T4 x2, about 2.5 h, unattended.

  python scripts/round7.py            # results/r7.json, results/r6.json, results/ROUND7.md
  python scripts/round7.py --smoke    # a few minutes on CPU with stand-ins: checks the script only

Each part writes its result as soon as it finishes; a re-run skips finished parts.

GPU 1 (one after another)
  deps        Round 6's dependency check, again (it ran out of GPU memory in round 6).
  compaction  PLAID index of 20,000 HotpotQA passages: update 1%, then compact. Compaction now keeps
              PLAID (chunk ids are stable) instead of re-clustering, which took 49 min in round 6.
  verify      Round 7 (below).
  scale       300,000 passages with exact search only (round 6 stopped at 150,000), after the load
              test so the two do not compete for host memory.

GPU 0
  serving     SGLang 0.5.23 with Qwen3-4B-Instruct-2507: mixed workload benchmark (short and long
              prompts, outputs of 32 / 128 / 512 tokens) at concurrency 4 and 16.
  judge       The same LLM as a judge of claim support, on round 7's labelled test pairs.
  soak        The Axon server (4 tenants, each with its own key, 30,000-passage index) answering
              through SGLang for 30 min with 8 concurrent clients: errors, latency per minute,
              memory, cross-tenant results. At minute 15 an index update (200 files added, 100
              deleted) runs while traffic continues. Then a burst of 48 clients for 60 s (requests
              above AXON_MAX_INFLIGHT must be refused at once with 503) and 60 s of recovery.
              Then 5 min with claim verification switched on (AXON_VERIFY=1): its latency cost.

Round 7: claim-level evidence checking (fixed before running)
  Q1  Labelled pairs from HotpotQA's supporting sentences (round 5's 500 questions): a supporting
      sentence with its own paragraph (supported); the same sentence with a distractor paragraph
      of the same question (not supported); the sentence with a number or the answer changed
      (not supported). Threshold chosen on the first 250 questions, results on the other 250:
      ROC AUC, precision, recall, F1, detection per negative type, for the NLI checker
      (DeBERTa-v3-large NLI) and for the LLM as judge.
  Q2  Round 5's real answers (base and fine-tuned, 500 answerable + 200 with the gold paragraphs
      withheld): share of claims supported; exact match of answers judged supported vs not; the
      effect of withholding unsupported answers (answers kept, exact match among them, wrong and
      right answers removed; answers given where the gold was withheld that are caught).
  Q3  A sheet of 200 claims for hand labelling (results/r7_label_sheet.csv): the checker's
      agreement with a person is measured after the run.
"""
import argparse
import contextlib
import csv
import fcntl
import json
import os
import random
import re
import secrets
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import zlib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(1, os.path.join(ROOT, "tests"))           # smoke mode's stand-in encoder
OUT = os.path.join(ROOT, "results", "r7.json")
PROMPTS = os.path.join(ROOT, "results", "r5_prompts.json")
LLM, SERVED = "Qwen/Qwen3-4B-Instruct-2507", "axon-llm"
PY = sys.executable


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


@contextlib.contextmanager
def _locked():
    with open(OUT + ".lock", "w") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def load():
    return json.load(open(OUT)) if os.path.exists(OUT) else {}


def save(part, value):
    with _locked():                                   # steps run in parallel processes
        r = load()
        r[part] = value
        tmp = OUT + ".tmp"
        json.dump(r, open(tmp, "w"), indent=1)
        os.replace(tmp, OUT)


def done(part):
    v = load().get(part)
    return v is not None and not (isinstance(v, dict) and "error" in v)


def pct(xs, q):
    xs = sorted(x for x in xs if x is not None)
    return None if not xs else xs[min(len(xs) - 1, int(q / 100 * (len(xs) - 1) + 0.5))]


def ms(x):
    return None if x is None else round(1000 * x, 1)


# --------------------------------------------------------------------------- round 7 data

def labelled_pairs(smoke=False):
    """[(question index, premise, claim, label, kind)] from round 5's answer prompts."""
    from nyaya import rag_eval
    prompts = json.load(open(PROMPTS))["answer"]
    rng = random.Random(7)
    answers = [p["item"]["answer"] for p in prompts]
    out = []
    for qi, p in enumerate(prompts[:40] if smoke else prompts):
        it = p["item"]
        paras = {t: rag_eval.passage(t, x) for t, x in it["paragraphs"]}
        distract = [t for t in paras if t not in it["gold"]]
        for s in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", it["evidence"]):
            s = s.strip()
            home = next((g for g in it["gold"] if s and s in paras.get(g, "")), None)
            if home is None or len(s.split()) < 6:
                continue
            out.append((qi, paras[home], s, True, "supported"))
            if distract:
                out.append((qi, paras[rng.choice(distract)], s, False, "wrong_source"))
            nums = re.findall(r"\b\d{2,4}\b", s)
            if nums:
                n = rng.choice(nums)
                out.append((qi, paras[home], re.sub(rf"\b{n}\b", str(int(n) + rng.choice([3, 7, 11])), s, count=1),
                            False, "changed_number"))
            elif len(it["answer"]) > 3 and it["answer"] in s:
                other = rng.choice([a for a in answers if a != it["answer"] and len(a) > 3])
                out.append((qi, paras[home], s.replace(it["answer"], other, 1), False, "changed_answer"))
    return out


def auroc(scores, labels):
    pos = [s for s, y in zip(scores, labels) if y]
    neg = [s for s, y in zip(scores, labels) if not y]
    if not pos or not neg:
        return None
    ranked = sorted(scores)
    rank = {}
    for i, s in enumerate(ranked):                    # average rank for ties
        rank.setdefault(s, []).append(i + 1)
    r = sum(sum(rank[s]) / len(rank[s]) for s in pos)
    return round((r - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)), 4)


def prf(pred, labels):
    tp = sum(p and y for p, y in zip(pred, labels))
    fp = sum(p and not y for p, y in zip(pred, labels))
    fn = sum((not p) and y for p, y in zip(pred, labels))
    prec, rec = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    return {"precision": round(prec, 4), "recall": round(rec, 4), "f1": round(2 * prec * rec / max(prec + rec, 1e-9), 4),
            "accuracy": round(sum(p == y for p, y in zip(pred, labels)) / max(len(labels), 1), 4)}


def make_verifier(smoke):
    from nyaya import verify
    if smoke:                                          # lexical stand-in: entailed when every word appears
        v = verify.Verifier.__new__(verify.Verifier)
        v.threshold, v.name = 0.5, "lexical-stand-in"
        v.entailment = lambda pairs: [sum(w in p.lower() for w in verify._norm(h).split()) / max(1, len(verify._norm(h).split()))
                                      for p, h in pairs]
        return v
    return verify.Verifier()


def step_verify(a):
    """Q1 and Q2 with the NLI checker."""
    from nyaya import verify
    v = make_verifier(a.smoke)
    pairs = labelled_pairs(a.smoke)
    half = 20 if a.smoke else 250
    t = time.perf_counter()
    scores = v.entailment([(p, c) for _, p, c, _, _ in pairs])
    per_pair_ms = 1000 * (time.perf_counter() - t) / max(len(pairs), 1)
    cal = [(s, y) for (qi, _, _, y, _), s in zip(pairs, scores) if qi < half]
    test = [(s, y, k) for (qi, _, _, y, k), s in zip(pairs, scores) if qi >= half]
    thr = verify.calibrate([s for s, _ in cal], [y for _, y in cal])
    thr95 = verify.calibrate([s for s, _ in cal], [y for _, y in cal], target_precision=0.95)
    lab = [y for _, y, _ in test]
    bench = {"model": v.name, "pairs": len(pairs), "test_pairs": len(test),
             "kinds": {k: sum(1 for *_, kk in pairs if kk == k) for k in ("supported", "wrong_source", "changed_number", "changed_answer")},
             "threshold_f1": round(thr, 4), "threshold_p95": round(thr95, 4), "ms_per_pair": round(per_pair_ms, 2),
             "auroc": auroc([s for s, _, _ in test], lab),
             "at_threshold_f1": prf([s >= thr for s, _, _ in test], lab),
             "at_threshold_p95": prf([s >= thr95 for s, _, _ in test], lab),
             "negatives_caught": {k: round(sum(s < thr for s, _, kk in test if kk == k) /
                                           max(1, sum(1 for *_, kk in test if kk == k)), 4)
                                  for k in ("wrong_source", "changed_number", "changed_answer")}}
    save("verifier_bench", bench)
    save("verifier_pair_scores", [round(s, 5) for s in scores])     # for the judge comparison
    log(f"verifier bench: {json.dumps(bench)}")
    v.threshold = thr
    save("verify_real", real_answers(v, a.smoke))


def real_answers(v, smoke):
    """Q2: round 5's generated answers, checked claim by claim."""
    prompts = json.load(open(PROMPTS))
    out, sheet = {}, []
    for tag in ("base", "ft"):
        g = json.load(open(os.path.join(ROOT, "results", f"r5_gen_{tag}.json")))["arms"]
        res = {}
        for arm in ("answer", "unanswerable"):
            src = {p["item"]["id"]: p for p in prompts[arm]}
            recs = g[arm][:30] if smoke else g[arm]
            rows, t0 = [], time.perf_counter()
            for r in recs:
                p = src[r["id"]]
                vd = v.check(r["reply"], p["sources"])
                rows.append((r, vd))
                for c in vd.claims:
                    sheet.append({"model": tag, "arm": arm, "id": r["id"], "question": p["item"]["question"],
                                  "claim": c.text, "cited": " ".join(c.cited),
                                  "cited_text": " || ".join(s["text"] for s in p["sources"] if s["id"] in c.cited),
                                  "score": None if c.score is None else round(c.score, 4),
                                  "verdict": "supported" if c.supported else "not supported", "your_label": ""})
            sec = (time.perf_counter() - t0) / max(len(recs), 1)
            given = [(r, vd) for r, vd in rows if not r["abstained"]]
            claims = [c for _, vd in given for c in vd.claims]
            sup = [(r, vd) for r, vd in given if vd.supported]
            uns = [(r, vd) for r, vd in given if not vd.supported]
            em = lambda xs: round(sum(r["em"] for r, _ in xs) / len(xs), 4) if xs else None  # noqa: E731
            row = {"answers": len(recs), "answers_given": len(given), "claims": len(claims),
                   "claims_supported": round(sum(bool(c.supported) for c in claims) / max(len(claims), 1), 4),
                   "claims_uncited": sum(not c.cited for c in claims),
                   "answers_supported": len(sup), "answers_unsupported": len(uns),
                   "reasons_unsupported": {k: sum(vd.reason == k for _, vd in uns) for k in {vd.reason for _, vd in uns}},
                   "seconds_per_answer": round(sec, 4)}
            if arm == "answer":
                row.update({"em_if_supported": em(sup), "em_if_unsupported": em(uns),
                            "before": {"answered": len(given), "em_among_answered": em(given),
                                       "em_overall": round(sum(r["em"] for r in recs) / len(recs), 4),
                                       "wrong_answers": sum(r["em"] == 0 for r, _ in given)},
                            "after_withholding_unsupported": {
                                "answered": len(sup), "em_among_answered": em(sup),
                                "em_overall": round(sum(r["em"] for r, _ in sup) / len(recs), 4),
                                "wrong_answers": sum(r["em"] == 0 for r, _ in sup),
                                "wrong_removed": sum(r["em"] == 0 for r, _ in uns),
                                "right_removed": sum(r["em"] > 0 for r, _ in uns)}})
            else:                                    # gold withheld: an answer here should have been an abstention
                row.update({"answered_without_gold": len(given), "caught_by_check": len(uns)})
            res[arm] = row
        out[tag] = res
    rng = random.Random(3)
    picked = []
    for verdict in ("supported", "not supported"):    # 200 claims, half of each verdict, both models
        pool = [s for s in sheet if s["verdict"] == verdict]
        picked += rng.sample(pool, min(100, len(pool)))
    rng.shuffle(picked)
    if picked:
        with open(os.path.join(ROOT, "results", "r7_label_sheet.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(picked[0]))
            w.writeheader()
            w.writerows(picked)
    log(f"real answers: {json.dumps(out)}")
    return out


def step_judge(a):
    """The LLM as judge, on the same test pairs."""
    from concurrent.futures import ThreadPoolExecutor

    from nyaya import verify
    pairs = labelled_pairs(a.smoke)
    half = 20 if a.smoke else 250
    test = [(i, p, c, y, k) for i, (qi, p, c, y, k) in enumerate(pairs) if qi >= half]

    def one(x):
        try:
            return verify.judge_with_llm(x[1], x[2], base=a.llm_base, model=SERVED)
        except Exception:
            return None
    t = time.perf_counter()
    with ThreadPoolExecutor(16) as ex:
        votes = list(ex.map(one, test))
    ok = [(v, x) for v, x in zip(votes, test) if v is not None]
    res = {"model": LLM, "test_pairs": len(test), "unparsed": len(test) - len(ok),
           "ms_per_pair_at_concurrency_16": round(1000 * (time.perf_counter() - t) / max(len(test), 1), 1),
           "metrics": prf([v for v, _ in ok], [x[3] for _, x in ok]),
           "negatives_caught": {k: round(sum(not v for v, x in ok if x[4] == k) / max(1, sum(1 for _, x in ok if x[4] == k)), 4)
                                for k in ("wrong_source", "changed_number", "changed_answer")}}
    save("judge", res)
    log(f"judge: {json.dumps(res)}")


# --------------------------------------------------------------------------- serving benchmark

def step_serving(a):
    from concurrent.futures import ThreadPoolExecutor

    from nyaya.llm import timed_chat
    from nyaya.llm_bench import GpuSampler
    prompts = json.load(open(PROMPTS))["answer"]
    rng = random.Random(11)
    levels = [(2, 12)] if a.smoke else [(4, 120), (16, 240)]
    out = {}
    for conc, n in levels:
        work = []
        for i in range(n):
            p = prompts[rng.randrange(len(prompts))]
            long = rng.random() < 0.5
            msgs = p["messages"] if long else [{"role": "user", "content": p["item"]["question"]}]
            work.append((msgs, "long" if long else "short", rng.choice([32, 128, 512])))

        def one(w):
            try:
                r = timed_chat(w[0], model=SERVED, base=a.llm_base, max_tokens=w[2], temperature=0.0,
                               extra={"ignore_eos": True})
                return {**r, "prompt": w[1], "max_tokens": w[2]}
            except Exception as e:
                return {"error": repr(e)[:200]}
        with GpuSampler(None if a.smoke else 0) as g:
            t0 = time.perf_counter()
            with ThreadPoolExecutor(conc) as ex:
                res = list(ex.map(one, work))
            wall = time.perf_counter() - t0
        ok = [r for r in res if "error" not in r]
        row = {"requests": n, "failed": n - len(ok), "wall_s": round(wall, 1),
               "output_tokens_per_s": round(sum(r["completion_tokens"] or 0 for r in ok) / wall, 1),
               "requests_per_s": round(len(ok) / wall, 3), **g.summary()}
        for key, sel in (("prompt", ("short", "long")), ("max_tokens", (32, 128, 512))):
            for v in sel:
                rs = [r for r in ok if r[key] == v]
                row[f"{key}={v}"] = {"n": len(rs), "ttft_ms_p50": ms(pct([r["ttft"] for r in rs], 50)),
                                     "ttft_ms_p95": ms(pct([r["ttft"] for r in rs], 95)),
                                     "tpot_ms_p50": ms(pct([r["tpot"] for r in rs], 50)),
                                     "e2e_ms_p50": ms(pct([r["e2e"] for r in rs], 50)),
                                     "e2e_ms_p95": ms(pct([r["e2e"] for r in rs], 95))}
        itl = [x for r in ok for x in r["itl"]]
        row["itl_ms_p50"], row["itl_ms_p99"] = ms(pct(itl, 50)), ms(pct(itl, 99))
        out[f"concurrency_{conc}"] = row
        log(f"serving c{conc}: {json.dumps(row)}")
    save("serving_mixed", out)


# --------------------------------------------------------------------------- soak

def soak_paths(a):
    return os.path.join(a.work, "soak_src"), os.path.join(a.work, "soak_index"), os.path.join(a.work, "soak_extra")


def tenant_of(title):
    return f"t{zlib.crc32(title.encode()) % 4}"


def soak_questions(a):
    """(questions [(question, tenant)], passages for the index, passages added mid-run)."""
    from nyaya import rag_eval
    if a.smoke:
        rng = random.Random(0)
        words = [f"w{i}" for i in range(500)]
        paras = [(f"T{i}", " ".join(rng.choice(words) for _ in range(25))) for i in range(400)]
        qs = [(" ".join(p[1].split()[:5]), tenant_of(p[0])) for p in paras[:100]]
        return qs, paras[:350], paras[350:]
    items = rag_eval.load_hotpot("validation", n=None, seed=0, cache=os.path.join(a.data, "hotpot"))
    pool, extra = items[700:3700], items[3700:3720]
    seen, paras = set(), []
    for it in pool:
        for t, x in it["paragraphs"]:
            if t not in seen:
                seen.add(t)
                paras.append((t, x))
    add = [(t, x) for it in extra for t, x in it["paragraphs"] if t not in seen][:200]
    return [(it["question"], tenant_of(it["gold"][0])) for it in pool], paras, add


def step_soak_index(a):
    from nyaya import rag_eval
    src, idx, extra = soak_paths(a)
    _, paras, add = soak_questions(a)
    for root, items in ((src, paras), (extra, add)):
        shutil.rmtree(root, ignore_errors=True)
        for i, (t, x) in enumerate(items):
            d = os.path.join(root, tenant_of(t))
            os.makedirs(d, exist_ok=True)
            with open(os.path.join(d, f"p{i:06d}.txt"), "w") as f:
                f.write(rag_eval.passage(t, x))
    from nyaya import index as ix
    if a.smoke:
        from test_system import _FakeAxon
        ax = _FakeAxon()
    else:
        from round6 import make_axon
        ax = make_axon(False, record=False)
    rep = ix.build(src, idx, ax, tenants_from_dirs=True, log=log)
    save("soak_index", {"passages": rep["chunks"], "seconds": rep["seconds"], "tenants": 4, "extra_files": len(add)})


class AxonServer:
    def __init__(self, a, env_extra, logname):
        src, idx, _ = soak_paths(a)
        self.keys = {f"t{i}": secrets.token_urlsafe(24) for i in range(4)}
        self.mkey = secrets.token_urlsafe(24)
        self.port = 8080
        env = {**os.environ, "AXON_INDEX": idx, "AXON_TENANT_KEYS": ",".join(f"{t}:{k}" for t, k in self.keys.items()),
               "AXON_METRICS_KEY": self.mkey, "AXON_RATE_PER_MIN": "0", "AXON_MAX_INFLIGHT": "16",
               "AXON_MODEL": "lightonai/LateOn", "AXON_MODEL_REVISION": "62911e105059585d244384c7d17826e35f669c17",
               "NYAYA_LLM_BASE_URL": a.llm_base, "NYAYA_LLM_MODEL": SERVED, "CUDA_VISIBLE_DEVICES": "0",
               **env_extra}
        target = ["scripts/r7_fake_server.py"] if a.smoke else ["-m", "uvicorn", "nyaya.server:app"]
        self.f = open(os.path.join(ROOT, "logs", f"{logname}.log"), "a")
        self.p = subprocess.Popen([PY, *target, "--host", "127.0.0.1", "--port", str(self.port)], cwd=ROOT, env=env,
                                  stdout=self.f, stderr=subprocess.STDOUT, start_new_session=True)

    def call(self, path, body=None, key=None, timeout=300):
        req = urllib.request.Request(f"http://127.0.0.1:{self.port}{path}", method="POST" if body else "GET",
                                     data=json.dumps(body).encode() if body else None,
                                     headers={"Content-Type": "application/json",
                                              **({"Authorization": f"Bearer {key}"} if key else {})})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:  # our own local server  # nosec B310
                return r.status, r.read()
        except urllib.error.HTTPError as e:
            return e.code, e.read()
        except Exception as e:
            return None, repr(e).encode()

    def wait_ready(self, seconds=1200):
        t0 = time.time()
        while time.time() - t0 < seconds:
            if self.p.poll() is not None:
                return False
            st, body = self.call("/ready")
            if st == 200:
                return True
            if b'"failed"' in body:                       # preload failed: waiting longer changes nothing
                return False
            time.sleep(3)
        return False

    def metrics(self):
        from nyaya.metrics import parse
        st, body = self.call("/metrics", key=self.mkey)
        return parse(body.decode()) if st == 200 else {}

    def stop(self):
        import signal
        with contextlib.suppress(ProcessLookupError):
            os.killpg(self.p.pid, signal.SIGTERM)
        with contextlib.suppress(subprocess.TimeoutExpired):
            self.p.wait(30)
        with contextlib.suppress(ProcessLookupError):
            os.killpg(self.p.pid, signal.SIGKILL)
        self.f.close()


def drive(srv, questions, clients, seconds, records, stop_flag=None):
    """Closed-loop clients: each sends its next question as soon as the last one is answered."""
    rng = random.Random(len(records))
    order = list(range(len(questions)))
    rng.shuffle(order)
    nxt = [0]
    lock = threading.Lock()
    t_end = time.time() + seconds

    def client():
        while time.time() < t_end and not (stop_flag and stop_flag.is_set()):
            with lock:
                q, tenant = questions[order[nxt[0] % len(order)]]
                nxt[0] += 1
            t = time.time()
            st, body = srv.call("/ask", {"question": q, "k": 5}, key=srv.keys[tenant])
            rec = {"t": t, "status": st, "s": time.time() - t}
            if st == 200:
                out = json.loads(body)
                rec.update(ttft=(out.get("timing") or {}).get("ttft"), abstained=out["abstained"],
                           leaks=sum(1 for s in out["sources"] if not s["source"].startswith(tenant + "/")),
                           verified=(out.get("verification") or {}).get("supported"))
            with lock:
                records.append(rec)
    th = [threading.Thread(target=client, daemon=True) for _ in range(clients)]
    for x in th:
        x.start()
    for x in th:
        x.join()


def summarize(recs, t0, minutes=True):
    ok = [r for r in recs if r["status"] == 200]
    codes = {}
    for r in recs:
        codes[str(r["status"])] = codes.get(str(r["status"]), 0) + 1
    out = {"requests": len(recs), "status": codes, "error_rate": round(1 - len(ok) / max(len(recs), 1), 5),
           "latency_ms": {q: ms(pct([r["s"] for r in ok], q)) for q in (50, 95, 99)},
           "ttft_ms": {q: ms(pct([r.get("ttft") for r in ok], q)) for q in (50, 95)},
           "abstained": round(sum(bool(r.get("abstained")) for r in ok) / max(len(ok), 1), 4),
           "cross_tenant_results": sum(r.get("leaks", 0) for r in ok)}
    if minutes and recs:
        per = {}
        for r in recs:
            per.setdefault(int((r["t"] - t0) // 60), []).append(r)
        out["per_minute"] = [{"minute": m, "requests": len(v), "errors": sum(r["status"] != 200 for r in v),
                              "p50_ms": ms(pct([r["s"] for r in v if r["status"] == 200], 50)),
                              "p99_ms": ms(pct([r["s"] for r in v if r["status"] == 200], 99))}
                             for m, v in sorted(per.items())]
    return out


def step_soak(a):
    from nyaya.llm_bench import GpuSampler
    qs, _, _ = soak_questions(a)
    src, idx, extra = soak_paths(a)
    minutes = 2 if a.smoke else 30
    srv = AxonServer(a, {}, "r7_axon_soak")
    try:
        if not srv.wait_ready():
            save("soak", {"error": "Axon server did not become ready (logs/r7_axon_soak.log)"})
            return
        recs, samples, upd = [], [], {}
        stop = threading.Event()

        def sampler():
            while not stop.is_set():
                m = srv.metrics()
                samples.append({"t": time.time(), "rss": m.get(("process_resident_memory_bytes", "")),
                                "version": m.get(("axon_index_version", "")),
                                "rejected": m.get(("axon_ask_rejected_total", ""))})
                stop.wait(15)

        def updater():
            stop.wait(minutes * 30)                               # half way
            if stop.is_set():
                return
            files = sorted(os.path.join(r, f) for r, _, fs in os.walk(os.path.join(src, "t2")) for f in fs)[:100]
            for f in files:
                os.remove(f)
            for r, _, fs in os.walk(extra):
                for f in fs:
                    d = os.path.join(src, os.path.relpath(r, extra))
                    os.makedirs(d, exist_ok=True)
                    shutil.copy(os.path.join(r, f), os.path.join(d, "x_" + f))
            v0 = srv.metrics().get(("axon_index_version", ""))
            t = time.time()
            if a.smoke:
                from test_system import _FakeAxon

                from nyaya import index as ix
                rep = ix.update(src, idx, _FakeAxon())
                rc = 0
            else:
                rc = subprocess.run([PY, "-m", "nyaya.rag", "update", "--src", src, "--index", idx], cwd=ROOT,
                                    env={**os.environ, "CUDA_VISIBLE_DEVICES": "0"}, capture_output=True,
                                    text=True).returncode
                rep = None
            t_upd = time.time()
            while time.time() - t_upd < 300 and srv.metrics().get(("axon_index_version", "")) == v0:
                time.sleep(1)
            upd.update(start=t, update_seconds=round(t_upd - t, 1), rc=rc, report=rep,
                       version_before=v0, version_after=srv.metrics().get(("axon_index_version", "")),
                       served_after_seconds=round(time.time() - t_upd, 1), deleted=len(files))
        th = [threading.Thread(target=f, daemon=True) for f in (sampler, updater)]
        for x in th:
            x.start()
        t0 = time.time()
        with GpuSampler(None if a.smoke else 0) as g:
            drive(srv, qs, 8, minutes * 60, recs)
        stop.set()
        res = {"minutes": minutes, "clients": 8, **summarize(recs, t0), "gpu0": g.summary(), "update": upd}
        if upd.get("start"):
            win = [r for r in recs if upd["start"] <= r["t"] <= upd["start"] + 120]
            res["during_update"] = summarize(win, t0, minutes=False)
        first = [r for r in recs if r["t"] - t0 < 300 and r["status"] == 200]
        last = [r for r in recs if r["t"] - t0 > minutes * 60 - 300 and r["status"] == 200]
        res["drift"] = {"first_5_min_p50_ms": ms(pct([r["s"] for r in first], 50)),
                        "last_5_min_p50_ms": ms(pct([r["s"] for r in last], 50)),
                        "first_5_min_p99_ms": ms(pct([r["s"] for r in first], 99)),
                        "last_5_min_p99_ms": ms(pct([r["s"] for r in last], 99))}
        rss = [(s["t"], s["rss"]) for s in samples if s["rss"]]
        if len(rss) > 3:
            n = len(rss)
            mt, mr = sum(t for t, _ in rss) / n, sum(r for _, r in rss) / n
            slope = sum((t - mt) * (r - mr) for t, r in rss) / max(sum((t - mt) ** 2 for t, _ in rss), 1e-9)
            res["memory"] = {"rss_start_mb": round(rss[0][1] / 2 ** 20), "rss_end_mb": round(rss[-1][1] / 2 ** 20),
                             "rss_max_mb": round(max(r for _, r in rss) / 2 ** 20),
                             "slope_mb_per_hour": round(slope * 3600 / 2 ** 20, 1)}
        save("soak", res)
        log(f"soak: {json.dumps({k: v for k, v in res.items() if k != 'per_minute'})}")

        burst, rec2 = [], []
        tb = time.time()
        drive(srv, qs, 48, 20 if a.smoke else 60, burst)
        drive(srv, qs, 8, 20 if a.smoke else 60, rec2)
        b = summarize(burst, tb, minutes=False)
        accepted = [r["s"] for r in burst if r["status"] == 200]
        rejected = [r["s"] for r in burst if r["status"] == 503]
        b.update(clients=48, max_inflight=16, rejected_503=len(rejected),
                 rejection_ms_p99=ms(pct(rejected, 99)), accepted_p99_ms=ms(pct(accepted, 99)),
                 other_errors=sum(1 for r in burst if r["status"] not in (200, 503)))
        save("burst", {"burst": b, "recovery": summarize(rec2, tb, minutes=False)})
        log(f"burst: {json.dumps(b)}")
    finally:
        srv.stop()


def step_soak_verify(a):
    qs, _, _ = soak_questions(a)
    thr = (load().get("verifier_bench") or {}).get("threshold_f1", 0.5)
    srv = AxonServer(a, {"AXON_VERIFY": "1", "AXON_VERIFY_THRESHOLD": str(thr)}, "r7_axon_verify")
    try:
        if not srv.wait_ready():
            save("soak_verify", {"error": "server with verification did not become ready (logs/r7_axon_verify.log)"})
            return
        recs = []
        t0 = time.time()
        drive(srv, qs, 8, 60 if a.smoke else 300, recs)
        m = srv.metrics()
        s = m.get(('rag_stage_seconds_sum', '{stage="verify"}'))
        c = m.get(('rag_stage_seconds_count', '{stage="verify"}'))
        res = {**summarize(recs, t0, minutes=False), "threshold": thr,
               "verify_ms_mean": ms(s / c) if s and c else None,
               "withheld_unsupported": m.get(('rag_events_total', '{event="unsupported_answer_withheld"}'), 0.0)}
        save("soak_verify", res)
        log(f"soak with verification: {json.dumps(res)}")
    finally:
        srv.stop()


# --------------------------------------------------------------------------- PLAID compaction (GPU 1)

def step_compaction(a):
    import round6

    from nyaya import index as ix
    from nyaya.rag import AxonRAG
    n = 1200 if a.smoke else 20_000
    ns = argparse.Namespace(smoke=a.smoke, data=a.data, sizes=[n])
    paras, evals = round6.corpus(ns)
    ax = round6.make_axon(a.smoke, record=False)
    src, out = os.path.join(a.work, "cmp_src"), os.path.join(a.work, "cmp_index")
    shutil.rmtree(src, ignore_errors=True)
    shutil.rmtree(out, ignore_errors=True)
    round6.write_files(src, paras, n)
    b = ix.build(src, out, ax, plaid=True, log=log)
    rag = AxonRAG(out, axon=ax)
    before_e, _ = round6.recall(rag, evals, approximate=False)
    before_p, _ = round6.recall(rag, evals, approximate=True) if rag.plaid else (None, None)
    rng = random.Random(5)
    n_eval = max(p for _, g in evals for p in g) + 1
    pick = rng.sample(range(n_eval, n), n // 100)
    for i in pick[: len(pick) // 2]:
        os.remove(os.path.join(src, f"p{i:07d}.txt"))
    for i in pick[len(pick) // 2:]:
        with open(os.path.join(src, f"p{i:07d}.txt"), "a") as f:
            f.write(" Revised.")
    up = ix.update(src, out, ax)
    c = ix.compact(out)
    rag = AxonRAG(out, axon=ax)
    res = {"passages": n, "build_seconds": b["seconds"], "plaid": b["plaid"], "update": up, "compact": c,
           "recall@100_exact_before": round(before_e, 4), "recall@100_exact_after": round(round6.recall(rag, evals, False)[0], 4),
           "recall@100_plaid_before": None if before_p is None else round(before_p, 4),
           "recall@100_plaid_after": round(round6.recall(rag, evals, True)[0], 4) if rag.plaid else None,
           "plaid_in_use_after": rag.plaid is not None}
    save("compaction", res)
    log(f"compaction: {json.dumps(res)}")


# --------------------------------------------------------------------------- orchestration

def step(name, args, gpu, timeout, a):
    if done(name):
        log(f"{name}: already done")
        return True
    log(f"{name}: start")
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)} if gpu is not None else dict(os.environ)
    with open(os.path.join(ROOT, "logs", f"r7_{name}.log"), "a") as f:
        try:
            rc = subprocess.run([PY, os.path.abspath(__file__), *args, *(["--smoke"] if a.smoke else []),
                                 "--work", a.work, "--data", a.data, "--llm_base", a.llm_base],
                                cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT, timeout=timeout).returncode
        except subprocess.TimeoutExpired:
            rc = -9
    if rc != 0 and not done(name):
        save(name, {"error": f"exit code {rc} (logs/r7_{name}.log)"})
    log(f"{name}: {'done' if done(name) else 'FAILED, see logs/r7_' + name + '.log'}")
    return done(name)


def round6_run(args, gpu, logname, a):
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)}
    with open(os.path.join(ROOT, "logs", f"{logname}.log"), "a") as f:
        return subprocess.run([PY, "scripts/round6.py", *args, *(["--smoke"] if a.smoke else [])], cwd=ROOT, env=env,
                              stdout=f, stderr=subprocess.STDOUT).returncode


def lane_gpu1(a, soak_done):
    rc = round6_run(["--skip", "evidence", "scale", "update"], 1, "r7_deps", a)
    log(f"deps (round 6): exit {rc}")
    step("compaction", ["--step", "compaction"], 1, 3 * 3600, a)
    step("verify_real", ["--step", "verify"], 1, 3 * 3600, a)
    soak_done.wait(timeout=4 * 3600)                    # 300k needs host memory the load test is using
    sizes = ["100000", "300000"] if not a.smoke else ["300", "1500"]
    rc = round6_run(["--skip", "evidence", "update", "deps", "--no_plaid", "--max_build_minutes", "600",
                     "--sizes", *sizes], 1, "r7_scale", a)
    log(f"scale (round 6): exit {rc}")


def start_llm(a):
    """SGLang (round 5's tested set-up), with room left on the GPU for the Axon server."""
    if a.smoke:
        f = open(os.path.join(ROOT, "logs", "r7_fake_llm.log"), "a")
        p = subprocess.Popen([PY, "-I", "deploy/ci/fake_llm.py"], cwd=ROOT, stdout=f, stderr=subprocess.STDOUT,
                             start_new_session=True)
        a.llm_base = "http://127.0.0.1:8000/v1"
        time.sleep(2)
        return p
    import round5 as r5
    r5.STATUS = os.path.join(ROOT, "results", "r7_status.json")
    r5.freeze_env = lambda py, name: subprocess.run(
        [PY, "-m", "uv", "pip", "freeze", "--python", py],
        stdout=open(os.path.join(ROOT, "results", f"r7_pip_freeze_{name}.txt"), "w"), stderr=subprocess.DEVNULL)
    py = r5.sglang_python()
    if py is None:
        return None
    for flags in r5.SGLANG_ATTEMPTS:
        cmd = [py, "-m", "sglang.launch_server", "--model-path", LLM, "--served-model-name", SERVED, "--host", "127.0.0.1",
               "--port", "30000", "--dtype", "half", "--context-length", "4096", "--mem-fraction-static", "0.70",
               "--max-running-requests", "32", "--enable-metrics", *flags]
        s = r5.Server(cmd, {**r5.gpu_env(0), **r5.cuda_link_env()}, "r7_sglang", "http://127.0.0.1:30000/v1")
        if s.ready(SERVED, timeout=r5.READY_TIMEOUT):
            save("sglang", {"flags": flags, "mem_fraction_static": 0.70})
            return s
        s.stop()
    return None


def lane_gpu0(a, soak_done):
    try:
        step("soak_index", ["--step", "soak_index"], 0, 3600, a)
        llm = start_llm(a)
        if llm is None:
            save("sglang", {"error": "SGLang did not start (logs/r7_sglang.log, logs/sglang_install.log)"})
            return
        try:
            step("serving_mixed", ["--step", "serving"], None, 3 * 3600, a)
            step("judge", ["--step", "judge"], None, 3600, a)
            if done("soak_index"):
                step("soak", ["--step", "soak"], None, 3 * 3600, a)
            soak_done.set()
            for _ in range(240):                       # the threshold comes from the GPU 1 verifier step
                if done("verifier_bench") or "error" in (load().get("verify_real") or {}):
                    break
                time.sleep(15)
            if done("soak_index"):
                step("soak_verify", ["--step", "soak_verify"], None, 3600, a)
        finally:
            (llm.stop() if hasattr(llm, "stop") else llm.terminate())
    finally:
        soak_done.set()


# --------------------------------------------------------------------------- summary

def summary():
    r = load()
    r6 = json.load(open(os.path.join(ROOT, "results", "r6.json"))) if os.path.exists(os.path.join(ROOT, "results", "r6.json")) else {}
    L = ["# Round 6b + 7", ""]
    for k in ("deps", "scale_300000"):
        if k in r6:
            L += [f"## Round 6: {k}", "", "```", json.dumps(r6[k], indent=1)[:3000], "```", ""]
    for k in ("compaction", "serving_mixed", "soak", "burst", "soak_verify", "verifier_bench", "judge", "verify_real",
              "soak_index", "sglang"):
        if k in r:
            v = r[k]
            if k == "soak" and isinstance(v, dict):
                v = {kk: vv for kk, vv in v.items() if kk != "per_minute"}
            L += [f"## {k}", "", "```", json.dumps(v, indent=1)[:6000], "```", ""]
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--step", default=None)
    ap.add_argument("--work", default=None)
    ap.add_argument("--data", default=os.path.join(ROOT, "data"))
    ap.add_argument("--llm_base", default="http://127.0.0.1:30000/v1")
    ap.add_argument("--summary_only", action="store_true")
    a = ap.parse_args()
    a.work = a.work or ("/tmp/r7_smoke" if a.smoke else "/tmp/r7")  # single-user Kaggle VM scratch  # nosec B108
    for d in (a.work, os.path.join(ROOT, "logs"), os.path.join(ROOT, "results")):
        os.makedirs(d, exist_ok=True)
    if a.step:
        if a.smoke and a.step in ("soak", "soak_verify", "serving", "judge"):
            a.llm_base = "http://127.0.0.1:8000/v1"
        {"verify": step_verify, "judge": step_judge, "serving": step_serving, "soak_index": step_soak_index,
         "soak": step_soak, "soak_verify": step_soak_verify, "compaction": step_compaction}[a.step](a)
        return
    if a.summary_only:
        print(summary())
        return
    if not os.path.exists(PROMPTS):
        raise SystemExit(f"{PROMPTS} is missing: it holds round 5's prompts (copy it into results/)")
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    soak_done = threading.Event()
    t1 = threading.Thread(target=lane_gpu1, args=(a, soak_done))
    t1.start()
    lane_gpu0(a, soak_done)
    t1.join()
    md = summary()
    with open(os.path.join(ROOT, "results", "ROUND7.md"), "w") as f:
        f.write(md)
    print("\n" + md, flush=True)


if __name__ == "__main__":
    main()
