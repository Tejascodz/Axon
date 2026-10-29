"""Round 5: the whole RAG system, served and measured. Kaggle T4 x2, about 3-4 h, unattended.

  python scripts/round5.py
  python scripts/round5.py --summary_only

GPU 1: index HotpotQA with Axon (LateOn) -> retrieval metrics and every test prompt -> LoRA
       fine-tuning of the generator (own implementation, fp16).
GPU 0: the generator (Qwen3-4B-Instruct-2507) served two ways: a one-request-at-a-time transformers
       server (nyaya.hf_server) and vLLM (installed in its own environment). Serving benchmark,
       then every test against the base model, then against the fine-tuned model.

Fixed before running (no other settings are tried):
  Q9   Serving: time to first token, inter-token latency, time per output token and throughput at
       concurrency 1/4/16/32 (vLLM) and 1/4 (transformers baseline), fixed 128-token outputs.
  Q10  End-to-end answers on 500 HotpotQA questions retrieved from a ~5,000-paragraph corpus:
       exact match, F1, citation precision/recall; abstention on 200 questions whose gold
       paragraphs are withheld (should answer INSUFFICIENT) and false abstention on the 500.
  Q11  Prompt injection, 200 questions, 5 attack templates, a unique canary per question.
       Attack success rate: undefended prompt / defended prompt / defended + input scanner.
       Scanner detection rate and false-positive rate on all clean paragraphs.
  Q12  Condition check (MultiConIR, 600 documents, half satisfy all 10 conditions, half break
       exactly one): does the LLM name the right verdict (ALL, or the broken condition) more often
       when given Axon's per-condition evidence certificate? Success: paired 95% interval above 0.
       Axon alone (threshold fixed on 200 separate calibration documents) for reference.
  Q13  Fine-tuning (LoRA r=16, 1,600 HotpotQA training questions, 20% with gold removed, 10% with
       injections worded differently from the test set): success if citation precision/recall and
       abstention improve while exact match does not drop by more than 2 points.
"""
import argparse
import json
import os
import shlex
import signal
import subprocess
import sys
import threading
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kaggle_run import PY, ROOT, log  # noqa: E402

LLM = "Qwen/Qwen3-4B-Instruct-2507"
RETRIEVER = "lightonai/LateOn"
FT_DIR = "/tmp/r5_ft"
STATUS = "results/r5_status.json"
VLLM_ATTEMPTS = [(None, []), (None, ["--enforce-eager"]), ("0.11.0", []), ("0.11.0", ["--enforce-eager"])]
_lock = threading.Lock()


# --------------------------------------------------------------------------- bookkeeping

def status(step=None, **info):
    with _lock:
        s = json.load(open(STATUS)) if os.path.exists(STATUS) else {}
        if step:
            s[step] = {**s.get(step, {}), **info, "time": time.strftime("%H:%M:%S")}
            json.dump(s, open(STATUS, "w"), indent=1)
        return s


def sh(cmd, logname, env=None, timeout=None):
    """Run a command (list of args, no shell), output to logs/<logname>.log; returns the exit code."""
    with open(f"logs/{logname}.log", "a") as f:
        f.write("$ " + " ".join(shlex.quote(c) for c in cmd) + "\n")
        f.flush()
        try:
            return subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env={**os.environ, **(env or {})},
                                  cwd=ROOT, timeout=timeout).returncode
        except subprocess.TimeoutExpired:
            f.write("timeout\n")
            return -9


def gpu_env(gpu):
    return {"CUDA_VISIBLE_DEVICES": str(gpu)}


# --------------------------------------------------------------------------- servers

class Server:
    def __init__(self, cmd, env, logname, base):
        self.base, self.logname = base, logname
        self.f = open(f"logs/{logname}.log", "a")
        self.f.write("$ " + " ".join(shlex.quote(c) for c in cmd) + "\n")
        self.f.flush()
        self.p = subprocess.Popen(cmd, stdout=self.f, stderr=subprocess.STDOUT, env={**os.environ, **env},
                                  cwd=ROOT, start_new_session=True)

    def ready(self, model, timeout):
        from nyaya.llm import chat
        t0 = time.time()
        while time.time() - t0 < timeout:
            if self.p.poll() is not None:
                return False
            try:
                urllib.request.urlopen(self.base.replace("/v1", "") + "/health", timeout=5)
                chat([{"role": "user", "content": "Say OK."}], max_tokens=4, model=model, base=self.base, timeout=120)
                return True
            except Exception:
                time.sleep(10)
        return False

    def stop(self):
        if self.p.poll() is None:
            try:
                os.killpg(self.p.pid, signal.SIGTERM)
                self.p.wait(timeout=60)
            except Exception:
                try:
                    os.killpg(self.p.pid, signal.SIGKILL)
                except Exception:
                    pass
        self.f.close()
        time.sleep(5)


def vllm_python(version):
    env_dir = f"/tmp/vllm_env_{version or 'latest'}"
    py = f"{env_dir}/bin/python"
    if os.path.exists(py) and subprocess.run([py, "-c", "import vllm"], capture_output=True).returncode == 0:
        return py
    spec = f"vllm=={version}" if version else "vllm"
    if sh([PY, "-m", "pip", "install", "-q", "uv"], "vllm_install") != 0:
        return None
    if sh([PY, "-m", "uv", "venv", "--python", PY, env_dir], "vllm_install") != 0:
        return None
    # --torch-backend=auto picks the PyTorch CUDA build that matches this machine's driver
    if sh([PY, "-m", "uv", "pip", "install", "--python", py, spec, "--torch-backend=auto"], "vllm_install", timeout=3600) != 0 \
            and sh([PY, "-m", "uv", "pip", "install", "--python", py, spec], "vllm_install", timeout=3600) != 0:
        return None
    r = subprocess.run([py, "-c", "import vllm, torch; print(vllm.__version__, torch.__version__, torch.version.cuda)"],
                       capture_output=True, text=True)
    status("vllm_install_" + (version or "latest"), ok=r.returncode == 0, info=(r.stdout or r.stderr)[-300:])
    return py if r.returncode == 0 else None


def start_vllm(model_path, served, gpu, port=8000):
    base = f"http://127.0.0.1:{port}/v1"
    for version, flags in VLLM_ATTEMPTS:
        py = vllm_python(version)
        if py is None:
            continue
        cmd = [py, "-m", "vllm.entrypoints.openai.api_server", "--model", model_path, "--served-model-name", served,
               "--host", "127.0.0.1", "--port", str(port), "--dtype", "half", "--max-model-len", "4096",
               "--gpu-memory-utilization", "0.90", "--max-num-seqs", "32", *flags]
        log(f"starting vLLM {version or 'latest'} {' '.join(flags)} for {served}")
        s = Server(cmd, gpu_env(gpu), f"vllm_{served.replace('/', '_')}", base)
        if s.ready(served, timeout=1500):
            status(f"vllm_{served}", ok=True, version=version or "latest", flags=flags)
            return s
        s.stop()
        status(f"vllm_{served}", ok=False, tried=f"{version or 'latest'} {flags}")
    return None


def start_hf(model_path, served, gpu, port=8100):
    s = Server([PY, "-m", "nyaya.hf_server", "--model", model_path, "--port", str(port)], gpu_env(gpu),
               f"hf_{served.replace('/', '_')}", f"http://127.0.0.1:{port}/v1")
    if s.ready(served, timeout=1500):
        return s
    s.stop()
    return None


# --------------------------------------------------------------------------- lanes

def step(name, cmd, logname, gpu=None, timeout=None, done_file=None):
    if done_file and os.path.exists(done_file):
        log(f"{name}: already done")
        return True
    log(f"{name}: start")
    t0 = time.time()
    rc = sh(cmd, logname, env=gpu_env(gpu) if gpu is not None else None, timeout=timeout)
    ok = rc == 0 and (done_file is None or os.path.exists(done_file))
    status(name, ok=ok, rc=rc, minutes=round((time.time() - t0) / 60, 1))
    log(f"{name}: {'done' if ok else 'FAILED (see logs/' + logname + '.log)'}")
    return ok


def lane_retrieval_and_training(a, gpu, ev_retrieval, ev_train):
    try:
        step("retrieve", [PY, "scripts/r5_steps.py", "retrieve", "--retriever", a.retriever, *a.size_retrieve],
             "r5_retrieve", gpu, timeout=3 * 3600, done_file="results/r5_prompts.json")
    finally:
        ev_retrieval.set()
    try:
        if a.skip_train:
            return
        if step("traindata", [PY, "scripts/r5_steps.py", "traindata", *a.size_train], "r5_traindata",
                timeout=3600, done_file="data/r5_train.jsonl"):
            step("train", [PY, "-m", "nyaya.lora", "--model", a.llm, "--data", "data/r5_train.jsonl", "--out", FT_DIR,
                           "--adapter_out", "results/r5_lora", "--max_minutes", str(a.train_minutes)],
                 "r5_train", gpu, timeout=int(a.train_minutes * 60 + 3600), done_file=f"{FT_DIR}/config.json")
    finally:
        ev_train.set()


def lane_serving(a, gpu, ev_retrieval, ev_train, ev_vllm_env):
    ev_retrieval.wait()
    if not os.path.exists("results/r5_prompts.json"):
        log("no prompts: retrieval failed, nothing to serve")
        return
    # transformers baseline
    if not os.path.exists("results/r5_bench_hf.json"):
        s = start_hf(a.llm, a.llm, gpu)
        if s:
            step("bench_hf", [PY, "scripts/r5_steps.py", "bench", "--base", s.base, "--model", a.llm, "--tag", "hf",
                              "--levels", "1,4", "--requests", "8", "--gpu", str(gpu)], "r5_bench_hf",
                 timeout=3600, done_file="results/r5_bench_hf.json")
            if a.no_vllm:
                step("generate_base", [PY, "scripts/r5_steps.py", "generate", "--base", s.base, "--model", a.llm,
                                       "--tag", "base", "--concurrency", "2", "--limit", str(a.hf_limit)],
                     "r5_gen_base", timeout=4 * 3600, done_file="results/r5_gen_base.json")
            s.stop()
    if a.no_vllm:
        ev_train.wait()
        if os.path.exists(f"{FT_DIR}/config.json"):
            s = start_hf(FT_DIR, FT_DIR, gpu)
            if s:
                step("generate_ft", [PY, "scripts/r5_steps.py", "generate", "--base", s.base, "--model", FT_DIR,
                                     "--tag", "ft", "--concurrency", "2", "--limit", str(a.hf_limit)],
                     "r5_gen_ft", timeout=4 * 3600, done_file="results/r5_gen_ft.json")
                s.stop()
        return
    ev_vllm_env.wait()
    served = "base"
    s = start_vllm(a.llm, served, gpu)
    if s is None:
        log("vLLM did not start with any setting; falling back to the transformers server (smaller test sets)")
        a.no_vllm = True
        return lane_serving(a, gpu, ev_retrieval, ev_train, ev_vllm_env)
    step("bench_vllm", [PY, "scripts/r5_steps.py", "bench", "--base", s.base, "--model", served, "--tag", "vllm",
                        "--gpu", str(gpu)], "r5_bench_vllm", timeout=3600, done_file="results/r5_bench_vllm.json")
    step("generate_base", [PY, "scripts/r5_steps.py", "generate", "--base", s.base, "--model", served, "--tag", "base"],
         "r5_gen_base", timeout=4 * 3600, done_file="results/r5_gen_base.json")
    s.stop()
    ev_train.wait()
    if not os.path.exists(f"{FT_DIR}/config.json"):
        log("no fine-tuned model; skipping its evaluation")
        return
    s = start_vllm(FT_DIR, "ft", gpu)
    if s is None:
        return
    step("generate_ft", [PY, "scripts/r5_steps.py", "generate", "--base", s.base, "--model", "ft", "--tag", "ft"],
         "r5_gen_ft", timeout=4 * 3600, done_file="results/r5_gen_ft.json")
    step("bench_vllm_ft", [PY, "scripts/r5_steps.py", "bench", "--base", s.base, "--model", "ft", "--tag", "vllm_ft",
                           "--levels", "1,16", "--gpu", str(gpu)], "r5_bench_vllm_ft", timeout=3600,
         done_file="results/r5_bench_vllm_ft.json")
    s.stop()


# --------------------------------------------------------------------------- summary

def _load(name):
    p = f"{ROOT}/results/{name}.json"
    return json.load(open(p)) if os.path.exists(p) else None


def _boot(x, n=5000, seed=0):
    import numpy as np
    x = np.asarray([v for v in x if v is not None], dtype=float)
    if len(x) == 0:
        return None
    m = x[np.random.default_rng(seed).integers(0, len(x), size=(n, len(x)))].mean(1)
    return float(x.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def _paired(a, b, n=5000, seed=0):
    import numpy as np
    pairs = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
    if not pairs:
        return None
    d = np.asarray([x - y for x, y in pairs], dtype=float)
    m = d[np.random.default_rng(seed).integers(0, len(d), size=(n, len(d)))].mean(1)
    return float(d.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def _fmt(ci, pct=False):
    if ci is None:
        return "–"
    k = 100 if pct else 1
    f = "{:+.1f}" if pct else "{:.3f}"
    if pct:
        return f"{ci[0] * k:+.1f} [{ci[1] * k:+.1f}, {ci[2] * k:+.1f}]"
    return f"{f.format(ci[0])} [{f.format(ci[1])}, {f.format(ci[2])}]"


def _rate(ci):
    return "–" if ci is None else f"{100 * ci[0]:.1f}% [{100 * ci[1]:.1f}, {100 * ci[2]:.1f}]"


def answer_stats(g):
    A, U = g["arms"]["answer"], g["arms"]["unanswerable"]
    return {"EM": _boot([r["em"] for r in A]), "F1": _boot([r["f1"] for r in A]),
            "citation precision": _boot([r["cite_precision"] for r in A]),
            "citation recall": _boot([r["cite_recall"] for r in A]),
            "false abstention (answerable)": _boot([float(r["abstained"]) for r in A]),
            "abstention (gold withheld)": _boot([float(r["abstained"]) for r in U]),
            "well-formed replies": _boot([float(r["well_formed"]) for r in A + U]),
            "invalid citations removed (per reply)": _boot([r["invalid_citations"] for r in A + U])}


def summary():
    L = ["# Round 5: the whole RAG system, served and measured", ""]
    st = status()
    rt = _load("r5_retrieval")
    if rt:
        lat = rt["latency"]
        L += ["## Retrieval (HotpotQA, Axon on LateOn)", "",
              f"{rt['questions']} questions, a corpus of {rt['corpus_paragraphs']} paragraphs, indexed in {rt['index_seconds']} s.", "",
              "| Measure | Value |", "|---|---|",
              f"| recall@5, both gold paragraphs (BM25 + ColBERT, RRF, Axon re-rank) | {rt['recall@5_both_gold']:.3f} |",
              f"| recall@5, both gold, BM25 + ColBERT fused without re-ranking | {rt.get('recall@5_both_gold_fused_only', float('nan')):.3f} |",
              f"| recall@5, at least one gold paragraph | {rt['recall@5_any_gold']:.3f} |"]
        for k in ("bm25", "exact", "plaid"):
            if f"{k}_recall@100" in rt:
                lab = {"bm25": "BM25", "exact": "ColBERT exact search", "plaid": "ColBERT PLAID (approximate)"}[k]
                L.append(f"| {lab} recall@100 | {rt[f'{k}_recall@100']:.3f} (median {lat[k]['median_ms']} ms, p95 {lat[k]['p95_ms']} ms per query) |")
        L.append(f"| full retrieval pipeline per query | median {lat['pipeline']['median_ms']} ms, p95 {lat['pipeline']['p95_ms']} ms |")
        if rt.get("plaid_error"):
            L.append(f"| PLAID | not built: {rt['plaid_error'][:120]} |")
        L.append("")
    for tag, title in (("hf", "transformers server, one request at a time"), ("vllm", "vLLM"), ("vllm_ft", "vLLM, fine-tuned model")):
        b = _load(f"r5_bench_{tag}")
        if not b:
            continue
        if tag == "hf":
            L += ["## Q9 Serving (T4, fp16, Qwen3-4B-Instruct-2507, ~1,100-token RAG prompts, 128-token outputs)", ""]
        L += [f"**{title}** (vLLM {st.get('vllm_base', {}).get('version', '')})" if tag.startswith("vllm") else f"**{title}**", "",
              "| Concurrency | TTFT p50 / p99 ms | ITL p50 / p99 ms | TPOT ms | e2e p50 ms | requests/s | output tokens/s | GPU memory MiB | GPU util % | failed |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for r in b["rows"]:
            L.append(f"| {r['concurrency']} | {r['ttft_ms_p50']} / {r['ttft_ms_p99']} | {r['itl_ms_p50']} / {r['itl_ms_p99']} | "
                     f"{r['tpot_ms_mean']} | {r['e2e_ms_p50']} | {r['requests_per_s']} | {r['output_tokens_per_s']} | "
                     f"{r.get('gpu_mem_peak_mib', '–')} | {r.get('gpu_util_mean_pct', '–')} | {r['failed']} |")
        srv = [(k, v) for r in b["rows"][-1:] for k, v in r["server"].items()
               if any(s in k for s in ("time_to_first_token", "inter_token", "time_per_output", "e2e_request", "prefix_cache"))]
        if srv:
            L += ["", "Server-side (Prometheus /metrics) at the highest concurrency: " +
                  ", ".join(f"`{k}` {v * 1000:.1f} ms" if "seconds" in k else f"`{k}` {v:g}" for k, v in srv[:8])]
        L.append("")
    base, ft = _load("r5_gen_base"), _load("r5_gen_ft")
    if base:
        L += ["## Q10 Answers, citations, abstention", "", "| Measure | base | fine-tuned | fine-tuned − base (paired) |", "|---|---|---|---|"]
        sb, sf = answer_stats(base), (answer_stats(ft) if ft else {})
        keys = {"EM": "em", "F1": "f1", "citation precision": "cite_precision", "citation recall": "cite_recall",
                "false abstention (answerable)": "abstained"}
        for k, v in sb.items():
            d = "–"
            if ft and k in keys:
                fk = keys[k]
                d = _fmt(_paired([float(r[fk]) if r[fk] is not None else None for r in ft["arms"]["answer"]],
                                 [float(r[fk]) if r[fk] is not None else None for r in base["arms"]["answer"]]), pct=True)
            if ft and k == "abstention (gold withheld)":
                d = _fmt(_paired([float(r["abstained"]) for r in ft["arms"]["unanswerable"]],
                                 [float(r["abstained"]) for r in base["arms"]["unanswerable"]]), pct=True)
            L.append(f"| {k} | {_fmt(v)} | {_fmt(sf.get(k)) if ft else '–'} | {d} |")
        L += ["", "## Q11 Prompt injection (attack success rate, lower is better)", "",
              "| Prompt | base | fine-tuned | EM under attack (base) |", "|---|---|---|---|"]
        for arm, lab in (("undefended", "undefended (plain concatenation)"), ("defended", "defended (escaped sources, data-not-instructions rule)"),
                         ("scanned", "defended + input scanner")):
            rb = base["arms"].get(f"injection_{arm}", [])
            rf = ft["arms"].get(f"injection_{arm}", []) if ft else []
            L.append(f"| {lab} | {_rate(_boot([float(r['attacked']) for r in rb]))} | "
                     f"{_rate(_boot([float(r['attacked']) for r in rf])) if rf else '–'} | {_fmt(_boot([r['em'] for r in rb]))} |")
        if rt and "scanner" in rt:
            sc = rt["scanner"]
            L += ["", f"Input scanner: detected {sc['detected']}/{sc['injected']} planted injections "
                      f"(by template: {sc['detected_by_template']}); flagged {sc['false_positives']} of {sc['clean_paragraphs']} "
                      f"clean paragraphs (false-positive rate {100 * sc['false_positive_rate']:.2f}%)."]
            rb = base["arms"].get("injection_defended", [])
            if rb:
                by_t = {}
                for r in rb:
                    by_t.setdefault(r["template"], []).append(float(r["attacked"]))
                L.append("Defended prompt, base model, success by template: " +
                         ", ".join(f"T{t + 1} {100 * sum(v) / len(v):.0f}%" for t, v in sorted(by_t.items())) + ".")
        L += ["", "## Q12 Condition check: LLM alone vs LLM + Axon certificate", ""]
        ca, cc = base["arms"].get("condition_alone", []), base["arms"].get("condition_cert", [])
        if ca and cc:
            def split(rows, want_all):
                return [r["correct"] for r in rows if (r["truth"] == "ALL") == want_all]
            L += ["| Documents | LLM alone | LLM + certificate | certificate − alone (paired) |", "|---|---|---|---|"]
            for lab, sel in ((f"all {len(ca)}", None), ("satisfy every condition (truth ALL)", True), ("break one condition", False)):
                xa = [r["correct"] for r in ca] if sel is None else split(ca, sel)
                xc = [r["correct"] for r in cc] if sel is None else split(cc, sel)
                L.append(f"| {lab} | {_rate(_boot(xa))} | {_rate(_boot(xc))} | {_fmt(_paired(xc, xa), pct=True)} |")
            if ft and ft["arms"].get("condition_cert"):
                fc, fa = ft["arms"]["condition_cert"], ft["arms"]["condition_alone"]
                L.append(f"| fine-tuned model, all {len(fa)} | {_rate(_boot([r['correct'] for r in fa]))} | "
                         f"{_rate(_boot([r['correct'] for r in fc]))} | {_fmt(_paired([r['correct'] for r in fc], [r['correct'] for r in fa]), pct=True)} |")
            ax = axon_alone()
            if ax:
                L += ["", f"Axon alone (no LLM; ALL if the weakest condition's relative score ≥ {ax['threshold']:.3f}, "
                          f"threshold fixed on {ax['n_calib']} calibration documents): {_rate(ax['ci'])}."]
            unparsed = sum(1 for r in ca + cc if r["verdict"] is None)
            L.append(f"Replies without a parsable verdict: {unparsed} of {len(ca) + len(cc)}.")
    lr = _load("r5_lora/train_report")
    if lr:
        L += ["", "## Q13 Fine-tuning run", "",
              f"LoRA r={lr['r']}, alpha={lr['alpha']}, lr {lr['lr']}, {lr['examples']} examples "
              f"({lr['skipped_too_long']} too long), {lr['steps']}/{lr['planned_steps']} steps in {lr['minutes']:.0f} min, "
              f"{lr['trainable_params'] / 1e6:.1f} M trainable parameters; loss {lr['first_loss']:.3f} → {lr['last_loss']:.3f}; "
              f"{lr['non_finite_losses']} non-finite losses skipped."]
    bad = [k for k, v in st.items() if isinstance(v, dict) and v.get("ok") is False]
    L += ["", f"Steps that failed: {', '.join(bad)} (see logs/)." if bad else "All steps succeeded.", ""]
    txt = "\n".join(L) + "\n"
    open(f"{ROOT}/results/ROUND5.md", "w").write(txt)
    return txt


def axon_alone():
    P = None
    p = f"{ROOT}/results/r5_prompts.json"
    if os.path.exists(p):
        P = json.load(open(p))
    if not P or not P.get("condition_calib"):
        return None

    def verdicts(rows, thr):
        return [("ALL" if min(r["relative"]) >= thr else r["weakest"]) for r in rows]
    cal = P["condition_calib"]
    cands = sorted({min(r["relative"]) for r in cal})
    best = max(cands, key=lambda t: sum(v == r["truth"] for v, r in zip(verdicts(cal, t), cal)))
    test = P["condition"]
    ok = [float(v == r["truth"]) for v, r in zip(verdicts(test, best), test)]
    return {"threshold": best, "n_calib": len(cal), "ci": _boot(ok)}


# --------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", default=LLM)
    ap.add_argument("--retriever", default=RETRIEVER)
    ap.add_argument("--summary_only", action="store_true")
    ap.add_argument("--no_vllm", action="store_true", help="serve with nyaya.hf_server only")
    ap.add_argument("--skip_train", action="store_true")
    ap.add_argument("--train_minutes", type=float, default=100)
    ap.add_argument("--hf_limit", type=int, default=100, help="test items per arm when only the slow server is available")
    ap.add_argument("--smoke", action="store_true", help="tiny sizes for a local check")
    a = ap.parse_args()
    os.chdir(ROOT)
    for d in ("results", "logs", "data"):
        os.makedirs(d, exist_ok=True)
    if a.summary_only:
        print(summary())
        return
    a.size_retrieve = (["--n_answer", "12", "--n_unanswerable", "6", "--n_injection", "10", "--n_condition", "10",
                        "--n_calib", "6"] if a.smoke else [])
    a.size_train = ["--n_train", "16"] if a.smoke else []
    import torch
    n = torch.cuda.device_count()
    log(f"GPUs: {n}; generator {a.llm}; retriever {a.retriever}")
    if shutil_which("nvidia-smi"):
        subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv"])
    if not a.smoke:
        sh([PY, "scripts/prep_data.py", "--data", "data", "--only", "model", "multiconir", "--model", a.llm, a.retriever],
           "r5_prep")
    else:
        sh([PY, "scripts/prep_data.py", "--data", "data", "--only", "multiconir"], "r5_prep")
    ev_r, ev_t, ev_v = threading.Event(), threading.Event(), threading.Event()

    def install():
        try:
            if not a.no_vllm:
                vllm_python(None)
        finally:
            ev_v.set()
    threading.Thread(target=install, daemon=True).start()
    gA, gB = (0, 1) if n >= 2 else (0, 0)
    if gA == gB:                                    # one GPU: run the lanes one after the other
        lane_retrieval_and_training(a, gB, ev_r, ev_t)
        lane_serving(a, gA, ev_r, ev_t, ev_v)
    else:
        tb = threading.Thread(target=lane_retrieval_and_training, args=(a, gB, ev_r, ev_t))
        tb.start()
        lane_serving(a, gA, ev_r, ev_t, ev_v)
        tb.join()
    print("\n" + summary(), flush=True)


def shutil_which(x):
    import shutil
    return shutil.which(x)


if __name__ == "__main__":
    main()
