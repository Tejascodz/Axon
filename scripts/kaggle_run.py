"""Unattended full run for Kaggle (Save & Run All) or any machine with 1+ GPUs.

  python scripts/kaggle_run.py                    # everything (about 2-3 h on T4 x2)
  python scripts/kaggle_run.py --plan quick       # fewer steps and one seed (about 1 h)
  python scripts/kaggle_run.py --summary_only     # rebuild results/SUMMARY.md from finished jobs

How the GPUs are used: every job (one evaluation, or one fine-tune + its evaluation) goes into
a single queue, longest first. Each GPU has a worker that takes the next job as soon as it is
free, so neither GPU waits for the other. A job that fails is logged and skipped; the rest
continue. Finished work is never redone, so the script can be restarted after an interruption.

Outputs (small, in results/): one JSON per job, logs/<job>.log, results/SUMMARY.md.
Checkpoints go to --ckpt_dir (default /tmp, not saved) unless --keep_ckpt.
"""
import argparse
import json
import os
import queue
import shlex
import subprocess
import sys
import threading
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable


Q = shlex.quote                       # every user-supplied value is quoted before it reaches the shell


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


# --------------------------------------------------------------------------- jobs

def make_jobs(a, prep):
    ok = {k: v.get("ok", False) for k, v in prep.items()}
    jobs = []
    base_suites = ["andor"] + (["multiconir"] if ok.get("multiconir") else [])
    general = (["nanobeir"] if ok.get("nanobeir") else []) + (["limit-small", "limit"] if ok.get("limit") else [])
    if a.plan in ("quick", "smoke"):
        general = [s for s in general if s != "limit"]

    def zs(name, variant, qlen, suites, est, extra=""):
        q = f"--query_length {int(qlen)}" if qlen else ""
        jobs.append(dict(name=name, est=est, cmd=(
            f"{Q(PY)} scripts/eval_suite.py --zero_shot {Q(a.model)} --variant {Q(variant)} {q} --tau {float(a.tau)} "
            f"--suites {' '.join(suites)} --out results/{name}.json {a.size} {extra}")))

    # 1. training-free: the released model, only the scoring rule changes
    zs("zs_colbert_q32", "colbert", None, base_suites, 8)                       # exactly as released
    zs("zs_colbert_long", "colbert", 128, base_suites + general, 25)            # fair baseline
    zs("zs_colbert_grouped", "colbert_grouped", 128, base_suites, 8)
    zs("zs_nyaya_tok", "nyaya_tok", 128, base_suites, 14,
       extra="--tau_sweep 0.02 0.1 0.2" if ok.get("multiconir") else "")

    # 2. fine-tuned from the same checkpoint, identical data / steps / learning rates
    mix = f"--mix_jsonl data/msmarco_triples.jsonl --mix_ratio {float(a.mix_ratio)}" if ok.get("msmarco") else ""
    for seed in a.seeds:
        for variant, tag in (("colbert", "colbert_long"), ("colbert_grouped", "colbert_grouped"),
                             ("nyaya_tok", "nyaya_tok")):
            name = f"ft_{tag}_s{seed}"
            ck = Q(os.path.join(a.ckpt_dir, f"{name}.pt"))
            train = (f"[ -f {ck} ] || {Q(PY)} -m nyaya.train --backbone {Q('pylate:' + a.model)} --variant {variant} "
                     f"--steps {int(a.steps)} --batch {int(a.batch)} --lr {float(a.lr)} "
                     f"--backbone_lr {float(a.backbone_lr)} --dim 128 "
                     f"--fp16 --grad_ckpt --query_length 128 --seed {int(seed)} {mix} --save {ck}")
            ev = (f"{Q(PY)} scripts/eval_suite.py --ckpt {ck} --suites {' '.join(base_suites + general)} "
                  f"--out results/{name}.json {a.size}")
            clean = "" if a.keep_ckpt else f" && rm -f {ck}"
            jobs.append(dict(name=name, est=10 + a.steps * 0.012 + 15, cmd=f"{train} && {ev}{clean}"))
    return sorted(jobs, key=lambda j: -j["est"])


def job_done(name, status):
    return status.get(name, {}).get("state") == "done"


# --------------------------------------------------------------------------- runner

def run_all(jobs, gpus, status_path):
    status = json.load(open(status_path)) if os.path.exists(status_path) else {}
    q = queue.Queue()
    for j in jobs:
        if job_done(j["name"], status):
            log(f"[skip] {j['name']} (done earlier)")
        else:
            q.put(j)
    lock = threading.Lock()

    def save():
        with lock:
            json.dump(status, open(status_path, "w"), indent=2)

    def worker(gpu):
        while True:
            try:
                j = q.get_nowait()
            except queue.Empty:
                return
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONWARNINGS="ignore",
                       TOKENIZERS_PARALLELISM="false", TRANSFORMERS_VERBOSITY="error")
            t0 = time.time()
            status[j["name"]] = {"state": "running", "gpu": gpu, "started": time.strftime("%H:%M:%S")}
            save()
            log(f"[gpu {gpu}] start {j['name']}  (~{j['est']:.0f} min)")
            with open(f"logs/{j['name']}.log", "a") as f:
                f.write(f"$ {j['cmd']}\n")
                f.flush()
                rc = subprocess.run(["bash", "-c", j["cmd"]], cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT).returncode
            mins = round((time.time() - t0) / 60, 1)
            if rc == 0:
                status[j["name"]] = {"state": "done", "gpu": gpu, "minutes": mins}
                log(f"[gpu {gpu}] done  {j['name']} in {mins} min")
            else:
                tail = open(f"logs/{j['name']}.log").read().splitlines()[-15:]
                status[j["name"]] = {"state": "failed", "gpu": gpu, "minutes": mins, "tail": tail}
                log(f"[gpu {gpu}] FAILED {j['name']} after {mins} min; last lines:\n    " + "\n    ".join(tail))
            save()

    threads = [threading.Thread(target=worker, args=(g,), daemon=True) for g in gpus]
    for t in threads:
        t.start()
    tick = 0
    while any(t.is_alive() for t in threads):
        time.sleep(5)
        tick += 5
        if tick % 300:
            continue
        running = [f"{k}@gpu{v['gpu']}" for k, v in status.items() if v.get("state") == "running"]
        log(f"[heartbeat] running: {', '.join(running) or '-'}; queued: {q.qsize()}")
    for t in threads:
        t.join()
    return status


# --------------------------------------------------------------------------- summary

def _load(name):
    p = os.path.join(ROOT, "results", f"{name}.json")
    return json.load(open(p)) if os.path.exists(p) else None


def _f(x, d=3):
    return "–" if x is None else f"{x:.{d}f}"


def _get(r, *path):
    for k in path:
        if r is None or k not in r:
            return None
        r = r[k]
    return r


def _mean(rs, *path):
    v = [x for x in (_get(r, *path) for r in rs) if x is not None]
    if not v:
        return "–"
    m = sum(v) / len(v)
    return f"{m:.3f}" if len(v) == 1 else f"{m:.3f} ±{(max(v) - min(v)) / 2:.3f}"


def summary(seeds):
    L = ["# Nyaya results", ""]
    zs = {n: _load(f"zs_{n}") for n in ("colbert_q32", "colbert_long", "colbert_grouped", "nyaya_tok")}
    label = {"colbert_q32": "LateOn as released (32-token queries)", "colbert_long": "LateOn, full query (baseline)",
             "colbert_grouped": "+ clause grouping", "nyaya_tok": "+ clause grouping + soft AND (Nyaya)"}
    L += ["## 1. Training-free (released LateOn weights; only the scoring rule changes)", "",
          "Synthetic AND-of-OR, Recall@10 by number of conditions:", "",
          "| Scoring | all | w2 | w3 | w4 | w5 | w6 | w7 | w8 |", "|---|---|---|---|---|---|---|---|---|"]
    for n, r in zs.items():
        L.append(f"| {label[n]} | {_f(_get(r, 'andor', 'all', 'recall@10'))} | " +
                 " | ".join(_f(_get(r, 'andor', f'w{w}', 'recall@10')) for w in range(2, 9)) + " |")
    L += ["", "MultiConIR (win rates; task 1 = positive beats the hard negative for a query with k conditions):", "",
          "| Scoring | T1 k=1 | T1 k=5 | T1 k=10 | T1 mean | T2 pos>HN1 | T2 monotonicity | T3 pos>HN1 | T3 monotonicity |",
          "|---|---|---|---|---|---|---|---|---|"]
    for n, r in zs.items():
        m = _get(r, "multiconir")
        L.append(f"| {label[n]} | {_f(_get(m, 'task1', 'wr1'))} | {_f(_get(m, 'task1', 'wr5'))} | "
                 f"{_f(_get(m, 'task1', 'wr10'))} | {_f(_get(m, 'task1', 'mean'))} | "
                 f"{_f(_get(m, 'task2', 'pos_vs_hn1'))} | {_f(_get(m, 'task2', 'monotonicity'))} | "
                 f"{_f(_get(m, 'task3', 'pos_vs_hn1'))} | {_f(_get(m, 'task3', 'monotonicity'))} |")
    b = zs["colbert_long"]
    L += ["", f"Baseline on ordinary search: NanoBEIR mean nDCG@10 {_f(_get(b, 'nanobeir', 'mean_ndcg@10'))}; "
          f"LIMIT R@2 {_f(_get(b, 'limit', 'all', 'recall@2'))}, R@10 {_f(_get(b, 'limit', 'all', 'recall@10'))}. "
          "(LIMIT and NanoBEIR queries have one clause, so all scoring rules give the same result there.)", ""]
    sw = zs["nyaya_tok"]
    if sw:
        rows = [(t, _get(sw, f"multiconir@tau{t}") or (_get(sw, "multiconir") if t == 0.05 else None))
                for t in (0.02, 0.05, 0.1, 0.2)]
        L += ["Temperature sensitivity (sensitivity only; τ = 0.05 was fixed in advance):", "",
              "| τ | T1 mean | T1 k=10 | T2 pos>HN1 | T2 monotonicity |", "|---|---|---|---|---|"]
        for t, m in rows:
            L.append(f"| {t} | {_f(_get(m, 'task1', 'mean'))} | {_f(_get(m, 'task1', 'wr10'))} | "
                     f"{_f(_get(m, 'task2', 'pos_vs_hn1'))} | {_f(_get(m, 'task2', 'monotonicity'))} |")
        L.append("")
    L += [f"## 2. Fine-tuned from LateOn (mean ± half-range over seeds {seeds})", "",
          "| Variant | synth all | synth w8 | MCIR T1 mean | MCIR T1 k=10 | MCIR T2 | MCIR T3 | NanoBEIR nDCG@10 | LIMIT-small R@2 | LIMIT R@2 |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for tag, lab in (("colbert_long", "ColBERT (baseline)"), ("colbert_grouped", "+ grouping"),
                     ("nyaya_tok", "+ grouping + soft AND")):
        rs = [r for r in (_load(f"ft_{tag}_s{s}") for s in seeds) if r]
        L.append(f"| {lab} | {_mean(rs, 'andor', 'all', 'recall@10')} | {_mean(rs, 'andor', 'w8', 'recall@10')} | "
                 f"{_mean(rs, 'multiconir', 'task1', 'mean')} | {_mean(rs, 'multiconir', 'task1', 'wr10')} | "
                 f"{_mean(rs, 'multiconir', 'task2', 'pos_vs_hn1')} | {_mean(rs, 'multiconir', 'task3', 'pos_vs_hn1')} | "
                 f"{_mean(rs, 'nanobeir', 'mean_ndcg@10')} | {_mean(rs, 'limit-small', 'all', 'recall@2')} | "
                 f"{_mean(rs, 'limit', 'all', 'recall@2')} |")
    st = os.path.join(ROOT, "results", "jobs_status.json")
    if os.path.exists(st):
        s = json.load(open(st))
        bad = [k for k, v in s.items() if v.get("state") != "done"]
        L += ["", f"Jobs: {sum(v.get('state') == 'done' for v in s.values())} done, {len(bad)} not done"
              + (f" ({', '.join(bad)}; see logs/)" if bad else "") + "."]
    prep = os.path.join(ROOT, "data", "prep_status.json")
    if os.path.exists(prep):
        p = json.load(open(prep))
        miss = [k for k, v in p.items() if not v.get("ok")]
        if miss:
            L.append(f"Datasets unavailable in this run: {', '.join(miss)} (see data/prep_status.json).")
    txt = "\n".join(L) + "\n"
    open(os.path.join(ROOT, "results", "SUMMARY.md"), "w").write(txt)
    return txt


# --------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", choices=["full", "quick", "smoke"], default="full",
                    help="smoke = a few minutes, tiny sizes, only to check that everything runs")
    ap.add_argument("--model", default="lightonai/LateOn")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--seeds", type=int, nargs="+", default=None)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--backbone_lr", type=float, default=1e-5)
    ap.add_argument("--mix_ratio", type=float, default=0.5)
    ap.add_argument("--tau", type=float, default=0.05)
    ap.add_argument("--ckpt_dir", default="/tmp/nyaya_ckpt")
    ap.add_argument("--keep_ckpt", action="store_true")
    ap.add_argument("--summary_only", action="store_true")
    ap.add_argument("--keep_data", action="store_true", help="keep downloaded datasets in data/ (large) after the run")
    a = ap.parse_args()
    a.steps = a.steps or {"full": 1500, "quick": 600, "smoke": 5}[a.plan]
    a.seeds = a.seeds or ([0, 1] if a.plan == "full" else [0])
    a.size = "--n_docs 400 --q_per_width 2 --mc_rows 3 --nano scifact" if a.plan == "smoke" else ""
    os.chdir(ROOT)
    for d in ("results", "logs", a.ckpt_dir):
        os.makedirs(d, exist_ok=True)
    if a.summary_only:
        print(summary(a.seeds))
        return

    import torch
    n = torch.cuda.device_count()
    gpus = list(range(n)) or [0]
    log(f"GPUs: {n} ({', '.join(torch.cuda.get_device_name(i) for i in range(n)) or 'none, CPU only'}); "
        f"plan {a.plan}, {a.steps} steps, seeds {a.seeds}")

    log("preparing data (downloads happen once, here)")
    subprocess.run([PY, "scripts/prep_data.py", "--data", "data", "--model", a.model], cwd=ROOT)  # noqa
    prep = json.load(open("data/prep_status.json"))
    jobs = make_jobs(a, prep)
    log(f"{len(jobs)} jobs, ~{sum(j['est'] for j in jobs):.0f} GPU-minutes in total, "
        f"~{sum(j['est'] for j in jobs) / len(gpus):.0f} min on {len(gpus)} GPU(s)")
    json.dump(jobs, open("results/jobs_plan.json", "w"), indent=2)
    run_all(jobs, gpus, "results/jobs_status.json")
    print("\n" + summary(a.seeds), flush=True)
    if not a.keep_data:                       # keep the saved output small: datasets can be re-downloaded
        import shutil
        for d in ("limit", "MultiConIR", "msmarco_triples.jsonl"):
            p = os.path.join("data", d)
            shutil.rmtree(p, ignore_errors=True) if os.path.isdir(p) else (os.path.exists(p) and os.remove(p))


if __name__ == "__main__":
    main()
