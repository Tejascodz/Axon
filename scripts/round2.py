"""Round 2: is clause grouping a real, general effect? Training-free, MultiConIR only, ~2 h on T4 x2.

  python scripts/round2.py
  python scripts/round2.py --summary_only

Questions, fixed before running (no other variants or temperatures are tried):
  Q1  Does clause grouping beat the same model with ordinary ColBERT scoring on MultiConIR,
      by more than chance? (paired bootstrap over documents, 95% intervals)
  Q2  Does it hold for other late-interaction models, not only LateOn?
  Q3  Does the soft AND (tau = 0.05) work once each clause is put on a common scale by
      dividing by its score against its own text (`nyaya_cal`)? If not, the AND is dropped.
      `grouped_cal` (calibrated clauses, averaged) separates the effect of calibration itself.

Models: lightonai/LateOn, lightonai/GTE-ModernColBERT-v1, lightonai/mLateOn.
All variants use 128-token queries without expansion, so nothing is truncated.
"""
import argparse
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kaggle_run import PY, Q, ROOT, log, run_all  # noqa: E402

MODELS = ["lightonai/LateOn", "lightonai/GTE-ModernColBERT-v1", "lightonai/mLateOn"]
VARIANTS = [("colbert", "ColBERT scoring (baseline)"), ("colbert_grouped", "+ clause grouping"),
            ("grouped_cal", "+ grouping, calibrated clauses"), ("nyaya_tok", "+ grouping + soft AND"),
            ("nyaya_cal", "+ grouping + soft AND, calibrated")]


def tag(model):
    return model.split("/")[-1].replace(".", "_")


def make_jobs(models, size):
    jobs = []
    for m in models:
        for v, _ in VARIANTS:
            name = f"r2_{tag(m)}_{v}"
            jobs.append(dict(name=name, est=12, cmd=(
                f"{Q(PY)} scripts/eval_suite.py --zero_shot {Q(m)} --variant {v} --query_length 128 --tau 0.05 "
                f"--suites multiconir --out results/{name}.json {size}")))
    return jobs


# --------------------------------------------------------------------------- statistics

def _row_values(r, what):
    p = r.get("multiconir_pairs") if r else None
    if not p:
        return None
    if what == "t1":
        return [sum(w) / 10 for w in p["task1"]]
    if what == "t1k10":
        return [w[9] for w in p["task1"]]
    if what == "t2mono":
        return [x[1] for x in p["task2"]]
    if what == "t3":
        return [x[0] for x in p["task3"]]
    raise ValueError(what)


def paired_ci(a, b, n_boot=5000, seed=0):
    """Mean difference a - b over documents, with a 95% paired-bootstrap interval."""
    import numpy as np
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    idx = np.random.default_rng(seed).integers(0, len(d), size=(n_boot, len(d)))
    means = d[idx].mean(1)
    return float(d.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def summary(models):
    L = ["# Round 2: clause grouping across models (training-free, MultiConIR)", ""]
    load = lambda m, v: (json.load(open(f"{ROOT}/results/r2_{tag(m)}_{v}.json"))  # noqa: E731
                         if os.path.exists(f"{ROOT}/results/r2_{tag(m)}_{v}.json") else None)
    f = lambda x: "–" if x is None else f"{x:.3f}"  # noqa: E731

    def g(r, *keys):
        for k in keys:
            if not isinstance(r, dict):
                return None
            r = r.get(k)
        return r

    for m in models:
        L += [f"## {m}", "",
              "| Scoring | T1 mean | T1 k=5 | T1 k=10 | T2 monotonicity | T3 pos>HN1 |",
              "|---|---|---|---|---|---|"]
        for v, lab in VARIANTS:
            r = load(m, v)
            L.append(f"| {lab} | {f(g(r, 'multiconir', 'task1', 'mean'))} | {f(g(r, 'multiconir', 'task1', 'wr5'))} | "
                     f"{f(g(r, 'multiconir', 'task1', 'wr10'))} | {f(g(r, 'multiconir', 'task2', 'monotonicity'))} | "
                     f"{f(g(r, 'multiconir', 'task3', 'pos_vs_hn1'))} |")
        L += ["", "Paired differences over all MultiConIR documents (mean, 95% bootstrap interval; "
              "an interval that excludes 0 means the difference is unlikely to be chance):", "",
              "| Comparison | T1 mean | T1 k=10 | T2 monotonicity | T3 pos>HN1 |", "|---|---|---|---|---|"]
        base = load(m, "colbert")
        for v, ref, lab in (("colbert_grouped", "colbert", "grouping − baseline"),
                            ("grouped_cal", "colbert_grouped", "calibration − grouping"),
                            ("nyaya_tok", "colbert_grouped", "soft AND − grouping"),
                            ("nyaya_cal", "colbert_grouped", "calibrated soft AND − grouping")):
            a, b = load(m, v), (base if ref == "colbert" else load(m, ref))
            cells = []
            for what in ("t1", "t1k10", "t2mono", "t3"):
                va, vb = _row_values(a, what), _row_values(b, what)
                if va is None or vb is None or len(va) != len(vb):
                    cells.append("–")
                    continue
                mu, lo, hi = paired_ci(va, vb)
                cells.append(f"{mu:+.3f} [{lo:+.3f}, {hi:+.3f}]")
            L.append(f"| {lab} | " + " | ".join(cells) + " |")
        L.append("")
    L += ["Per-document means are pooled over the five domains, so they can differ slightly from the "
          "macro (per-domain) averages in the first table.", ""]
    st = f"{ROOT}/results/jobs_status.json"
    if os.path.exists(st):
        s = json.load(open(st))
        bad = [k for k, v in s.items() if k.startswith("r2_") and v.get("state") != "done"]
        L.append(f"Jobs not finished: {', '.join(bad)} (see logs/)." if bad else "All jobs finished.")
    txt = "\n".join(L) + "\n"
    open(f"{ROOT}/results/ROUND2.md", "w").write(txt)
    return txt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=MODELS)
    ap.add_argument("--smoke", action="store_true", help="tiny sizes, only to check that everything runs")
    ap.add_argument("--summary_only", action="store_true")
    a = ap.parse_args()
    os.chdir(ROOT)
    for d in ("results", "logs"):
        os.makedirs(d, exist_ok=True)
    if a.summary_only:
        print(summary(a.models))
        return
    import torch
    n = torch.cuda.device_count()
    gpus = list(range(n)) or [0]
    log(f"GPUs: {n}; models: {', '.join(a.models)}")
    subprocess.run([PY, "scripts/prep_data.py", "--data", "data", "--only", "model", "multiconir",
                    "--model", *[m for m in a.models if "/" in m and not os.path.isdir(m)]], cwd=ROOT)
    size = "--n_docs 400 --q_per_width 2 --mc_rows 3" if a.smoke else ""
    jobs = make_jobs(a.models, size)
    log(f"{len(jobs)} jobs on {len(gpus)} GPU(s)")
    run_all(jobs, gpus, "results/jobs_status.json")
    print("\n" + summary(a.models), flush=True)


if __name__ == "__main__":
    main()
