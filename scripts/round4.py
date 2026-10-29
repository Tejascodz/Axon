"""Round 4: Competitive MaxSim, violated-condition localization, the lead-in confound, latency.
Training-free. Kaggle T4 x2, about 2-2.5 h.

  python scripts/round4.py
  python scripts/round4.py --summary_only

Fixed before running (no other settings are tried):
  Q4  Does Competitive MaxSim (document tokens shared out among conditions, tau = 0.1) beat plain
      clause grouping on MultiConIR task 1 (paired 95% interval above zero) for at least 2 of
      the 3 models? tau = 0.03 and 0.3 are a sensitivity check on LateOn only.
  Q5  Localization: on near-miss documents that break exactly one condition, how often does each
      method name that condition (chance 10%), compared with a word-overlap baseline?
  Q6  Is the grouping gain more than removing the query's lead-in? `colbert_nolead` scores the
      same text as grouping, minus the lead-in, with ordinary ColBERT.
  Q7  (LateOn) separate encoding vs equal weight per condition: `separate_flat` encodes each
      condition apart but weights every token equally.
  Q8  Latency of re-ranking 100 candidates on a T4.
"""
import argparse
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kaggle_run import PY, Q, ROOT, log, run_all  # noqa: E402
from round2 import _row_values, paired_ci, tag  # noqa: E402

MODELS = ["lightonai/LateOn", "lightonai/GTE-ModernColBERT-v1", "lightonai/mLateOn"]
ALL = ["colbert", "colbert_nolead", "colbert_grouped", "competitive"]
LATEON_EXTRA = ["separate_flat", "competitive_t03", "competitive_t3"]
GROUPED = {"colbert_grouped", "competitive", "competitive_t03", "competitive_t3"}   # have per-condition scores
LATENCY = {"colbert", "colbert_grouped", "competitive"}
LABEL = {"colbert": "ColBERT", "colbert_nolead": "ColBERT, lead-in removed", "separate_flat": "conditions encoded apart, token-weighted",
         "colbert_grouped": "clause grouping", "competitive": "Competitive MaxSim (τ 0.1)",
         "competitive_t03": "Competitive MaxSim (τ 0.03)", "competitive_t3": "Competitive MaxSim (τ 0.3)"}


def variants(model):
    return ALL + (LATEON_EXTRA if model.endswith("/LateOn") else [])


def make_jobs(models, size):
    jobs = []
    for m in models:
        for v in variants(m):
            suites = ["multiconir"] + (["localization"] if v in GROUPED else []) + \
                     (["latency"] if v in LATENCY and m.endswith("/LateOn") else [])
            name = f"r4_{tag(m)}_{v}"
            jobs.append(dict(name=name, est=15, cmd=(
                f"{Q(PY)} scripts/eval_suite.py --zero_shot {Q(m)} --variant {v} --query_length 128 --tau 0.05 "
                f"--suites {' '.join(suites)} --out results/{name}.json {size}")))
    return jobs


def _load(m, v):
    p = f"{ROOT}/results/r4_{tag(m)}_{v}.json"
    return json.load(open(p)) if os.path.exists(p) else None


def _ci(a, b, what):
    va, vb = _row_values(a, what), _row_values(b, what)
    if va is None or vb is None or len(va) != len(vb):
        return "–"
    mu, lo, hi = paired_ci(va, vb)
    return f"{mu:+.3f} [{lo:+.3f}, {hi:+.3f}]"


def _acc_ci(hits):
    import numpy as np
    h = np.asarray(hits, dtype=float)
    idx = np.random.default_rng(0).integers(0, len(h), size=(5000, len(h)))
    m = h[idx].mean(1)
    return f"{h.mean():.3f} [{np.percentile(m, 2.5):.3f}, {np.percentile(m, 97.5):.3f}]"


def summary(models):
    L = ["# Round 4: Competitive MaxSim, localization, lead-in confound, latency (training-free)", ""]
    for m in models:
        L += [f"## {m}", "", "| Scoring | T1 mean | T1 k=10 | T2 monotonicity | T3 pos>HN1 |", "|---|---|---|---|---|"]
        for v in variants(m):
            r = _load(m, v)
            g = (r or {}).get("multiconir") or {}

            def f(task, key):
                x = g.get(task, {}).get(key)
                return f"{x:.3f}" if isinstance(x, (int, float)) else "–"
            L.append(f"| {LABEL[v]} | {f('task1', 'mean')} | {f('task1', 'wr10')} | {f('task2', 'monotonicity')} | "
                     f"{f('task3', 'pos_vs_hn1')} |")
        L += ["", "Paired differences over documents (95% bootstrap interval):", "",
              "| Comparison | T1 mean | T1 k=10 | T2 monotonicity | T3 pos>HN1 |", "|---|---|---|---|---|"]
        pairs = [("competitive", "colbert_grouped", "Q4  Competitive MaxSim − grouping"),
                 ("colbert_grouped", "colbert", "grouping − ColBERT"),
                 ("colbert_nolead", "colbert", "Q6  lead-in removed − ColBERT"),
                 ("colbert_grouped", "colbert_nolead", "Q6  grouping − lead-in removed")]
        if m.endswith("/LateOn"):
            pairs += [("separate_flat", "colbert_nolead", "Q7  separate encoding (token-weighted) − lead-in removed"),
                      ("colbert_grouped", "separate_flat", "Q7  equal weight per condition − token-weighted"),
                      ("competitive_t03", "colbert_grouped", "Q4  sensitivity τ 0.03 − grouping"),
                      ("competitive_t3", "colbert_grouped", "Q4  sensitivity τ 0.3 − grouping")]
        for a, b, lab in pairs:
            ra, rb = _load(m, a), _load(m, b)
            L.append(f"| {lab} | " + " | ".join(_ci(ra, rb, w) for w in ("t1", "t1k10", "t2mono", "t3")) + " |")
        L += ["", "Q5  Which condition does a near-miss break? Accuracy with 95% interval (chance 0.100):", "",
              "| Method | n | comparative (sees the positive) | absolute (near-miss only) |", "|---|---|---|---|"]
        lex_done = False
        for v in ("colbert_grouped", "competitive"):
            r = _load(m, v)
            h = (r or {}).get("localization_hits")
            if not h:
                continue
            n = r["localization"]["n"]
            if not lex_done:
                L.append(f"| word-overlap baseline (no model) | {n} | – | {_acc_ci(h['lexical'])} |")
                lex_done = True
            L.append(f"| {LABEL[v]} | {n} | {_acc_ci(h['comparative'])} | {_acc_ci(h['absolute'])} |")
        lat = [(v, (_load(m, v) or {}).get("latency")) for v in ("colbert", "colbert_grouped", "competitive")]
        if any(x for _, x in lat):
            L += ["", "Q8  Latency, re-ranking 100 candidates (documents pre-encoded):", "",
                  "| Scoring | device | docs/s to encode | ms per query (median) | ms per query (p95) |", "|---|---|---|---|---|"]
            for v, x in lat:
                if x:
                    L.append(f"| {LABEL[v]} | {x['device']} | {x['docs_per_s']} | {x['query_ms_median']} | {x['query_ms_p95']} |")
        L.append("")
    st = f"{ROOT}/results/jobs_status.json"
    if os.path.exists(st):
        s = json.load(open(st))
        bad = [k for k, v in s.items() if k.startswith("r4_") and v.get("state") != "done"]
        L.append(f"Jobs not finished: {', '.join(bad)} (see logs/)." if bad else "All jobs finished.")
    txt = "\n".join(L) + "\n"
    open(f"{ROOT}/results/ROUND4.md", "w").write(txt)
    return txt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=MODELS)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--summary_only", action="store_true")
    a = ap.parse_args()
    os.chdir(ROOT)
    for d in ("results", "logs"):
        os.makedirs(d, exist_ok=True)
    if a.summary_only:
        print(summary(a.models))
        return
    import torch
    gpus = list(range(torch.cuda.device_count())) or [0]
    log(f"GPUs: {torch.cuda.device_count()}; models: {', '.join(a.models)}")
    subprocess.run([PY, "scripts/prep_data.py", "--data", "data", "--only", "model", "multiconir",
                    "--model", *[m for m in a.models if "/" in m and not os.path.isdir(m)]], cwd=ROOT)
    jobs = make_jobs(a.models, "--n_docs 400 --q_per_width 2 --mc_rows 3" if a.smoke else "")
    log(f"{len(jobs)} jobs on {len(gpus)} GPU(s)")
    run_all(sorted(jobs, key=lambda j: -j["est"]), gpus, "results/jobs_status.json")
    print("\n" + summary(a.models), flush=True)


if __name__ == "__main__":
    main()
