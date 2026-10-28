"""One evaluation entry point for every model and benchmark.

  # a fine-tuned checkpoint
  python scripts/eval_suite.py --ckpt runs/x.pt --suites andor multiconir nanobeir limit --out results/x.json
  # a released ColBERT used as-is, with one of our scoring rules (no training)
  python scripts/eval_suite.py --zero_shot lightonai/LateOn --variant nyaya_tok --query_length 128 \
      --suites andor multiconir --out results/zs_nyaya_tok.json

Suites
  andor       synthetic AND-of-OR set (20k docs, widths 2..8; 7-8 never seen in training)
  multiconir  MultiConIR tasks 1-3 (needs data/MultiConIR)
  nanobeir    NanoBEIR subsets, ordinary search (downloads from Hugging Face)
  limit-small, limit   LIMIT (needs data/limit)
"""
import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nyaya.benchmarks import NANO_DEFAULT, multiconir, nanobeir  # noqa: E402
from nyaya.data_andor import AndOrWorld  # noqa: E402
from nyaya.evaluate import evaluate, load_beir  # noqa: E402
from nyaya.model import NyayaRetriever, PyLateBackbone  # noqa: E402
from nyaya.rag import load_retriever  # noqa: E402
from nyaya.train import VARIANTS  # noqa: E402


def build_zero_shot(name, variant, query_length, tau):
    q_mode, d_mode, agg, *rest = VARIANTS[variant]
    bb = PyLateBackbone(name, query_length=query_length)
    return NyayaRetriever(bb, None, q_mode, d_mode, agg, dim=bb.hidden, tau=tau,
                          within=rest[0] if rest else "mean", calibrate=len(rest) > 1 and rest[1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt")
    ap.add_argument("--zero_shot", help="PyLate model id, e.g. lightonai/LateOn")
    ap.add_argument("--variant", default="colbert")
    ap.add_argument("--query_length", type=int, default=None)
    ap.add_argument("--tau", type=float, default=0.05)
    ap.add_argument("--tau_sweep", type=float, nargs="*", default=[],
                    help="extra temperatures, reported separately as a sensitivity analysis")
    ap.add_argument("--suites", nargs="+", default=["andor", "multiconir", "nanobeir", "limit-small"])
    ap.add_argument("--data", default="data")
    ap.add_argument("--nano", nargs="+", default=NANO_DEFAULT)
    ap.add_argument("--n_docs", type=int, default=20000)
    ap.add_argument("--q_per_width", type=int, default=50)
    ap.add_argument("--mc_rows", type=int, default=None, help="MultiConIR rows per domain (default: all)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--fp32", action="store_true", help="disable fp16 inference on GPU")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if args.ckpt:
        model = load_retriever(args.ckpt, device=dev)
    else:
        model = build_zero_shot(args.zero_shot, args.variant, args.query_length, args.tau).to(dev).eval()
    report = json.load(open(args.out)) if os.path.exists(args.out) else {}
    report["config"] = {**model.config, "ckpt": args.ckpt, "zero_shot": args.zero_shot,
                        "variant": args.variant, "query_length": args.query_length,
                        "precision": "fp32" if (args.fp32 or dev != "cuda") else "fp16"}

    def save():
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        json.dump(report, open(args.out, "w"), indent=2)

    def run(name, fn):
        if name in report:
            print(f"[skip] {name} already done", flush=True)
            return
        t0 = time.time()
        # fp16 tensor cores on GPU (several times faster on a T4); scores are aggregated in fp32
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=(dev == "cuda" and not args.fp32)):
            report[name] = fn()
        report[name + "_minutes"] = round((time.time() - t0) / 60, 1)
        save()
        print(f"[done] {name} in {report[name + '_minutes']} min: {json.dumps(report[name])[:400]}", flush=True)

    for s in args.suites:
        if s == "andor":
            widths = [2, 3, 4, 5, 6, 7, 8]
            docs, qs = AndOrWorld(seed=4242).make_split(args.n_docs, args.q_per_width * len(widths), widths,
                                                         n_pos=5, n_hard=10, prefix="t")
            run("andor", lambda: evaluate(model, docs, qs, ks=(2, 10, 100), max_len=300))
        elif s == "multiconir":
            def mc():
                res, pairs = multiconir(model, os.path.join(args.data, "MultiConIR"), limit_rows=args.mc_rows,
                                        return_pairs=True)
                report["multiconir_pairs"] = pairs           # every comparison, for significance tests
                return res
            run("multiconir", mc)
        elif s == "nanobeir":
            run("nanobeir", lambda: nanobeir(model, args.nano))
        elif s in ("limit", "limit-small"):
            docs, qs = load_beir(os.path.join(args.data, "limit", "data", s))
            run(s, lambda: evaluate(model, docs, qs, ks=(2, 10, 20, 100), group_by_width=False, max_len=300))
        else:
            raise ValueError(f"unknown suite {s}")

    for t in args.tau_sweep:                      # sensitivity only; the headline uses --tau fixed in advance
        model.tau_override = t
        for s in args.suites:
            if s == "multiconir":
                run(f"multiconir@tau{t}", lambda: multiconir(model, os.path.join(args.data, "MultiConIR"), limit_rows=args.mc_rows))
        model.tau_override = None


if __name__ == "__main__":
    main()
