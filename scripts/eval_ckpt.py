"""Evaluate a saved checkpoint.

  # held-out AND-of-OR set (widths 2..8, 7-8 unseen in training)
  python scripts/eval_ckpt.py --ckpt runs/nyaya.pt --andor

  # any BEIR-format folder, e.g. LIMIT (git clone https://github.com/google-deepmind/limit)
  python scripts/eval_ckpt.py --ckpt runs/nyaya.pt --beir limit/data/limit-small
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nyaya.data_andor import AndOrWorld  # noqa: E402
from nyaya.evaluate import evaluate, load_beir  # noqa: E402
from nyaya.rag import load_retriever  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--andor", action="store_true")
    ap.add_argument("--beir", default=None)
    ap.add_argument("--n_docs", type=int, default=20000)
    ap.add_argument("--q_per_width", type=int, default=50)
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    model = load_retriever(args.ckpt)
    report = {"ckpt": args.ckpt, "config": model.config}
    if args.andor:
        widths = [2, 3, 4, 5, 6, 7, 8]
        docs, qs = AndOrWorld(seed=4242).make_split(args.n_docs, args.q_per_width * len(widths), widths,
                                                     n_pos=5, n_hard=10, prefix="t")
        report["andor"] = evaluate(model, docs, qs, max_len=args.max_len)
    if args.beir:
        docs, qs = load_beir(args.beir)
        report[os.path.basename(args.beir.rstrip("/"))] = evaluate(model, docs, qs, ks=(2, 10, 20, 100),
                                                                     group_by_width=False, max_len=args.max_len)
    print(json.dumps(report, indent=2))
    if args.out:
        json.dump(report, open(args.out, "w"), indent=2)


if __name__ == "__main__":
    main()
