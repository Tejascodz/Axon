"""Controlled experiment: same encoder, same data, same steps; only the representation
and aggregation rule change.

Trains each variant from scratch on AND-of-OR product data and evaluates on a held-out
corpus (different random world) with queries of width 2..8. Widths 7 and 8 are never
seen in training, so they test length generalisation of the conjunction.

  python scripts/cpu_experiment.py --steps 2000 --no_paraphrase
"""
import argparse
import json
import os
import random
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nyaya.data_andor import AndOrWorld  # noqa: E402
from nyaya.evaluate import evaluate  # noqa: E402
from nyaya.train import AndOrStream, build_model, train  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+", default=["dense", "colbert", "clauses_sum", "nyaya", "nyaya_hybrid"])
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--n_docs", type=int, default=5000)
    ap.add_argument("--q_per_width", type=int, default=25)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--no_paraphrase", action="store_true", help="exact value words only (isolates the AND question)")
    ap.add_argument("--out", default="results/cpu_experiment.json")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    test_world = AndOrWorld(seed=4242, paraphrase=not args.no_paraphrase)
    widths = [2, 3, 4, 5, 6, 7, 8]
    docs, queries = test_world.make_split(args.n_docs, args.q_per_width * len(widths),
                                          widths, n_pos=5, n_hard=10, prefix="t")
    print(f"test corpus: {len(docs)} docs, {len(queries)} queries", flush=True)

    vocab_src = AndOrStream(AndOrWorld(seed=999, paraphrase=not args.no_paraphrase), hard=2)
    vocab_texts = [t for _ in range(300) for t in sum(vocab_src.batch(16)[:2], [])]

    results = {}
    if os.path.exists(args.out):
        results = json.load(open(args.out))
    for seed in args.seeds:
        for v in args.variants:
            key = f"{v}/seed{seed}"
            if key in results:
                continue
            torch.manual_seed(seed)
            random.seed(seed)
            ns = argparse.Namespace(variant=v, backbone="tiny", hidden=128, layers=2, dim=64,
                                    q_slots=8, d_slots=16, no_gates=False, tau=0.1, grad_ckpt=False)
            model = build_model(ns, vocab_texts)
            stream = AndOrStream(AndOrWorld(seed=1000 + seed, paraphrase=not args.no_paraphrase), widths=(2, 3, 4, 5, 6), hard=2, seed=seed)
            t0 = time.time()
            print(f"== {key}", flush=True)
            train(model, [stream], args.steps, args.batch, 1e-3, "cpu", log_every=250)
            res = evaluate(model, docs, queries)
            res["train_minutes"] = round((time.time() - t0) / 60, 1)
            results[key] = res
            print(key, json.dumps(res["all"]), flush=True)
            os.makedirs(os.path.dirname(args.out), exist_ok=True)
            json.dump(results, open(args.out, "w"), indent=2)


if __name__ == "__main__":
    main()
