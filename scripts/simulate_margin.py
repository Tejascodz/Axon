"""Controlled simulation: how should clause-level match scores be aggregated?

Setting (mirrors the AND-of-OR relevance used by the ANDOR benchmark):
  * a query has m clauses; a document is relevant only if every clause is satisfied
  * hard negatives satisfy all clauses but one
  * random negatives satisfy each clause independently with probability p
Each (clause, document) pair yields a noisy match score:
  satisfied   -> 1.0      + N(0, sigma^2)
  unsatisfied -> partial  + N(0, sigma^2)   (semantically close but wrong value)

We compare aggregators over the m clause scores:
  sum      : Chamfer / MaxSim style (sum over query vectors)
  min      : Goedel t-norm (hard AND)
  softmin  : -tau * log(mean(exp(-s / tau)))   (interpolates min <-> mean)

No learning happens here; this isolates the aggregation rule.
"""
import argparse
import json

import numpy as np


def aggregate(S, kind, tau=0.1):
    # S: [num_docs, m]
    if kind == "sum":
        return S.mean(axis=1)
    if kind == "min":
        return S.min(axis=1)
    if kind == "softmin":
        z = -S / tau
        zmax = z.max(axis=1, keepdims=True)
        lse = zmax[:, 0] + np.log(np.exp(z - zmax).mean(axis=1))
        return -tau * lse
    raise ValueError(kind)


def ndcg_at_k(order, rel, k=10):
    gains = rel[order[:k]]
    disc = 1.0 / np.log2(np.arange(2, k + 2))
    dcg = (gains * disc[: len(gains)]).sum()
    ideal = np.sort(rel)[::-1][:k]
    idcg = (ideal * disc[: len(ideal)]).sum()
    return dcg / idcg if idcg > 0 else 0.0


def run(m, sigma, partial, n_queries, n_pos, n_hard, n_rand, p_sat, rng, kinds, tau):
    out = {k: {"recall@pos": 0.0, "ndcg@10": 0.0} for k in kinds}
    for _ in range(n_queries):
        sat_pos = np.ones((n_pos, m), dtype=bool)
        sat_hard = np.ones((n_hard, m), dtype=bool)
        sat_hard[np.arange(n_hard), rng.integers(0, m, n_hard)] = False
        sat_rand = rng.random((n_rand, m)) < p_sat
        sat_rand[sat_rand.all(axis=1), 0] = False  # keep random negatives irrelevant
        sat = np.vstack([sat_pos, sat_hard, sat_rand])
        rel = np.zeros(len(sat))
        rel[:n_pos] = 1.0
        S = np.where(sat, 1.0, partial) + rng.normal(0.0, sigma, sat.shape)
        for k in kinds:
            order = np.argsort(-aggregate(S, k, tau))
            out[k]["recall@pos"] += rel[order[:n_pos]].sum() / n_pos
            out[k]["ndcg@10"] += ndcg_at_k(order, rel, 10)
    for k in kinds:
        for metric in out[k]:
            out[k][metric] /= n_queries
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", type=int, default=300)
    ap.add_argument("--pos", type=int, default=10)
    ap.add_argument("--hard", type=int, default=200)
    ap.add_argument("--rand", type=int, default=2000)
    ap.add_argument("--p_sat", type=float, default=0.5)
    ap.add_argument("--partial", type=float, default=0.4)
    ap.add_argument("--tau", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/simulation.json")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    kinds = ["sum", "softmin", "min"]
    rows = []
    for sigma in (0.05, 0.10, 0.20, 0.30):
        for m in (2, 3, 4, 6, 8):
            res = run(m, sigma, args.partial, args.queries, args.pos, args.hard,
                      args.rand, args.p_sat, rng, kinds, args.tau)
            row = {"sigma": sigma, "m": m}
            for k in kinds:
                row[f"{k}_recall"] = round(res[k]["recall@pos"], 4)
                row[f"{k}_ndcg10"] = round(res[k]["ndcg@10"], 4)
            rows.append(row)
            print(f"sigma={sigma:.2f} m={m}  "
                  + "  ".join(f"{k}: R={row[k + '_recall']:.3f}" for k in kinds))
    with open(args.out, "w") as f:
        json.dump({"args": vars(args), "rows": rows}, f, indent=2)


if __name__ == "__main__":
    main()
