"""Late-interaction scoring rules.

All functions work on L2-normalised vectors.

Shapes
  Q : [B, C, d]  query vectors (tokens or clause slots)
  D : [N, K, d]  document vectors (tokens or facet slots)
  S : [B, N, C]  per-query-vector best match inside each document

Aggregators (over the C query vectors)
  chamfer : mean_c S          ColBERT / MaxSim (rank-equivalent to the sum)
  softmin : -tau * log mean_c exp(-S / tau)
            a smooth Goedel t-norm: -> min_c S as tau -> 0, -> mean_c S as tau -> inf
  hybrid  : a * softmin + (1 - a) * chamfer, with a in [0, 1]
"""
from __future__ import annotations

import torch

NEG = -1e4


def maxsim(Q: torch.Tensor, D: torch.Tensor, d_mask: torch.Tensor | None = None) -> torch.Tensor:
    """Best match per query vector: [B, C, d] x [N, K, d] -> [B, N, C]."""
    if Q.dtype != D.dtype:                       # e.g. fp16 corpus on GPU: compute in the corpus dtype
        Q = Q.to(D.dtype)
    sim = torch.einsum("bcd,nkd->bnck", Q, D)
    if d_mask is not None:
        sim = sim.masked_fill(~d_mask[None, :, None, :], NEG)
    return sim.max(dim=-1).values.float()


def chamfer(S: torch.Tensor, q_mask: torch.Tensor | None = None) -> torch.Tensor:
    if q_mask is None:
        return S.mean(dim=-1)
    m = q_mask[:, None, :].to(S.dtype)
    return (S * m).sum(-1) / m.sum(-1).clamp(min=1.0)


def softmin(S: torch.Tensor, q_mask: torch.Tensor | None = None, tau: float | torch.Tensor = 0.1) -> torch.Tensor:
    z = -S / tau
    if q_mask is None:
        n = S.new_full(S.shape[:-1], float(S.shape[-1]))
    else:
        z = z.masked_fill(~q_mask[:, None, :], float("-inf"))
        n = q_mask.sum(-1).to(S.dtype)[:, None].expand(S.shape[:-1])
    return -tau * (torch.logsumexp(z, dim=-1) - torch.log(n.clamp(min=1.0)))


def aggregate(S: torch.Tensor, kind: str, q_mask: torch.Tensor | None = None,
              tau: float | torch.Tensor = 0.1, alpha: torch.Tensor | None = None) -> torch.Tensor:
    if kind == "chamfer":
        return chamfer(S, q_mask)
    if kind == "softmin":
        return softmin(S, q_mask, tau)
    if kind == "hybrid":
        if alpha is None:
            raise ValueError("hybrid needs alpha")
        a = alpha.view(-1, 1) if alpha.dim() > 0 else alpha
        return a * softmin(S, q_mask, tau) + (1 - a) * chamfer(S, q_mask)
    raise ValueError(f"unknown aggregator {kind}")


def score(Q, D, kind="softmin", q_mask=None, d_mask=None, tau=0.1, alpha=None, gates=None):
    """Full score matrix [B, N].

    gates : optional [B, C] in [0, 1]. A gate of 0 turns that query vector into
            "always satisfied" (score 1), so filler slots cannot veto a document.
    """
    S = maxsim(Q, D, d_mask)
    if gates is not None:
        g = gates[:, None, :].to(S.dtype)
        S = g * S + (1 - g) * 1.0
    return aggregate(S, kind, q_mask, tau, alpha)


def score_grouped(Q, D, kind="softmin", q_mask=None, d_mask=None, tau=0.1, alpha=None, within="mean",
                  ceiling=None, compete=None):
    """Clause-grouped token late interaction.

    Q      : [B, C, L, d]  token vectors of each query clause (C clauses, up to L tokens each)
    q_mask : [B, C, L]     real tokens
    Every query token takes its best document token (ColBERT MaxSim). Tokens are averaged
    inside their clause (within="mean", ColBERT's own rule) or maxed (within="max",
    an OR over the clause's words), and clauses are combined with `kind` (an AND for softmin).
    ceiling : optional [B, C], each clause's score against its own text. Dividing by it puts
              clauses on a common scale (1 = as well matched as the clause matches itself)
              before they are compared or combined.
    compete : optional temperature; if set, document tokens are shared out among conditions
              (Competitive MaxSim, see competitive_maxsim) instead of being reusable by all.
    """
    clause = clause_scores(Q, D, q_mask, d_mask, within, ceiling, compete)
    if within == "flat":                                                      # already aggregated over tokens
        return clause
    if q_mask is None:
        q_mask = torch.ones(Q.shape[:3], dtype=torch.bool, device=Q.device)
    return aggregate(clause, kind, q_mask.any(-1), tau, alpha)


def competitive_maxsim(Q, D, q_mask, d_mask, tau):
    """Competitive MaxSim. Every document token is shared out among the query's conditions
    (softmax over conditions of how strongly the token matches each condition, temperature
    `tau`), and a query token can only use the share of a document token that belongs to its
    own condition. A document therefore cannot satisfy two conditions with the same evidence.
    With one condition every share is 1 and this is exactly MaxSim.

    Q [B, C, L, d], D [N, K, d] -> best match per query token [B, N, C, L] and the index of
    the document token it used [B, N, C, L]."""
    B, C, L, d = Q.shape
    if Q.dtype != D.dtype:
        Q = Q.to(D.dtype)
    sim = torch.einsum("bcd,nkd->bnck", Q.reshape(B, C * L, d), D).float().view(B, -1, C, L, D.shape[1])
    dm = None if d_mask is None else d_mask[None, :, None, None, :]
    qm = q_mask[:, None, :, :, None]
    r = sim.masked_fill(~qm, NEG).amax(3)                                    # [B, N, C, K] token-to-condition fit
    r = r.masked_fill(~q_mask.any(-1)[:, None, :, None], float("-inf"))
    share = torch.softmax(r / tau, dim=2)                                    # conditions compete for each token
    sim = sim * share[:, :, :, None, :]
    if dm is not None:
        sim = sim.masked_fill(~dm, NEG)
    best, idx = sim.max(-1)
    return best, idx


def clause_scores(Q, D, q_mask=None, d_mask=None, within="mean", ceiling=None, compete=None, return_index=False):
    """Per-condition scores [B, N, C] (or [B, N] for within='flat'), before conditions are combined."""
    B, C, L, d = Q.shape
    if q_mask is None:
        q_mask = torch.ones(B, C, L, dtype=torch.bool, device=Q.device)
    if compete:
        S, idx = competitive_maxsim(Q, D, q_mask, d_mask, compete)
    else:
        if Q.dtype != D.dtype:
            Q = Q.to(D.dtype)
        sim = torch.einsum("bcd,nkd->bnck", Q.reshape(B, C * L, d), D)
        if d_mask is not None:
            sim = sim.masked_fill(~d_mask[None, :, None, :], NEG)
        S, idx = sim.max(-1)
        S, idx = S.float().view(B, -1, C, L), idx.view(B, -1, C, L)
    m = q_mask[:, None].to(S.dtype)
    if within == "flat":                         # ablation: clauses encoded apart, every token weighted equally
        out = (S * m).sum((-1, -2)) / m.sum((-1, -2)).clamp(min=1.0)
        return (out, S, idx) if return_index else out
    if within == "mean":
        clause = (S * m).sum(-1) / m.sum(-1).clamp(min=1.0)
    elif within == "max":
        clause = S.masked_fill(m == 0, NEG).max(-1).values
    else:
        raise ValueError(f"unknown within-clause rule {within}")
    if ceiling is not None:
        clause = clause / ceiling[:, None, :].float().clamp(min=1e-3)
    return (clause, S, idx) if return_index else clause


def score_chunked(Q, D, chunk: int = 4096, **kw) -> torch.Tensor:
    """Same as score() but walks the corpus in chunks to bound memory."""
    d_mask = kw.pop("d_mask", None)
    outs = []
    for i in range(0, D.shape[0], chunk):
        dm = None if d_mask is None else d_mask[i:i + chunk]
        outs.append(score(Q, D[i:i + chunk], d_mask=dm, **kw))
    return torch.cat(outs, dim=1)
