"""Training for Nyaya and baselines.

Runs on CPU (tiny backbone, controlled experiments) or a single 16 GB GPU such as a
Colab / Kaggle T4 (Hugging Face backbone, fp16 autocast, gradient checkpointing).

Examples
  # controlled CPU experiment (from-scratch tiny encoder)
  python -m nyaya.train --backbone tiny --variant nyaya --steps 1500

  # T4: fine-tune a pretrained encoder, mixing AND-of-OR data with general triples
  python -m nyaya.train --backbone answerdotai/ModernBERT-base --variant nyaya \
      --steps 4000 --batch 32 --fp16 --grad_ckpt --mix_jsonl msmarco_triples.jsonl --mix_ratio 0.5
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time

import torch

from .data_andor import AndOrWorld
from .model import HFBackbone, NyayaRetriever, PyLateBackbone, TinyBackbone, WordTokenizer

VARIANTS = {
    # name           q_mode    d_mode    aggregator
    "dense":        ("single", "single", "chamfer"),
    "colbert":      ("token",  "token",  "chamfer"),
    "colbert_and":  ("token",  "token",  "softmin"),
    "slots_sum":    ("slots",  "slots",  "chamfer"),
    "clauses_sum":  ("clauses", "slots", "chamfer"),   # explicit clauses, ColBERT-style mean
    "nyaya":        ("clauses", "slots", "softmin"),   # explicit clauses over facet slots, fuzzy AND
    "nyaya_hybrid": ("clauses", "slots", "hybrid"),
    "nyaya_slots":  ("slots",  "slots",  "softmin"),   # learned query slots, no splitter
    # clause-grouped ColBERT: query tokens grouped by clause, document tokens kept (no facet slots)
    "colbert_grouped": ("clause_tokens", "token", "chamfer", "mean"),   # control: mean of clause means
    "nyaya_tok":       ("clause_tokens", "token", "softmin", "mean"),   # same vectors, fuzzy AND over clauses
    "nyaya_tok_or":    ("clause_tokens", "token", "softmin", "max"),    # + OR (max) over a clause's words
    "nyaya_tok_hybrid": ("clause_tokens", "token", "hybrid", "mean"),
    # clause scores divided by each clause's score against its own text, then the soft AND
    "nyaya_cal":       ("clause_tokens", "token", "softmin", "mean", True),
    "grouped_cal":     ("clause_tokens", "token", "chamfer", "mean", True),
}


class AndOrStream:
    """Endless batches: B queries, each with 1 planted positive and `hard` one-clause-off negatives."""

    def __init__(self, world: AndOrWorld, widths=(2, 3, 4, 5, 6), hard=2, seed=0):
        self.w, self.widths, self.hard = world, widths, hard
        self.rng = random.Random(seed)

    def batch(self, B):
        specs, docs = [], []
        for _ in range(B):
            spec = self.w.sample_query_spec(self.rng.choice(self.widths))
            specs.append(spec)
            vals = self.w.random_values()
            for c, a in spec.items():
                vals[c] = self.rng.choice(a)
            docs.append(vals)
        for spec in specs:
            for _ in range(self.hard):
                vals = self.w.random_values()
                for c, a in spec.items():
                    vals[c] = self.rng.choice(a)
                broken = self.rng.choice(list(spec))
                n = len(self.w_vocab(broken))
                vals[broken] = self.rng.choice([v for v in range(n) if v not in spec[broken]])
                docs.append(vals)
        q_text = [self.w.render_query("q", s) for s in specs]
        d_text = [self.w.render_doc("d", v).text for v in docs]
        rel = torch.tensor([[AndOrWorld.satisfies(v, s) for v in docs] for s in specs])
        return q_text, d_text, rel

    @staticmethod
    def w_vocab(cat):
        from .data_andor import VOCAB
        return VOCAB[cat]


class TripleStream:
    """General retrieval triples from jsonl lines: {"query": ..., "pos": ..., "negs": [...]}."""

    def __init__(self, path, hard=2, seed=0):
        self.rows = [json.loads(l) for l in open(path)]
        self.hard = hard
        self.rng = random.Random(seed)

    def batch(self, B):
        rows = self.rng.sample(self.rows, B)
        q = [r["query"] for r in rows]
        pos = [r["pos"] for r in rows]
        negs = [n for r in rows for n in (r.get("negs") or [])[: self.hard]]
        docs = pos + negs
        rel = torch.zeros(B, len(docs), dtype=torch.bool)
        rel[torch.arange(B), torch.arange(B)] = True
        return q, docs, rel


def build_model(args, vocab_texts=None):
    q_mode, d_mode, agg, *rest = VARIANTS[args.variant]
    if args.backbone == "tiny":
        tok = WordTokenizer(vocab_texts)
        bb = TinyBackbone(len(tok.itos), hidden=args.hidden, layers=args.layers)
        tokenize = tok
    elif args.backbone.startswith("pylate:"):                   # pretrained ColBERT, e.g. pylate:lightonai/LateOn
        bb = PyLateBackbone(args.backbone[len("pylate:"):], gradient_checkpointing=args.grad_ckpt,
                            query_length=getattr(args, "query_length", None))
        tokenize = None
    else:
        bb = HFBackbone(args.backbone, gradient_checkpointing=args.grad_ckpt)
        tokenize = bb.tokenize
    return NyayaRetriever(bb, tokenize, q_mode, d_mode, agg, dim=args.dim, q_slots=args.q_slots,
                          d_slots=args.d_slots, gated=not args.no_gates, tau=args.tau,
                          within=rest[0] if rest else "mean", calibrate=len(rest) > 1 and rest[1])


def train(model, streams, steps, batch, lr, device, fp16=False, slot_div=0.1, log_every=100,
          warmup=100, backbone_lr=None, mix_ratio=0.0, tau_schedule=(1.0, 0.05)):
    model.to(device).train()
    head = [p for n, p in model.named_parameters() if not n.startswith("backbone.")]
    body = [p for n, p in model.named_parameters() if n.startswith("backbone.")]
    opt = torch.optim.AdamW([{"params": head, "lr": lr},
                             {"params": body, "lr": backbone_lr or lr}], weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warmup) * max(0.0, 1 - s / steps))
    use_amp = fp16 and device.startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    t0, hist = time.time(), []
    anneal = int(0.6 * steps)                                   # mean-like early, min-like after 60% of steps
    for step in range(steps):
        if tau_schedule:
            t0_, t1_ = tau_schedule
            f = min(1.0, step / max(1, anneal))
            model.tau_override = t0_ * (t1_ / t0_) ** f
        src = streams[1] if len(streams) > 1 and random.random() < mix_ratio else streams[0]
        q, d, rel = src.batch(batch)
        with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
            loss = model.loss(q, d, rel.to(device), slot_div=slot_div)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        sched.step()
        hist.append(loss.item())
        if (step + 1) % log_every == 0:
            avg = sum(hist[-log_every:]) / log_every
            print(f"step {step + 1:5d}  loss {avg:.4f}  tau {float(model.tau):.3f}  "
                  f"{(time.time() - t0) / (step + 1):.3f}s/step", flush=True)
    if tau_schedule:                                            # make the final temperature part of the weights
        model.tau_override = None
        model.log_tau.data.fill_(math.log(tau_schedule[1]))
        model.config["tau"] = tau_schedule[1]
    return hist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="tiny")
    ap.add_argument("--variant", default="nyaya", choices=list(VARIANTS))
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--backbone_lr", type=float, default=None)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--q_slots", type=int, default=8)
    ap.add_argument("--d_slots", type=int, default=16)
    ap.add_argument("--tau", type=float, default=0.1)
    ap.add_argument("--no_gates", action="store_true")
    ap.add_argument("--slot_div", type=float, default=0.1)
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--grad_ckpt", action="store_true")
    ap.add_argument("--mix_jsonl", default=None)
    ap.add_argument("--mix_ratio", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--query_length", type=int, default=None, help="pylate backbones: max query tokens, no expansion")
    ap.add_argument("--no_paraphrase", action="store_true")
    ap.add_argument("--save", default=None)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    world = AndOrWorld(seed=1000 + args.seed, paraphrase=not args.no_paraphrase)
    vocab_texts = None
    if args.backbone == "tiny":
        s = AndOrStream(AndOrWorld(seed=999), hard=2)
        vocab_texts = [t for _ in range(200) for t in s.batch(16)[0] + s.batch(16)[1]]
    model = build_model(args, vocab_texts)
    streams = [AndOrStream(world, hard=2, seed=args.seed)]
    if args.mix_jsonl:
        streams.append(TripleStream(args.mix_jsonl))
    train(model, streams, args.steps, args.batch, args.lr, device, args.fp16, args.slot_div,
          backbone_lr=args.backbone_lr, mix_ratio=args.mix_ratio)
    if args.save:
        os.makedirs(os.path.dirname(args.save) or ".", exist_ok=True)
        ck = {"state": model.state_dict(), "config": model.config, "args": vars(args)}
        if args.backbone == "tiny":
            ck["itos"] = model.tokenize.itos
        torch.save(ck, args.save)


if __name__ == "__main__":
    main()
