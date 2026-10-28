"""Retriever with interchangeable query/document representations and scoring rules.

Representation modes per side
  single        : one mean-pooled vector                 (dense bi-encoder)
  token         : one vector per token                   (ColBERT)
  clause_tokens : query only; token vectors kept per condition (condition-grouped scoring)
  clauses       : query only; one vector per condition
  slots         : a fixed number of vectors from competitive slot attention
                  (query -> clause slots, document -> facet slots)

Competitive slot attention (one iteration of Slot Attention, Locatello et al. 2020):
attention is normalised over slots first, so tokens are shared out among slots,
then renormalised over tokens to form each slot's weighted average.
"""
from __future__ import annotations

import math
import re

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import scoring


# --------------------------------------------------------------------------- tokenizers / backbones

class WordTokenizer:
    """Lower-case word tokenizer with a vocabulary built from text (CPU experiments)."""
    PAD, UNK, CLS = 0, 1, 2
    _re = re.compile(r"[a-z0-9]+")

    def __init__(self, texts=None, max_vocab=20000):
        self.itos = ["[PAD]", "[UNK]", "[CLS]"]
        if texts is not None:
            counts: dict[str, int] = {}
            for t in texts:
                for w in self._re.findall(t.lower()):
                    counts[w] = counts.get(w, 0) + 1
            for w, _ in sorted(counts.items(), key=lambda x: -x[1])[: max_vocab - 3]:
                self.itos.append(w)
        self.stoi = {w: i for i, w in enumerate(self.itos)}

    def __call__(self, texts, max_len=96):
        ids = [[self.CLS] + [self.stoi.get(w, self.UNK) for w in self._re.findall(t.lower())][: max_len - 1]
               for t in texts]
        L = max(len(x) for x in ids)
        input_ids = torch.zeros(len(ids), L, dtype=torch.long)
        for i, x in enumerate(ids):
            input_ids[i, : len(x)] = torch.tensor(x)
        return {"input_ids": input_ids, "attention_mask": input_ids != self.PAD}


class TinyBackbone(nn.Module):
    """Small Transformer encoder trained from scratch; used for controlled experiments."""

    def __init__(self, vocab_size, hidden=128, layers=2, heads=4, max_len=256):
        super().__init__()
        self.hidden = hidden
        self.tok = nn.Embedding(vocab_size, hidden, padding_idx=0)
        self.pos = nn.Embedding(max_len, hidden)
        layer = nn.TransformerEncoderLayer(hidden, heads, 4 * hidden, dropout=0.1,
                                           batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, layers)
        self.norm = nn.LayerNorm(hidden)

    def forward(self, input_ids, attention_mask):
        pos = torch.arange(input_ids.shape[1], device=input_ids.device)[None]
        h = self.tok(input_ids) + self.pos(pos)
        h = self.enc(h, src_key_padding_mask=~attention_mask)
        return self.norm(h)


class HFBackbone(nn.Module):
    """Any Hugging Face encoder (e.g. ModernBERT, BGE, GTE) returning token states."""

    def __init__(self, name, gradient_checkpointing=False):
        super().__init__()
        from transformers import AutoModel, AutoTokenizer
        self.model = AutoModel.from_pretrained(name)
        self.tokenizer = AutoTokenizer.from_pretrained(name)
        self.hidden = self.model.config.hidden_size
        if gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

    def tokenize(self, texts, max_len=256):
        enc = self.tokenizer(texts, padding=True, truncation=True, max_length=max_len, return_tensors="pt")
        enc["attention_mask"] = enc["attention_mask"].bool()
        return enc

    def forward(self, input_ids, attention_mask, **kw):
        return self.model(input_ids=input_ids, attention_mask=attention_mask.long(), **kw).last_hidden_state


class PyLateBackbone(nn.Module):
    """A pretrained ColBERT checkpoint in PyLate format (e.g. lightonai/LateOn), used whole:
    encoder + its projection layers, its query/document markers, query expansion and
    punctuation skiplist. Outputs are already in the model's 128-d matching space, so the
    retriever's own projection starts as the identity: before any training, the `colbert`
    variant reproduces the original model."""
    projected = True

    def __init__(self, name, gradient_checkpointing=False, query_length=None):
        super().__init__()
        from pylate import models
        self.model = models.ColBERT(model_name_or_path=name)
        if query_length:                                  # longer queries, no [MASK] expansion: nothing truncated
            self.set_query_length(query_length)
        last = [m for m in self.model.children()][-1]
        self.hidden = getattr(last, "out_features", None) or self.model.get_sentence_embedding_dimension()
        if gradient_checkpointing:
            enc = self.model[0].auto_model
            if hasattr(enc, "gradient_checkpointing_enable"):
                enc.gradient_checkpointing_enable()

    def set_query_length(self, n):
        self.model.query_length = n
        self.model.do_query_expansion = False
        self.model.attend_to_expansion_tokens = False

    def encode_tokens(self, texts, is_query, device):
        feats = self.model.tokenize(texts, is_query=is_query)
        feats = {k: v.to(device) for k, v in feats.items()}
        h = self.model(feats)["token_embeddings"]
        if is_query and self.model.do_query_expansion:
            mask = torch.ones(h.shape[:2], dtype=torch.bool, device=device)   # expansion tokens are scored
        else:
            mask = feats["attention_mask"].bool()
            if not is_query:
                mask = mask & self.model.skiplist_mask(feats["input_ids"], self.model.skiplist)
        return h, mask


# --------------------------------------------------------------------------- pooling heads

class SlotPool(nn.Module):
    def __init__(self, hidden, n_slots, shared, gated=False):
        super().__init__()
        # unit-variance slot queries and unscaled logits: token-to-slot competition starts sharp,
        # which breaks the symmetry between slots (scaled-down init leaves every slot a mean pool)
        self.slots = nn.Parameter(torch.randn(n_slots, hidden))
        self.k = nn.Linear(hidden, hidden, bias=False)          # where to look: side-specific
        self.v, self.norm, self.proj = shared                    # what gets pooled: shared by both sides
        self.gate = nn.Linear(hidden, 1) if gated else None
        if self.gate is not None:                                 # slots start mostly "ignored": a slot must
            nn.init.constant_(self.gate.bias, -1.0)              # earn its right to veto a document

    def forward(self, h, mask):
        logits = torch.einsum("sh,bth->bst", self.slots, self.k(h))
        attn = logits.softmax(dim=1)                       # tokens are shared out among slots
        attn = attn * mask[:, None, :].to(attn.dtype)
        attn = attn / attn.sum(-1, keepdim=True).clamp(min=1e-6)
        pooled = self.norm(torch.einsum("bst,bth->bsh", attn, self.v(h)))
        vecs = F.normalize(self.proj(pooled), dim=-1)
        gates = torch.sigmoid(self.gate(pooled)).squeeze(-1) if self.gate is not None else None
        return vecs, attn, gates


class Side(nn.Module):
    """Turns token states into the vectors used for scoring, for one side (query or doc)."""

    def __init__(self, hidden, mode, shared, n_slots=8, gated=False):
        super().__init__()
        self.mode = mode
        if mode == "slots":
            self.pool = SlotPool(hidden, n_slots, shared, gated)
        else:
            self.proj = shared[2]

    def forward(self, h, mask):
        if self.mode == "single":
            m = mask[..., None].to(h.dtype)
            v = F.normalize(self.proj((h * m).sum(1) / m.sum(1).clamp(min=1)), dim=-1)
            return v[:, None, :], torch.ones(h.shape[0], 1, dtype=torch.bool, device=h.device), None, None
        if self.mode == "token":
            return F.normalize(self.proj(h), dim=-1), mask, None, None
        vecs, attn, gates = self.pool(h, mask)
        return vecs, torch.ones(vecs.shape[:2], dtype=torch.bool, device=h.device), gates, attn


# --------------------------------------------------------------------------- query clauses

_CLAUSE_SPLIT = re.compile(r";|\n|\s+and also\s+|\s+plus\s+", re.I)
_NUMBERED = re.compile(r"(?:^|(?<=[\s:,;(]))(\d{1,2})\s*[.)]\s*(?=\D)")
_SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")


def _numbered_items(text: str) -> list[str] | None:
    """'Find a book that: 1. X, 2. Y, 3. Z' -> [X, Y, Z] when the list is numbered 1, 2, 3, ..."""
    marks = [m for m in _NUMBERED.finditer(text)]
    seq, want = [], 1
    for m in marks:                                   # keep the first run that counts 1, 2, 3 ...
        if int(m.group(1)) == want:
            seq.append(m)
            want += 1
    if len(seq) < 2:
        return None
    items = []
    for a, b in zip(seq, seq[1:] + [None]):
        items.append(text[a.end(): b.start() if b is not None else len(text)].strip(" .,;:"))
    return [x for x in items if x]


def split_clauses(text: str, max_clauses: int = 12, mode: str = "auto") -> list[str]:
    """Rule-based clause splitter. An LLM query planner can replace it.

    auto      : a numbered list ("1. X, 2. Y") gives one clause per item (the lead-in such as
                "Find a book that meets these criteria:" is dropped); otherwise split on
                semicolons, newlines, "and also", "plus".
    sentences : additionally split prose into sentences (for descriptive multi-condition queries).
    """
    items = _numbered_items(text)
    if items is None:
        parts = _CLAUSE_SPLIT.split(text)
        if mode == "sentences":
            parts = [s for p in parts for s in _SENTENCE.split(p)]
        items = [p.strip(" .,") for p in parts]
    items = [p for p in items if len(p.split()) >= 2]
    return items[:max_clauses] or [text]


# --------------------------------------------------------------------------- full model

class NyayaRetriever(nn.Module):
    def __init__(self, backbone, tokenize, q_mode="slots", d_mode="slots", agg="softmin", dim=128,
                 q_slots=8, d_slots=16, gated=True, tau=0.1, learn_tau=False, scale=20.0, within="mean",
                 calibrate=False):
        super().__init__()
        self.backbone = backbone
        self.tokenize = tokenize
        self.agg = agg
        # one projection shared by both sides, so identical content maps to the same space from step one
        self.out_proj = nn.Linear(backbone.hidden, dim)
        if getattr(backbone, "projected", False) and dim == backbone.hidden:
            nn.init.eye_(self.out_proj.weight)
            nn.init.zeros_(self.out_proj.bias)
        self.slot_v = nn.Linear(backbone.hidden, backbone.hidden, bias=False)
        self.slot_norm = nn.LayerNorm(backbone.hidden)
        shared = (self.slot_v, self.slot_norm, self.out_proj)
        q_side_mode = {"clauses": "single", "clause_tokens": "token"}.get(q_mode, q_mode)
        self.q_side = Side(backbone.hidden, q_side_mode, shared, q_slots,
                           gated=gated and q_mode == "slots")
        self.d_side = Side(backbone.hidden, d_mode, shared, d_slots)
        self.log_tau = nn.Parameter(torch.tensor(math.log(tau)), requires_grad=learn_tau)
        self.tau_override: float | None = None                  # set by a temperature schedule
        self.clause_mode = "auto"                                 # how queries are split into clauses
        self.log_scale = nn.Parameter(torch.tensor(math.log(scale)))
        self.alpha_logit = nn.Parameter(torch.tensor(-1.0)) if agg == "hybrid" else None
        self.config = dict(q_mode=q_mode, d_mode=d_mode, agg=agg, dim=dim, q_slots=q_slots,
                           d_slots=d_slots, gated=gated, tau=tau, within=within, calibrate=calibrate)

    @property
    def tau(self):
        if self.tau_override is not None:
            return torch.tensor(self.tau_override, device=self.log_tau.device)
        return self.log_tau.exp().clamp(0.02, 1.0)

    def _device(self):
        return next(self.parameters()).device

    def _tokens(self, texts, side, max_len):
        """Token states [n, L, h] and a bool mask [n, L] for a list of texts."""
        if hasattr(self.backbone, "encode_tokens"):
            return self.backbone.encode_tokens(texts, side == "q", self._device())
        enc = self.tokenize(texts, max_len=max_len)
        enc = {k: v.to(self._device()) for k, v in enc.items()}
        return self.backbone(enc["input_ids"], enc["attention_mask"]), enc["attention_mask"].bool()

    def encode(self, texts, side, max_len=128):
        if side == "q" and self.config["q_mode"] == "clauses":
            return self._encode_clauses(texts, max_len)
        if side == "q" and self.config["q_mode"] == "clause_tokens":
            return self._encode_clause_tokens(texts, max_len)
        h, mask = self._tokens(texts, side, max_len)
        return (self.q_side if side == "q" else self.d_side)(h, mask)

    def _encode_clauses(self, texts, max_len):
        """Each clause becomes one mean-pooled vector: [B, C, d] with a clause mask."""
        groups = [split_clauses(t, mode=self.clause_mode) for t in texts]
        flat = [c for g in groups for c in g]
        h, am = self._tokens(flat, "q", max_len)
        m = am[..., None].to(h.dtype)
        pooled = self.slot_norm(self.slot_v((h * m).sum(1) / m.sum(1).clamp(min=1)))   # same path as doc facets
        v = F.normalize(self.out_proj(pooled), dim=-1)
        C = max(len(g) for g in groups)
        Q = v.new_zeros(len(groups), C, v.shape[-1])
        mask = torch.zeros(len(groups), C, dtype=torch.bool, device=v.device)
        o = 0
        for i, g in enumerate(groups):
            Q[i, : len(g)] = v[o:o + len(g)]
            mask[i, : len(g)] = True
            o += len(g)
        return Q, mask, None, None

    def _encode_clause_tokens(self, texts, max_len):
        """Token vectors kept per clause: [B, C, L, d] with a [B, C, L] mask (ColBERT tokens, grouped)."""
        groups = [split_clauses(t, mode=self.clause_mode) for t in texts]
        flat = [c for g in groups for c in g]
        h, am = self._tokens(flat, "q", max_len)
        v = F.normalize(self.out_proj(h), dim=-1)                                # [sum C, L, d]
        C, L = max(len(g) for g in groups), v.shape[1]
        Q = v.new_zeros(len(groups), C, L, v.shape[-1])
        mask = torch.zeros(len(groups), C, L, dtype=torch.bool, device=v.device)
        ceil = None
        if self.config.get("calibrate"):
            u = self._self_scores(flat, v, am, max_len)                          # [sum C]
            ceil = v.new_ones(len(groups), C)
        o = 0
        for i, g in enumerate(groups):
            Q[i, : len(g)] = v[o:o + len(g)]
            mask[i, : len(g)] = am[o:o + len(g)]
            if ceil is not None:
                ceil[i, : len(g)] = u[o:o + len(g)]
            o += len(g)
        return Q, mask, ceil, None                                               # slot 2 carries the ceiling

    def _self_scores(self, clauses, v, am, max_len):
        """Each clause scored against its own text encoded as a document: the best score it can expect."""
        hd, dm = self._tokens(clauses, "d", max_len)
        dv = F.normalize(self.out_proj(hd), dim=-1)
        sim = torch.einsum("cld,ckd->clk", v, dv).masked_fill(~dm[:, None, :], scoring.NEG).max(-1).values
        m = am.to(sim.dtype)
        return (sim * m).sum(-1) / m.sum(-1).clamp(min=1.0)

    def score(self, q, d):
        Q, q_mask, gates, _ = q
        D, d_mask, _, _ = d
        alpha = torch.sigmoid(self.alpha_logit) if self.alpha_logit is not None else None
        if Q.dim() == 4:                                                         # clause-grouped tokens
            return scoring.score_grouped(Q, D, kind=self.agg, q_mask=q_mask, d_mask=d_mask,
                                         tau=self.tau, alpha=alpha, within=self.config.get("within", "mean"),
                                         ceiling=gates)
        return scoring.score(Q, D, kind=self.agg, q_mask=q_mask, d_mask=d_mask,
                             tau=self.tau, alpha=alpha, gates=gates)

    def loss(self, queries, docs, labels, slot_div=0.0):
        """InfoNCE over all docs in the batch (positives + hard negatives of every query)."""
        q = self.encode(queries, "q")
        d = self.encode(docs, "d")
        logits = self.score(q, d) * self.log_scale.exp().clamp(max=100)
        if labels.dtype == torch.bool:                          # multi-positive: [B, N] relevance mask
            pos = logits.masked_fill(~labels, float("-inf"))
            loss = (torch.logsumexp(logits, -1) - torch.logsumexp(pos, -1)).mean()
        else:
            loss = F.cross_entropy(logits, labels)
        if slot_div > 0 and q[3] is not None:
            a = F.normalize(q[3], dim=-1)                       # [B, S, T]
            overlap = torch.einsum("bst,but->bsu", a, a)
            eye = torch.eye(overlap.shape[1], device=overlap.device)
            loss = loss + slot_div * ((overlap - eye) ** 2).mean()
        return loss
