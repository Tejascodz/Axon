"""Axon: condition-aware re-ranking with an evidence certificate for every result.

    from nyaya.axon import Axon
    ax = Axon("lightonai/LateOn")
    for r in ax.rerank("Find a jacket: 1. waterproof, 2. under 2 kg, 3. made in India", docs, top_k=5):
        print(r.score, r.text[:60])
        for c in r.conditions:
            print("  ", c.condition, round(c.relative, 2), c.evidence)

How it scores
  * The query is split into conditions (numbered lists, semicolons, new lines; or pass
    `conditions=[...]`, e.g. from an LLM planner). A query with one condition is scored exactly
    like the underlying ColBERT model, so simple queries never change.
  * "grouped": each condition's tokens are averaged, then conditions are averaged with equal weight.
What it returns
  For every result and condition: the condition's score, its score relative to the condition
  matched against its own text (1.0 = as good a match as the condition itself), and the
  document passages that matched it. `weakest` names the condition with the lowest relative score;
  from one document alone it is a hint (right 24-38% of the time on MultiConIR, chance 10%),
  much more reliable when compared against a document known to satisfy every condition.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch

from .model import NyayaRetriever, PyLateBackbone, split_clauses

METHODS = ("colbert", "grouped")


@dataclass
class ConditionMatch:
    condition: str
    score: float
    relative: float
    evidence: list[tuple[int, int, str]] = field(default_factory=list)   # (start, end, text) in the document


@dataclass
class Result:
    index: int
    text: str
    score: float
    conditions: list[ConditionMatch] = field(default_factory=list)
    weakest: int | None = None


@dataclass
class Index:
    texts: list[str]
    D: torch.Tensor
    M: torch.Tensor
    offsets: list[list[tuple[int, int]]] | None
    valid_only: bool = False            # D holds only unmasked tokens (built from a VectorStore)


class Axon:
    def __init__(self, model: str = "lightonai/LateOn", method: str = "grouped",
                 device: str | None = None, query_length: int = 128, batch_size: int = 32, fp16: bool = True):
        if method not in METHODS:
            raise ValueError(f"method must be one of {METHODS}")
        self.method, self.batch_size, self.model_name = method, batch_size, model
        self.device = device or ("cuda" if torch.cuda.is_available() else
                                 "mps" if torch.backends.mps.is_available() else "cpu")
        self.fp16 = fp16 and self.device == "cuda"
        self.backbone = PyLateBackbone(model, query_length=query_length)
        dim = self.backbone.hidden
        # one model in memory; each scoring method is a thin wrapper around it
        self._plain = NyayaRetriever(self.backbone, None, "token", "token", "chamfer", dim=dim).to(self.device).eval()
        self._by_method = {
            "grouped": NyayaRetriever(self.backbone, None, "clause_tokens", "token", "chamfer", dim=dim).to(self.device).eval(),
        }

    # ------------------------------------------------------------------ helpers
    def _ctx(self):
        return torch.autocast("cuda", dtype=torch.float16, enabled=self.fp16)

    def _doc_offsets(self, texts):
        """Character span of every document token, aligned with the model's own tokenization."""
        m = self.backbone.model
        tok = m.tokenizer
        prefix = getattr(m, "document_prefix_id", None)
        max_len = m.document_length - (1 if prefix is not None else 0)
        out = []
        for t in texts:
            lead = len(t) - len(t.lstrip())
            enc = tok(t.strip(), truncation=True, max_length=max_len, return_offsets_mapping=True)
            offs = [(a + lead, b + lead) if b > a else (0, 0) for a, b in enc["offset_mapping"]]
            if prefix is not None:
                offs.insert(1, (0, 0))
            out.append(offs)
        return out

    def _valid_offsets(self, texts):
        """Character spans of the unmasked document tokens only (what a VectorStore keeps), so that
        evidence can be shown for store-backed results without storing offsets for the whole index."""
        full = self._doc_offsets(texts)
        m = self.backbone.model
        feats = m.tokenize(list(texts), is_query=False)
        mask = (feats["attention_mask"].bool() & m.skiplist_mask(feats["input_ids"], m.skiplist)).tolist()
        out = []
        for offs, row, am in zip(full, mask, feats["attention_mask"].tolist()):
            n = int(sum(am))
            out.append([o for o, keep in zip(offs, row[:n]) if keep] if len(offs) == n else None)
        return out

    def encode_store(self, documents: list[str], log=None):
        """Encode a (possibly large) collection into a VectorStore, batch by batch on the device,
        keeping only each document's valid token vectors on the CPU."""
        from .store import VectorStore
        docs = []
        with torch.no_grad(), self._ctx():
            for i in range(0, len(documents), self.batch_size):
                D, M, _, _ = self._plain.encode(documents[i:i + self.batch_size], "d")
                docs += [D[j][M[j]].half().cpu() for j in range(D.shape[0])]
                if log and (i // self.batch_size) % 200 == 0:
                    log(f"encoded {min(i + self.batch_size, len(documents))}/{len(documents)}")
        return VectorStore.from_docs(docs, dim=self.backbone.hidden)

    def store_subset(self, store, ids: list[int], texts: list[str]) -> Index:
        """Candidates from a VectorStore as a small padded Index for re-ranking."""
        dt = torch.float16 if self.fp16 else torch.float32
        D, M = store.padded(ids, device=self.device, dtype=dt)
        return Index(list(texts), D, M, None, valid_only=True)

    @torch.no_grad()
    def search_store(self, query: str, store, k: int = 100, allowed=None, gpu_budget_gb: float = 6.0) -> list[int]:
        q = torch.as_tensor(self.query_vectors(query))
        return store.search(q, k=k, allowed=allowed, device=self.device, gpu_budget_gb=gpu_budget_gb)

    def _split(self, query, conditions):
        if conditions:
            return [c.strip() for c in conditions if c and c.strip()]
        parts = split_clauses(query)
        if len(parts) == 1 and parts[0] == query:              # prose: try sentences, else keep whole
            parts = split_clauses(query, mode="sentences")
        return parts

    # ------------------------------------------------------------------ public API
    @torch.no_grad()
    def index(self, documents: list[str]) -> Index:
        """Encode documents once; reuse the index for any number of queries."""
        vs, ms = [], []
        with self._ctx():
            for i in range(0, len(documents), self.batch_size):
                D, M, _, _ = self._plain.encode(documents[i:i + self.batch_size], "d")
                vs.append(D.half() if self.fp16 else D.float())
                ms.append(M)
        K = max(v.shape[1] for v in vs)
        D = vs[0].new_zeros(len(documents), K, vs[0].shape[-1])
        M = torch.zeros(len(documents), K, dtype=torch.bool, device=D.device)
        o = 0
        for v, m in zip(vs, ms):
            D[o:o + len(v), : v.shape[1]] = v
            M[o:o + len(v), : m.shape[1]] = m
            o += len(v)
        try:
            offsets = self._doc_offsets(documents)
        except Exception:                                        # explanations are optional; ranking is not
            offsets = None
        return Index(list(documents), D, M, offsets)

    def subset(self, index: Index, ids: list[int]) -> Index:
        """A view of `index` restricted to `ids` (e.g. first-stage candidates), in that order."""
        t = torch.as_tensor(ids, dtype=torch.long, device=index.D.device)
        offs = [index.offsets[i] for i in ids] if index.offsets else None
        return Index([index.texts[i] for i in ids], index.D[t], index.M[t], offs)

    @torch.no_grad()
    def query_vectors(self, query: str):
        """The query's token vectors as a [L, d] float32 array (for an approximate first stage)."""
        with self._ctx():
            Q, Qm, _, _ = self._plain.encode([query], "q")
        return Q[0][Qm[0]].float().cpu().numpy()

    @torch.no_grad()
    def search(self, query: str, index: Index, k: int = 100, chunk: int = 4096, allowed=None) -> list[int]:
        """Exact late-interaction search over a whole index (plain ColBERT scoring). `allowed` (bool
        array over the index) restricts the search to those documents: the others are never returned."""
        with self._ctx():
            q = self._plain.encode([query], "q")
            s = torch.cat([self._plain.score(q, (index.D[i:i + chunk], index.M[i:i + chunk], None, None))[0].float()
                           for i in range(0, index.D.shape[0], chunk)])
        if allowed is not None:
            s[~torch.as_tensor(allowed, device=s.device)] = float("-inf")
            k = min(k, int(allowed.sum()))
        if k <= 0:
            return []
        top = s.topk(min(k, s.shape[0]))
        return [i for i, v in zip(top.indices.tolist(), top.values.tolist()) if v != float("-inf")]

    @torch.no_grad()
    def rerank(self, query: str, documents, top_k: int | None = None, conditions: list[str] | None = None,
               explain: bool = True, method: str | None = None) -> list[Result]:
        method = method or self.method
        if method not in METHODS:
            raise ValueError(f"method must be one of {METHODS}")
        idx = documents if isinstance(documents, Index) else self.index(list(documents))
        conds = self._split(query, conditions)
        d = (idx.D, idx.M, None, None)
        with self._ctx():
            if method == "colbert" or len(conds) < 2:
                scores = self._plain.score(self._plain.encode([query], "q"), d)[0].float()
                clause, tok_idx, conds_used = None, None, [query]
            else:
                g = self._by_method[method]
                q = g.encode_conditions([conds])
                clause, S, tok_idx = g.clause_scores(q, d, return_index=True)
                clause = clause[0].float()                            # [N, C]
                scores = clause.mean(-1)
                conds_used = conds
                if explain:
                    own_idx = self.index(conds)
                    own = g.clause_scores(q, (own_idx.D, own_idx.M, None, None))[0].float()
                    ceil = own.diagonal().clamp(min=1e-3)
                qmask = q[1][0]                                       # [C, L]
        order = scores.argsort(descending=True).tolist()[: top_k or scores.shape[0]]
        offsets = idx.offsets
        if clause is not None and explain and offsets is None and idx.valid_only:
            try:                                                      # only for the results shown
                offsets = dict(zip(order, self._valid_offsets([idx.texts[n] for n in order])))
            except Exception:
                offsets = None
        results = []
        for n in order:
            r = Result(index=n, text=idx.texts[n], score=float(scores[n]))
            if clause is not None and explain:
                rel = (clause[n] / ceil).tolist()
                offs_n = offsets[n] if offsets is not None else None
                for c, cond in enumerate(conds_used):
                    ev = self._evidence(idx.texts[n], offs_n, tok_idx[0, n, c], S[0, n, c], qmask[c]) if offs_n else []
                    r.conditions.append(ConditionMatch(cond, float(clause[n, c]), float(rel[c]), ev))
                r.weakest = int(min(range(len(rel)), key=lambda c: rel[c]))
            results.append(r)
        return results

    def _evidence(self, text: str, offs, tokens, sims, mask, max_spans: int = 3):
        """Passages of a document that the condition's tokens matched, strongest first."""
        picked = {}
        for t, s, ok in zip(tokens.tolist(), sims.tolist(), mask.tolist()):
            if ok and t < len(offs) and offs[t][1] > offs[t][0]:
                picked[t] = max(picked.get(t, -1.0), s)
        if not picked:
            return []
        keep = sorted(picked, key=lambda t: -picked[t])[: max(3, len(picked) // 2)]
        spans = sorted(self._word_span(text, *offs[t]) + (picked[t],) for t in keep)
        merged = []
        for a, b, s in spans:                                     # merge tokens that sit next to each other
            if merged and a <= merged[-1][1] + 2:
                merged[-1] = (merged[-1][0], max(b, merged[-1][1]), merged[-1][2] + s)
            else:
                merged.append((a, b, s))
        merged.sort(key=lambda x: -x[2])
        return [(a, b, text[a:b]) for a, b, _ in merged[:max_spans]]

    @staticmethod
    def _word_span(text, a, b):
        while a > 0 and text[a - 1].isalnum():
            a -= 1
        while b < len(text) and text[b].isalnum():
            b += 1
        return a, b
