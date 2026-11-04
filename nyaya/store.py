"""Token-vector store for large collections.

ColBERT keeps one vector per token, so a padded [documents x max_tokens x dim] tensor wastes most of
its memory on padding and has to fit on one device. This store keeps every document's valid token
vectors back to back in one flat fp16 tensor, with document boundaries in `ptr`:

    flat [T, d]      all token vectors        ptr [N + 1]     document n = flat[ptr[n]:ptr[n+1]]
    live [N]         False for deleted documents (tombstones, removed by `compact`)

Exact MaxSim search streams the flat tensor through the GPU in chunks, so the collection never has
to fit in GPU memory; if it does fit (under `gpu_budget_gb`) it stays resident and is not copied
again. Re-ranking takes a small padded view of the candidates (`padded`).
"""
from __future__ import annotations

import os

import torch


class VectorStore:
    def __init__(self, flat: torch.Tensor, ptr: torch.Tensor, live: torch.Tensor | None = None):
        if ptr.dim() != 1 or int(ptr[0]) != 0 or int(ptr[-1]) != flat.shape[0]:
            raise ValueError("ptr must start at 0 and end at the number of token vectors")
        self.flat, self.ptr = flat, ptr.long()
        self.live = live.bool() if live is not None else torch.ones(len(ptr) - 1, dtype=torch.bool)
        if len(self.live) != len(self):
            raise ValueError("live mask length differs from the number of documents")
        self._resident = None                     # (device, flat on that device)

    # ------------------------------------------------------------------ construction
    @classmethod
    def from_docs(cls, docs: list[torch.Tensor], dim: int | None = None) -> "VectorStore":
        """docs: one [L_i, d] tensor per document (valid tokens only)."""
        if not docs:
            return cls(torch.zeros(0, dim or 128, dtype=torch.float16), torch.zeros(1, dtype=torch.long))
        lens = torch.tensor([len(x) for x in docs], dtype=torch.long)
        ptr = torch.zeros(len(docs) + 1, dtype=torch.long)
        ptr[1:] = torch.cumsum(lens, 0)
        flat = torch.cat([x.detach().to("cpu", torch.float16) for x in docs])
        return cls(flat, ptr)

    @classmethod
    def from_padded(cls, D: torch.Tensor, M: torch.Tensor) -> "VectorStore":
        """Convert the older padded format ([N, K, d] + mask)."""
        return cls.from_docs([D[i][M[i]] for i in range(D.shape[0])], dim=D.shape[-1])

    def append(self, other: "VectorStore") -> range:
        """Add documents; returns their positions."""
        start = len(self)
        self.flat = torch.cat([self.flat, other.flat.to(self.flat.dtype)])
        self.ptr = torch.cat([self.ptr, other.ptr[1:] + self.ptr[-1]])
        self.live = torch.cat([self.live, other.live])
        self._resident = None
        return range(start, len(self))

    def delete(self, positions) -> int:
        idx = torch.as_tensor(list(positions), dtype=torch.long)
        n = int(self.live[idx].sum()) if len(idx) else 0
        self.live[idx] = False
        return n

    def compacted(self) -> tuple["VectorStore", list[int]]:
        """A new store with only live documents, and the old position of each kept document."""
        keep = torch.nonzero(self.live).flatten().tolist()
        return VectorStore.from_docs([self.doc(i) for i in keep], dim=self.dim), keep

    # ------------------------------------------------------------------ access
    def __len__(self):
        return len(self.ptr) - 1

    @property
    def dim(self) -> int:
        return self.flat.shape[1]

    @property
    def n_live(self) -> int:
        return int(self.live.sum())

    def doc(self, i: int) -> torch.Tensor:
        return self.flat[self.ptr[i]:self.ptr[i + 1]]

    def docs(self, ids=None):
        for i in (range(len(self)) if ids is None else ids):
            yield self.doc(i)

    def padded(self, ids: list[int], device="cpu", dtype=torch.float16):
        """[n, K, d] vectors and [n, K] mask for a few documents (re-ranking candidates)."""
        lens = [int(self.ptr[i + 1] - self.ptr[i]) for i in ids]
        K = max(lens) if lens else 1
        D = torch.zeros(len(ids), K, self.dim, dtype=dtype)
        M = torch.zeros(len(ids), K, dtype=torch.bool)
        for j, (i, n) in enumerate(zip(ids, lens)):
            D[j, :n] = self.doc(i)
            M[j, :n] = True
        return D.to(device), M.to(device)

    def nbytes(self) -> int:
        return self.flat.numel() * self.flat.element_size()

    # ------------------------------------------------------------------ search
    def _on(self, device, gpu_budget_gb):
        if torch.device(device).type == "cpu":
            return self.flat, False
        if self._resident is not None and self._resident[0] == str(device):
            return self._resident[1], True
        if self.nbytes() <= gpu_budget_gb * 2 ** 30:
            self._resident = (str(device), self.flat.to(device))
            return self._resident[1], True
        return self.flat, False

    @torch.no_grad()
    def search(self, Q: torch.Tensor, k: int = 100, allowed=None, device="cpu", chunk_tokens: int = 1 << 20,
               gpu_budget_gb: float = 6.0) -> list[int]:
        """Exact ColBERT scoring (mean over query tokens of the best-matching document token) over
        every live, allowed document; returns the top-k positions. Q: [L, d] valid query tokens."""
        ok = self.live.clone()
        if allowed is not None:
            ok &= torch.as_tensor(allowed, dtype=torch.bool)
        n = len(self)
        if n == 0 or not bool(ok.any()):
            return []
        flat, resident = self._on(device, gpu_budget_gb)
        cdt = torch.float16 if torch.device(device).type == "cuda" else torch.float32
        q = Q.to(device, cdt)
        scores = torch.full((n,), float("-inf"))
        a = 0
        while a < n:                                          # documents [a, b) hold at most chunk_tokens tokens
            limit = int(self.ptr[a]) + chunk_tokens
            b = int(torch.searchsorted(self.ptr, torch.tensor(limit), right=True)) - 1
            b = min(max(b, a + 1), n)
            t0, t1 = int(self.ptr[a]), int(self.ptr[b])
            if t1 > t0:
                tok = flat[t0:t1].to(device, cdt, non_blocking=True) if not resident else flat[t0:t1].to(cdt)
                sim = (q @ tok.T).float()                                         # [L, tokens]
                seg = torch.repeat_interleave(torch.arange(b - a, device=sim.device),
                                              (self.ptr[a + 1:b + 1] - self.ptr[a:b]).to(sim.device))
                best = torch.full((q.shape[0], b - a), float("-inf"), device=sim.device)
                best.scatter_reduce_(1, seg.expand(q.shape[0], -1), sim, reduce="amax", include_self=True)
                scores[a:b] = best.mean(0).cpu()
            a = b
        scores[~ok] = float("-inf")
        top = scores.topk(min(k, int(ok.sum())))
        return [i for i, v in zip(top.indices.tolist(), top.values.tolist()) if v != float("-inf")]

    # ------------------------------------------------------------------ files
    def save(self, path: str):
        tmp = path + ".tmp"
        torch.save({"format": 2, "flat": self.flat, "ptr": self.ptr, "live": self.live}, tmp)
        os.replace(tmp, path)                                   # never a half-written index

    @classmethod
    def load(cls, path: str) -> "VectorStore":
        v = torch.load(path, map_location="cpu", weights_only=True)
        if "flat" in v:
            return cls(v["flat"], v["ptr"], v.get("live"))
        return cls.from_padded(v["D"], v["M"])                  # older padded indexes
