"""Token-vector store for large collections.

ColBERT keeps one vector per token, so a padded [documents x max_tokens x dim] tensor wastes most of
its memory on padding and has to fit on one device. This store keeps every document's valid token
vectors back to back in one flat fp16 array, with document boundaries in `ptr`:

    flat [T, d]      all token vectors        ptr [N + 1]     document n = flat[ptr[n]:ptr[n+1]]
    live [N]         False for deleted documents (tombstones, removed by `compact`)

On disk (format 3) the vectors are a raw little-endian fp16 file, memory-mapped when loaded: a
server starts without reading the whole file, and the operating system pages in what search
touches. Encoding writes the file batch by batch (`writer`), so building an index never holds two
copies of the vectors. Updates append to the file in place; a reader that opened the store
earlier only ever reads the prefix it knows about, so appending never disturbs it. `ptr` and `live`
are in a small metadata file written atomically. The index (nyaya.index) decides which metadata
file is current, so a crash mid-write never mixes two versions.

Exact MaxSim search streams the flat array through the GPU in chunks, so the collection never has
to fit in GPU memory; if it does fit (under `gpu_budget_gb`) it stays resident and is not copied
again. Re-ranking takes a small padded view of the candidates (`padded`).
"""
from __future__ import annotations

import os

import numpy as np
import torch

FORMAT = 3


def _mmap(path: str, n_tokens: int, dim: int) -> torch.Tensor:
    if n_tokens == 0:
        return torch.zeros(0, dim, dtype=torch.float16)
    # copy-on-write: the tensor is writable for torch, the file is never modified through it
    arr = np.memmap(path, dtype="<f2", mode="c", shape=(n_tokens, dim))
    return torch.from_numpy(arr)


class _Writer:
    """Appends documents' token vectors to a raw fp16 file without keeping them in memory."""

    def __init__(self, path: str, dim: int, mode: str = "wb", start_tokens: int = 0):
        self.path, self.dim = path, dim
        self.f = open(path, mode)
        if mode != "wb":                                   # append: drop any tail a crashed writer left
            self.f.truncate(start_tokens * dim * 2)
            self.f.seek(start_tokens * dim * 2)
        self.start, self.lens = start_tokens, []

    def add(self, x: torch.Tensor):
        if x.dim() != 2 or x.shape[1] != self.dim:
            raise ValueError(f"expected [tokens, {self.dim}], got {tuple(x.shape)}")
        self.f.write(x.detach().to("cpu", torch.float16).contiguous().numpy().astype("<f2", copy=False).tobytes())
        self.lens.append(int(x.shape[0]))

    def close(self):
        self.f.flush()
        os.fsync(self.f.fileno())
        self.f.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, *exc):
        if exc_type is None:
            self.close()
        else:                                              # a failed encode: release the file, keep nothing
            self.f.close()
        return False

    def ptr(self, base: torch.Tensor | None = None) -> torch.Tensor:
        own = torch.zeros(len(self.lens) + 1, dtype=torch.long)
        own[1:] = torch.cumsum(torch.tensor(self.lens, dtype=torch.long), 0) if self.lens else 0
        return own + self.start if base is None else torch.cat([base, own[1:] + self.start])


class VectorStore:
    def __init__(self, flat: torch.Tensor, ptr: torch.Tensor, live: torch.Tensor | None = None,
                 path: str | None = None):
        if ptr.dim() != 1 or int(ptr[0]) != 0 or int(ptr[-1]) != flat.shape[0]:
            raise ValueError("ptr must start at 0 and end at the number of token vectors")
        self.flat, self.ptr = flat, ptr.long()
        self.live = live.bool().clone() if live is not None else torch.ones(len(ptr) - 1, dtype=torch.bool)
        if len(self.live) != len(self):
            raise ValueError("live mask length differs from the number of documents")
        self.path = path                          # the data file behind `flat`, if it is memory-mapped
        self._resident = None                     # (device, flat on that device)

    # ------------------------------------------------------------------ construction
    @classmethod
    def from_docs(cls, docs, dim: int | None = None, path: str | None = None) -> "VectorStore":
        """docs: one [L_i, d] tensor per document (valid tokens only). With `path`, the vectors go
        straight to that file and the store is memory-mapped from it."""
        docs = iter(docs)
        if path is not None:
            first = next(docs, None)
            dim = dim or (first.shape[1] if first is not None else 128)
            with _Writer(path, dim) as w:
                for x in ([first] if first is not None else []):
                    w.add(x)
                for x in docs:
                    w.add(x)
            ptr = w.ptr()
            return cls(_mmap(path, int(ptr[-1]), dim), ptr, path=path)
        docs = list(docs)
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
        """Add documents in memory; returns their positions. (On disk, use `append_to`.)"""
        start = len(self)
        self.flat = torch.cat([self.flat, other.flat.to(self.flat.dtype)])
        self.ptr = torch.cat([self.ptr, other.ptr[1:] + self.ptr[-1]])
        self.live = torch.cat([self.live, other.live])
        self.path, self._resident = None, None
        return range(start, len(self))

    def append_to(self, other: "VectorStore") -> range:
        """Add documents by appending their vectors to this store's data file, which this store is
        memory-mapped from (anything past its last vector is a crashed writer's leftovers and is
        overwritten). Readers that mapped the file earlier keep seeing exactly what they saw."""
        if self.path is None:
            raise ValueError("this store is not on disk: write_data() it first")
        start = len(self)
        with _Writer(self.path, self.dim, mode="r+b", start_tokens=self.n_tokens) as w:
            for i in range(len(other)):
                w.add(other.doc(i))
        self.ptr = w.ptr(self.ptr)
        self.live = torch.cat([self.live, other.live])
        self.flat, self._resident = _mmap(self.path, self.n_tokens, self.dim), None
        return range(start, len(self))

    def delete(self, positions) -> int:
        idx = torch.as_tensor(list(positions), dtype=torch.long)
        n = int(self.live[idx].sum()) if len(idx) else 0
        self.live[idx] = False
        return n

    def compacted(self, path: str | None = None) -> tuple["VectorStore", list[int]]:
        """A new store with only live documents (written to `path` if given), and the old position
        of each kept document."""
        keep = torch.nonzero(self.live).flatten().tolist()
        return VectorStore.from_docs((self.doc(i) for i in keep), dim=self.dim, path=path), keep

    # ------------------------------------------------------------------ access
    def __len__(self):
        return len(self.ptr) - 1

    @property
    def dim(self) -> int:
        return self.flat.shape[1]

    @property
    def n_tokens(self) -> int:
        return int(self.ptr[-1])

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
    def write_data(self, data_path: str):
        """Write the vectors to a new data file and memory-map the store from it."""
        tmp = data_path + ".tmp"
        with _Writer(tmp, self.dim) as w:
            for i in range(len(self)):
                w.add(self.doc(i))
        os.replace(tmp, data_path)
        self.flat, self.path, self._resident = _mmap(data_path, self.n_tokens, self.dim), data_path, None

    def write_meta(self, meta_path: str):
        tmp = meta_path + ".tmp"
        with open(tmp, "wb") as f:
            torch.save({"format": FORMAT, "dim": self.dim, "ptr": self.ptr, "live": self.live}, f)
            f.flush()
            os.fsync(f.fileno())                   # durable before it is renamed into place
        os.replace(tmp, meta_path)

    @classmethod
    def open(cls, data_path: str, meta_path: str) -> "VectorStore":
        m = torch.load(meta_path, map_location="cpu", weights_only=True)
        n_tokens = int(m["ptr"][-1])
        need = n_tokens * m["dim"] * 2
        if os.path.getsize(data_path) < need:
            raise ValueError(f"{data_path} is shorter than its metadata says ({os.path.getsize(data_path)} < {need} bytes)")
        if len(m["ptr"]) < 1 or bool((m["ptr"][1:] < m["ptr"][:-1]).any()):
            raise ValueError(f"{meta_path}: document boundaries are not in order")
        return cls(_mmap(data_path, n_tokens, m["dim"]), m["ptr"], m["live"], path=data_path)

    # format 2 (one torch file) and the older padded format, kept for reading older indexes
    def save(self, path: str):
        tmp = path + ".tmp"
        torch.save({"format": 2, "flat": self.flat.clone(), "ptr": self.ptr, "live": self.live}, tmp)
        os.replace(tmp, path)

    @classmethod
    def load(cls, path: str) -> "VectorStore":
        v = torch.load(path, map_location="cpu", weights_only=True)
        if "flat" in v:
            return cls(v["flat"], v["ptr"], v.get("live"))
        return cls.from_padded(v["D"], v["M"])
