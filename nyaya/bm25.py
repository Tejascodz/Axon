"""Okapi BM25 over an in-memory inverted index (numpy only)."""
from __future__ import annotations

import collections
import math
import re

import numpy as np

_TOK = re.compile(r"[a-z0-9]+(?:[-_.][a-z0-9]+)*")


def tokenize(s: str) -> list[str]:
    out = []
    for t in _TOK.findall(s.lower()):
        out.append(t)
        if any(c in t for c in "-_."):
            out += [p for p in re.split(r"[-_.]", t) if p]
    return out


class BM25:
    def __init__(self, docs: list[str], k1: float = 1.2, b: float = 0.75):
        self.k1, self.b, self.N = k1, b, len(docs)
        post: dict[str, tuple[list, list]] = collections.defaultdict(lambda: ([], []))
        self.dl = np.zeros(self.N, dtype=np.float32)
        for i, d in enumerate(docs):
            tf = collections.Counter(tokenize(d))
            self.dl[i] = sum(tf.values())
            for t, c in tf.items():
                post[t][0].append(i)
                post[t][1].append(c)
        self.post = {t: (np.array(ids), np.array(tfs, dtype=np.float32)) for t, (ids, tfs) in post.items()}
        self.avgdl = float(self.dl.mean()) if self.N else 1.0
        self.idf = {t: math.log(1 + (self.N - len(ids) + 0.5) / (len(ids) + 0.5))
                    for t, (ids, _) in self.post.items()}

    def scores(self, q: str) -> np.ndarray:
        s = np.zeros(self.N, dtype=np.float32)
        for t in set(tokenize(q)):
            if t not in self.post:
                continue
            ids, tf = self.post[t]
            norm = self.k1 * (1 - self.b + self.b * self.dl[ids] / self.avgdl)
            s[ids] += self.idf[t] * tf * (self.k1 + 1) / (tf + norm)
        return s

    def search(self, q: str, k: int = 100) -> list[int]:
        s = self.scores(q)
        k = min(k, self.N)
        top = np.argpartition(-s, k - 1)[:k]
        return top[np.argsort(-s[top])].tolist()
