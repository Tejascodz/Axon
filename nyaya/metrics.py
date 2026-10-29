"""Minimal Prometheus metrics (text exposition format 0.0.4), no dependencies.

    from nyaya.metrics import REGISTRY
    REQS = REGISTRY.counter("axon_requests_total", "Re-rank requests", ["method", "status"])
    LAT = REGISTRY.histogram("axon_request_seconds", "Request latency", ["method"])
    REQS.inc(method="grouped", status="200"); LAT.observe(0.031, method="grouped")
    REGISTRY.render()        # served at GET /metrics

Also `parse` reads a Prometheus text page into {(name, labels): value}, used to read vLLM's own
/metrics before and after a benchmark.
"""
from __future__ import annotations

import bisect
import math
import re
import threading

LATENCY_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60)


def _labels(names, values):
    if not names:
        return ""
    body = ",".join(f'{n}="{str(values[n]).replace(chr(92), chr(92) * 2).replace(chr(34), chr(92) + chr(34))}"'
                    for n in names)
    return "{" + body + "}"


class _Metric:
    def __init__(self, name, doc, labels):
        if not re.fullmatch(r"[a-zA-Z_:][a-zA-Z0-9_:]*", name):
            raise ValueError(name)
        self.name, self.doc, self.labelnames = name, doc, tuple(labels)
        self._lock = threading.Lock()

    def _key(self, kw):
        if set(kw) != set(self.labelnames):
            raise ValueError(f"{self.name} expects labels {self.labelnames}")
        return tuple(str(kw[n]) for n in self.labelnames)


class Counter(_Metric):
    kind = "counter"

    def __init__(self, *a):
        super().__init__(*a)
        self._v = {}

    def inc(self, amount=1.0, **labels):
        k = self._key(labels)
        with self._lock:
            self._v[k] = self._v.get(k, 0.0) + amount

    def lines(self):
        for k, v in sorted(self._v.items()):
            yield f"{self.name}{_labels(self.labelnames, dict(zip(self.labelnames, k)))} {v}"


class Histogram(_Metric):
    kind = "histogram"

    def __init__(self, name, doc, labels, buckets=LATENCY_BUCKETS):
        super().__init__(name, doc, labels)
        self.buckets = tuple(sorted(buckets))
        self._v = {}

    def observe(self, value, **labels):
        k = self._key(labels)
        with self._lock:
            counts, s = self._v.get(k, ([0] * (len(self.buckets) + 1), 0.0))
            counts[bisect.bisect_left(self.buckets, value)] += 1
            self._v[k] = (counts, s + value)

    def lines(self):
        for k, (counts, s) in sorted(self._v.items()):
            lab = dict(zip(self.labelnames, k))
            acc = 0
            for b, c in zip(self.buckets + (math.inf,), counts):
                acc += c
                le = "+Inf" if b == math.inf else repr(float(b))
                yield f"{self.name}_bucket{_labels(self.labelnames + ('le',), {**lab, 'le': le})} {acc}"
            yield f"{self.name}_sum{_labels(self.labelnames, lab)} {s}"
            yield f"{self.name}_count{_labels(self.labelnames, lab)} {acc}"


class Registry:
    def __init__(self):
        self._m = {}

    def _get(self, cls, name, *a, **kw):
        if name not in self._m:
            self._m[name] = cls(name, *a, **kw)
        return self._m[name]

    def counter(self, name, doc, labels=()):
        return self._get(Counter, name, doc, labels)

    def histogram(self, name, doc, labels=(), buckets=LATENCY_BUCKETS):
        return self._get(Histogram, name, doc, labels, buckets=buckets)

    def render(self) -> str:
        out = []
        for m in self._m.values():
            out += [f"# HELP {m.name} {m.doc}", f"# TYPE {m.name} {m.kind}", *m.lines()]
        return "\n".join(out) + "\n"


REGISTRY = Registry()

_LINE = re.compile(r'^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{[^}]*\})?\s+([-+0-9.eE]+|NaN|[+-]Inf)(\s+\d+)?$')


def parse(text: str) -> dict:
    """{(name, labels_string): value} for every sample on a Prometheus text page."""
    out = {}
    for line in text.splitlines():
        m = _LINE.match(line.strip())
        if m:
            out[(m.group(1), m.group(2) or "")] = float(m.group(3))
    return out
