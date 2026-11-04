"""Load test for any OpenAI-compatible chat server: time to first token, inter-token latency,
time per output token, end-to-end latency and throughput, at several concurrency levels.

    from nyaya.llm_bench import benchmark
    rows = benchmark("http://127.0.0.1:8000/v1", "Qwen/Qwen3-4B-Instruct-2507", prompts, levels=(1, 4, 16, 32))

Every request streams, so latencies are measured on the client exactly as a user would see them.
Outputs are fixed-length (`ignore_eos`, supported by vLLM and nyaya.hf_server) so that servers are
compared on the same amount of work. Every request in a run gets a prompt not used before in that run
(warm-up and earlier levels included), so a server's prefix cache cannot make later levels look
faster; only the system prompt, which every real request shares, can be cached. The server's own Prometheus metrics (vLLM's /metrics) are
read before and after each level, and GPU memory and utilisation are sampled with nvidia-smi.
"""
from __future__ import annotations

import shutil
import statistics
import subprocess
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

from .llm import timed_chat
from .metrics import parse


def _pct(xs, p):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    k = min(len(xs) - 1, max(0, int(round(p / 100 * (len(xs) - 1)))))
    return xs[k]


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return statistics.fmean(xs) if xs else None


def scrape(base: str) -> dict:
    root = base.rstrip("/")
    root = root[:-3] if root.endswith("/v1") else root
    if not root.startswith(("http://", "https://")):
        raise ValueError("server URL must be http(s)")
    try:
        with urllib.request.urlopen(f"{root}/metrics", timeout=10) as r:  # scheme checked above  # nosec B310
            return parse(r.read().decode())
    except Exception:
        return {}


def server_side(before: dict, after: dict) -> dict:
    """Mean of every server histogram over the interval (delta sum / delta count), plus counter deltas."""
    out = {}
    for (name, lab), v in after.items():
        if name.endswith("_sum"):
            base = name[:-4]
            cnt = after.get((base + "_count", lab), 0) - before.get((base + "_count", lab), 0)
            if cnt > 0:
                out[f"{base}_mean{lab}"] = (v - before.get((name, lab), 0)) / cnt
        elif name.endswith("_total"):
            d = v - before.get((name, lab), 0)
            if d:
                out[f"{name}{lab}"] = d
    return out


class GpuSampler:
    def __init__(self, gpu: int | None, interval_ms: int = 500):
        self.gpu, self.interval, self.samples, self._p = gpu, interval_ms, [], None

    def __enter__(self):
        if self.gpu is None or not shutil.which("nvidia-smi"):
            return self
        # clock and temperature too: a hot T4 lowers its clock, which changes latency between runs
        self._p = subprocess.Popen(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,clocks.sm,temperature.gpu,power.draw",
                                    "--format=csv,noheader,nounits",
                                    "-i", str(self.gpu), "-lms", str(self.interval)], stdout=subprocess.PIPE, text=True)
        self._t = threading.Thread(target=self._read, daemon=True)
        self._t.start()
        return self

    def _read(self):
        for line in self._p.stdout:
            try:
                vals = [float(x) for x in line.strip().split(",")]
                if len(vals) == 5:
                    self.samples.append(vals)
            except ValueError:
                pass

    def __exit__(self, *exc):
        if self._p:
            self._p.terminate()
            self._p.wait(timeout=5)

    def summary(self):
        if not self.samples:
            return {}
        col = list(zip(*self.samples))
        return {"gpu_util_mean_pct": round(_mean(col[0]), 1), "gpu_mem_peak_mib": max(col[1]),
                "gpu_sm_clock_mean_mhz": round(_mean(col[2])), "gpu_temp_max_c": max(col[3]),
                "gpu_power_mean_w": round(_mean(col[4]), 1)}


def run_level(base, model, work, concurrency, max_tokens=128, ignore_eos=True, gpu=None):
    n_requests = len(work)
    extra = {"ignore_eos": True} if ignore_eos else None

    def one(msgs):
        try:
            return timed_chat(msgs, model=model, base=base, max_tokens=max_tokens, temperature=0.0, extra=extra)
        except Exception as e:                                  # a failed request is counted, not fatal
            return {"error": repr(e)[:200]}
    before = scrape(base)
    with GpuSampler(gpu) as g:
        t0 = time.perf_counter()
        with ThreadPoolExecutor(concurrency) as ex:
            res = list(ex.map(one, work))
        wall = time.perf_counter() - t0
    after = scrape(base)
    ok = [r for r in res if "error" not in r]
    itl = [x for r in ok for x in r["itl"]]
    out_tok = sum(r["completion_tokens"] or 0 for r in ok)
    in_tok = sum(r["prompt_tokens"] or 0 for r in ok)
    ms = (lambda x: None if x is None else round(1000 * x, 1))  # noqa: E731
    row = {"concurrency": concurrency, "requests": n_requests, "failed": len(res) - len(ok), "wall_s": round(wall, 2),
           "ttft_ms_mean": ms(_mean([r["ttft"] for r in ok])), "ttft_ms_p50": ms(_pct([r["ttft"] for r in ok], 50)),
           "ttft_ms_p99": ms(_pct([r["ttft"] for r in ok], 99)),
           "itl_ms_mean": ms(_mean(itl)), "itl_ms_p50": ms(_pct(itl, 50)), "itl_ms_p99": ms(_pct(itl, 99)),
           "tpot_ms_mean": ms(_mean([r["tpot"] for r in ok])),
           "e2e_ms_p50": ms(_pct([r["e2e"] for r in ok], 50)), "e2e_ms_p99": ms(_pct([r["e2e"] for r in ok], 99)),
           "requests_per_s": round(len(ok) / wall, 3), "output_tokens_per_s": round(out_tok / wall, 1),
           "prompt_tokens_per_s": round(in_tok / wall, 1), "mean_prompt_tokens": round(in_tok / max(1, len(ok)), 1),
           "mean_output_tokens": round(out_tok / max(1, len(ok)), 1),
           "tokens_per_chunk": round(out_tok / max(1, sum(r["chunks"] for r in ok)), 2)}
    # fixed-length only if the server really produced max_tokens every time (some accept ignore_eos and ignore it)
    row["fixed_length"] = bool(ok) and ignore_eos and min((r["completion_tokens"] or 0) for r in ok) >= max_tokens
    row.update(g.summary())
    row["server"] = {k: round(v, 5) for k, v in server_side(before, after).items()}
    if len(ok) < len(res):
        row["first_error"] = next(r["error"] for r in res if "error" in r)
    return row


def benchmark(base, model, prompts, levels=(1, 4, 16, 32), requests_per_level=None, max_tokens=128,
              ignore_eos=True, gpu=None, warmup=2):
    """`prompts` is a pool; each warm-up request and each benchmark request takes the next unused one
    (the pool is only reused, with a note in the row, if it runs out)."""
    pos = 0

    def take(n):
        nonlocal pos
        out = [prompts[(pos + i) % len(prompts)] for i in range(n)]
        pos += n
        return out, pos > len(prompts)
    for p in take(warmup)[0]:                                    # compile / cache warm-up, not measured
        try:
            timed_chat(p, model=model, base=base, max_tokens=8, temperature=0.0)
        except Exception:
            pass
    if ignore_eos:                                              # not every server accepts it: then lengths vary
        try:
            timed_chat(take(1)[0][0], model=model, base=base, max_tokens=8, temperature=0.0, extra={"ignore_eos": True})
        except Exception:
            ignore_eos = False
    rows = []
    for c in levels:
        n = requests_per_level(c) if callable(requests_per_level) else (requests_per_level or max(16, 4 * c))
        work, reused = take(n)
        row = run_level(base, model, work, c, max_tokens, ignore_eos, gpu)
        row["prompts_reused"] = reused
        rows.append(row)
    return rows
