"""Download every dataset once, before GPU jobs start (so parallel jobs never race on downloads).

  python scripts/prep_data.py --data data
Writes data/limit, data/MultiConIR, data/msmarco_triples.jsonl and warms the Hugging Face cache
for NanoBEIR. Each step is optional: a failure is reported in data/prep_status.json and the
jobs that need that dataset are skipped instead of crashing the whole run.
"""
import argparse
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def clone(url, dest):
    if os.path.isdir(dest):
        return "present"
    subprocess.run(["git", "clone", "-q", "--depth", "1", url, dest], check=True)
    return "cloned"


def msmarco(path, n):
    if os.path.exists(path):
        return "present"
    from datasets import load_dataset
    ds = load_dataset("sentence-transformers/msmarco-bm25", "triplet", split="train", streaming=True)
    k = 0
    with open(path + ".tmp", "w") as f:
        for r in ds:
            f.write(json.dumps({"query": r["query"], "pos": r["positive"], "negs": [r["negative"]]}) + "\n")
            k += 1
            if k >= n:
                break
    os.replace(path + ".tmp", path)
    return f"{k} triples"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--msmarco_n", type=int, default=100000)
    ap.add_argument("--model", nargs="*", default=[], help="pre-download these Hugging Face models once")
    ap.add_argument("--only", nargs="*", default=None, help="run only these steps (model limit multiconir msmarco nanobeir)")
    args = ap.parse_args()
    os.makedirs(args.data, exist_ok=True)
    status = {}
    steps = {
        "model": lambda: [__import__("huggingface_hub").snapshot_download(m) for m in args.model] or "none",
        "limit": lambda: clone("https://github.com/google-deepmind/limit", os.path.join(args.data, "limit")),
        "multiconir": lambda: clone("https://github.com/EIT-NLP/MultiConIR", os.path.join(args.data, "MultiConIR")),
        "msmarco": lambda: msmarco(os.path.join(args.data, "msmarco_triples.jsonl"), args.msmarco_n),
    }
    for name, fn in steps.items():
        if args.only is not None and name not in args.only:
            continue
        try:
            status[name] = {"ok": True, "info": fn()}
        except Exception as e:                      # keep going; dependent jobs are skipped
            status[name] = {"ok": False, "info": f"{type(e).__name__}: {e}"[:500]}
        print(f"prep {name}: {status[name]}", flush=True)
    try:
        if args.only is not None and "nanobeir" not in args.only:
            raise RuntimeError("skipped")
        from nyaya.benchmarks import NANO_DEFAULT, load_nanobeir
        for n in NANO_DEFAULT:
            d, q = load_nanobeir(n)
            print(f"prep nanobeir/{n}: {len(d)} docs, {len(q)} queries", flush=True)
        status["nanobeir"] = {"ok": True, "info": NANO_DEFAULT}
    except Exception as e:
        status["nanobeir"] = {"ok": False, "info": f"{type(e).__name__}: {e}"[:500]}
        print(f"prep nanobeir: {status['nanobeir']}", flush=True)
    json.dump(status, open(os.path.join(args.data, "prep_status.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
