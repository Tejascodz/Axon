"""Round 5: the whole RAG system, served and measured. Kaggle T4 x2, about 4.5 h, unattended.

  python scripts/round5.py
  python scripts/round5.py --summary_only
  python scripts/round5.py --rebench      # with an earlier run's results/: serving benchmarks only

--rebench keeps every answer and training result, moves the earlier benchmarks to *_run1.json,
rebuilds the fine-tuned model from its saved adapter, and measures serving again: vLLM (base
and fine-tuned) and SGLang, back to back under the same conditions, with GPU clock and
temperature recorded. The transformers baseline is kept from the earlier run.

Every step writes its output file when it finishes, and a re-run skips steps whose output exists,
so a run can be continued from an earlier run's results/ folder (copy it in first).

GPU 1: index HotpotQA with Axon (LateOn) -> retrieval metrics and every test prompt -> LoRA
       fine-tuning of the generator (own implementation, fp16).
GPU 0: the generator (Qwen3-4B-Instruct-2507) served three ways: a one-request-at-a-time transformers
       server (nyaya.hf_server), vLLM and SGLang (each installed in its own environment). Serving
       benchmarks, then every test against the base model (vLLM), then against the fine-tuned model.

Fixed before running (no other settings are tried):
  Q9   Serving: time to first token, inter-token latency, time per output token and throughput at
       concurrency 1/4/16/32 (vLLM and SGLang, same GPU, model and prompts) and 1/4 (transformers
       baseline), fixed 128-token outputs.
  Q10  End-to-end answers on 500 HotpotQA questions retrieved from a ~7,000-paragraph corpus:
       exact match, F1, citation precision/recall; abstention on 200 questions whose gold
       paragraphs are withheld (should answer INSUFFICIENT) and false abstention on the 500.
  Q11  Prompt injection, 200 questions, 8 attack templates (incl. system-prompt exfiltration and
       instructions hidden with zero-width characters or look-alike letters), a unique canary each.
       Attack success rate: undefended prompt / defended prompt / defended + input scanner.
       Scanner detection rate and false-positive rate on all clean paragraphs.
  Q12  Condition check (MultiConIR, 600 documents, half satisfy all 10 conditions, half break
       exactly one): does the LLM name the right verdict (ALL, or the broken condition) more often
       when given Axon's per-condition evidence certificate? Success: paired 95% interval above 0.
       Axon alone (threshold fixed on 200 separate calibration documents) for reference.
  Q13  Fine-tuning (LoRA r=16, 1,600 HotpotQA training questions, 20% with gold removed, 10% with
       injections worded differently from the test set): success if citation precision/recall and
       abstention improve while exact match does not drop by more than 2 points.
"""
import argparse
import json
import os
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kaggle_run import PY, ROOT, log  # noqa: E402

LLM = "Qwen/Qwen3-4B-Instruct-2507"
RETRIEVER = "lightonai/LateOn"
FT_DIR = "/tmp/r5_ft"  # single-user Kaggle VM scratch  # nosec B108
STATUS = "results/r5_status.json"
VLLM_PIN = "0.33.0"                 # the release that ran on Kaggle T4s in round 5; others are fallbacks
VLLM_ATTEMPTS = [(VLLM_PIN, []), (None, []), (VLLM_PIN, ["--enforce-eager"]), ("0.11.0", ["--enforce-eager"])]
READY_TIMEOUT = 1200                # seconds a server may take to load the model and answer once
_lock = threading.Lock()


# --------------------------------------------------------------------------- bookkeeping

def status(step=None, **info):
    with _lock:
        s = json.load(open(STATUS)) if os.path.exists(STATUS) else {}
        if step:
            s[step] = {**s.get(step, {}), **info, "time": time.strftime("%H:%M:%S")}
            json.dump(s, open(STATUS, "w"), indent=1)
        return s


def sh(cmd, logname, env=None, timeout=None):
    """Run a command (list of args, no shell), output to logs/<logname>.log; returns the exit code."""
    with open(f"logs/{logname}.log", "a") as f:
        f.write("$ " + " ".join(shlex.quote(c) for c in cmd) + "\n")
        f.flush()
        try:
            return subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env={**os.environ, **(env or {})},
                                  cwd=ROOT, timeout=timeout).returncode
        except subprocess.TimeoutExpired:
            f.write("timeout\n")
            return -9


def gpu_env(gpu):
    return {"CUDA_VISIBLE_DEVICES": str(gpu)}


# --------------------------------------------------------------------------- servers

def wait_gpu_free(gpu, timeout=180, below_mib=1500):
    """Wait until a stopped server's memory is really released, so the next server on this GPU
    does not start out of memory."""
    if gpu is None or not shutil.which("nvidia-smi"):
        return True
    t0 = time.time()
    while time.time() - t0 < timeout:
        r = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits", "-i", str(gpu)],
                           capture_output=True, text=True)
        try:
            if float(r.stdout.strip().split()[0]) < below_mib:
                return True
        except (ValueError, IndexError):
            return True
        time.sleep(5)
    log(f"GPU {gpu} still holds memory after {timeout} s")
    return False


class Server:
    def __init__(self, cmd, env, logname, base):
        self.base, self.logname = base, logname
        self.gpu = env.get("CUDA_VISIBLE_DEVICES")
        self.f = open(f"logs/{logname}.log", "a")
        self.f.write("$ " + " ".join(shlex.quote(c) for c in cmd) + "\n")
        self.f.flush()
        self.p = subprocess.Popen(cmd, stdout=self.f, stderr=subprocess.STDOUT, env={**os.environ, **env},
                                  cwd=ROOT, start_new_session=True)

    def ready(self, model, timeout):
        from nyaya.llm import chat
        t0 = time.time()
        while time.time() - t0 < timeout:
            if self.p.poll() is not None:
                return False
            try:
                urllib.request.urlopen(self.base.replace("/v1", "") + "/health", timeout=5)  # our own http://127.0.0.1 server  # nosec B310
                chat([{"role": "user", "content": "Say OK."}], max_tokens=4, model=model, base=self.base, timeout=120)
                return True
            except Exception:
                time.sleep(10)
        return False

    def stop(self):
        # the whole process group, even if the main process already exited: serving engines run
        # worker processes that can outlive it and keep holding GPU memory
        try:
            os.killpg(self.p.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            self.p.wait(timeout=60)
        except subprocess.TimeoutExpired:
            pass
        time.sleep(3)
        try:
            os.killpg(self.p.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        self.f.close()
        wait_gpu_free(self.gpu)


def free_gb(path="/tmp"):  # single-user Kaggle VM scratch  # nosec B108
    return shutil.disk_usage(path).free / 2 ** 30


def make_env(env_dir, specs, module, logname, python=None):
    """A separate Python environment for a serving engine, so its pinned torch never touches ours.
    `python` picks another interpreter version (uv downloads it if the machine does not have it)."""
    py = f"{env_dir}/bin/python"
    if os.path.exists(py) and subprocess.run([py, "-c", f"import {module}"], capture_output=True).returncode == 0:
        return py
    if sh([PY, "-m", "pip", "install", "-q", "uv"], logname) != 0:
        return None
    made = python and (sh([PY, "-m", "uv", "venv", "--python", python, "--managed-python", env_dir], logname) == 0
                       or sh([PY, "-m", "uv", "venv", "--python", python, env_dir], logname) == 0)
    if not made and sh([PY, "-m", "uv", "venv", "--python", PY, env_dir], logname) != 0:
        return None
    for spec in specs:
        # --torch-backend=auto picks the PyTorch CUDA build that matches this machine's driver
        if sh([PY, "-m", "uv", "pip", "install", "--python", py, spec, "--torch-backend=auto"], logname, timeout=3600) == 0 \
                or sh([PY, "-m", "uv", "pip", "install", "--python", py, spec], logname, timeout=3600) == 0:
            break
    else:
        return None
    r = subprocess.run([py, "-c", f"import {module}, torch; print({module}.__version__, torch.__version__, torch.version.cuda)"],
                       capture_output=True, text=True)
    status(f"{logname}_{os.path.basename(env_dir)}", ok=r.returncode == 0, info=(r.stdout or r.stderr)[-300:])
    if r.returncode == 0:
        freeze_env(py, os.path.basename(env_dir))
    return py if r.returncode == 0 else None


def vllm_python(version):
    return make_env(f"/tmp/vllm_env_{version or 'latest'}", [f"vllm=={version}" if version else "vllm"], "vllm",  # single-user Kaggle VM scratch  # nosec B108
                    "vllm_install")


def sglang_python():
    if free_gb() < 25:
        status("sglang_install", ok=False, info=f"skipped: only {free_gb():.0f} GB disk free")
        return None
    # Python 3.12: SGLang depends on outlines-core, which has no ready-made build for 3.13 and would
    # otherwise need a Rust compiler
    return make_env("/tmp/sglang_env", ["sglang", "sglang[all]"], "sglang", "sglang_install", python="3.12")  # single-user Kaggle VM scratch  # nosec B108


def start_vllm(model_path, served, gpu, port=8000):
    base = f"http://127.0.0.1:{port}/v1"
    for version, flags in VLLM_ATTEMPTS:
        py = vllm_python(version)
        if py is None:
            continue
        cmd = [py, "-m", "vllm.entrypoints.openai.api_server", "--model", model_path, "--served-model-name", served,
               "--host", "127.0.0.1", "--port", str(port), "--dtype", "half", "--max-model-len", "4096",
               "--gpu-memory-utilization", "0.90", "--max-num-seqs", "32", *flags]
        log(f"starting vLLM {version or 'latest'} {' '.join(flags)} for {served}")
        s = Server(cmd, gpu_env(gpu), f"vllm_{served.replace('/', '_')}", base)
        if s.ready(served, timeout=READY_TIMEOUT):
            r = subprocess.run([py, "-c", "import vllm; print(vllm.__version__)"], capture_output=True, text=True)
            status(f"vllm_{served}", ok=True, version=r.stdout.strip() or version or "latest", flags=flags)
            return s
        s.stop()
        status(f"vllm_{served}", ok=False, tried=f"{version or 'latest'} {flags}")
    return None


SGLANG_ATTEMPTS = [[], ["--attention-backend", "triton"], ["--attention-backend", "triton", "--disable-cuda-graph"]]


def cuda_link_env():
    """On a T4, SGLang compiles some GPU kernels on first start, and that build links against
    libcuda.so. Many machines (Kaggle included) ship only the driver's libcuda.so.1, so the link
    fails with "cannot find -lcuda". A private folder with a libcuda.so link fixes the build
    without touching system files."""
    import glob
    d = "/tmp/cuda_link"  # single-user Kaggle VM scratch  # nosec B108
    link = f"{d}/libcuda.so"
    if not os.path.exists(link):
        cands = []
        try:
            out = subprocess.run(["ldconfig", "-p"], capture_output=True, text=True, timeout=30).stdout
            cands += [ln.split("=>")[-1].strip() for ln in out.splitlines() if "libcuda.so" in ln]
        except Exception:
            pass
        for pat in ("/usr/lib/x86_64-linux-gnu/libcuda.so*", "/usr/local/nvidia/lib64/libcuda.so*", "/usr/lib64/libcuda.so*",
                    "/usr/local/cuda/lib64/stubs/libcuda.so", "/usr/local/cuda/compat/libcuda.so*"):
            cands += sorted(glob.glob(pat))
        found = next((c for c in cands if os.path.exists(c)), None)
        if not found:
            status("cuda_link", ok=False, info="no libcuda.so found")
            return {}
        os.makedirs(d, exist_ok=True)
        os.symlink(os.path.realpath(found), link)
        status("cuda_link", ok=True, info=os.path.realpath(found))
    return {"LIBRARY_PATH": d + (":" + os.environ["LIBRARY_PATH"] if os.environ.get("LIBRARY_PATH") else "")}


def start_sglang(py, model_path, served, gpu, port=30000):
    base = f"http://127.0.0.1:{port}/v1"
    for flags in SGLANG_ATTEMPTS:
        cmd = [py, "-m", "sglang.launch_server", "--model-path", model_path, "--served-model-name", served,
               "--host", "127.0.0.1", "--port", str(port), "--dtype", "half", "--context-length", "4096",
               "--mem-fraction-static", "0.85", "--max-running-requests", "32", "--enable-metrics", *flags]
        log(f"starting SGLang {' '.join(flags)} for {served}")
        s = Server(cmd, {**gpu_env(gpu), **cuda_link_env()}, f"sglang_{served}", base)
        if s.ready(served, timeout=READY_TIMEOUT):
            r = subprocess.run([py, "-c", "import sglang; print(sglang.__version__)"], capture_output=True, text=True)
            status(f"sglang_{served}", ok=True, version=r.stdout.strip(), flags=flags)
            return s
        s.stop()
        status(f"sglang_{served}", ok=False, tried=" ".join(flags) or "default")
    return None


def start_hf(model_path, served, gpu, port=8100):
    s = Server([PY, "-m", "nyaya.hf_server", "--model", model_path, "--port", str(port)], gpu_env(gpu),
               f"hf_{served.replace('/', '_')}", f"http://127.0.0.1:{port}/v1")
    if s.ready(served, timeout=READY_TIMEOUT):
        return s
    s.stop()
    return None


# --------------------------------------------------------------------------- lanes

def step(name, cmd, logname, gpu=None, timeout=None, done_file=None):
    if done_file and os.path.exists(done_file):
        log(f"{name}: already done")
        return True
    log(f"{name}: start")
    t0 = time.time()
    rc = sh(cmd, logname, env=gpu_env(gpu) if gpu is not None else None, timeout=timeout)
    ok = rc == 0 and (done_file is None or os.path.exists(done_file))
    status(name, ok=ok, rc=rc, minutes=round((time.time() - t0) / 60, 1))
    log(f"{name}: {'done' if ok else 'FAILED (see logs/' + logname + '.log)'}")
    return ok


def lane_retrieval_and_training(a, gpu, ev_retrieval, ev_train):
    try:
        step("retrieve", [PY, "scripts/r5_steps.py", "retrieve", "--retriever", a.retriever, *a.size_retrieve],
             "r5_retrieve", gpu, timeout=3 * 3600, done_file="results/r5_prompts.json")
    finally:
        ev_retrieval.set()
    ft_done = os.path.exists("results/r5_gen_ft.json")         # the fine-tuned model's answers are what matter
    try:
        if a.skip_train or ft_done:
            if ft_done:
                log("fine-tuned model results already present; not training again")
                adapter = "results/r5_lora/adapter.safetensors"
                if (a.rebench and not os.path.exists("results/r5_bench_vllm_ft.json")
                        and not os.path.exists(f"{FT_DIR}/config.json") and os.path.exists(adapter)):
                    step("rebuild_ft", [PY, "-m", "nyaya.lora", "--model", a.llm, "--apply_adapter", adapter,
                                        "--out", FT_DIR], "r5_rebuild_ft", gpu, timeout=3600,
                         done_file=f"{FT_DIR}/config.json")
            return
        if step("traindata", [PY, "scripts/r5_steps.py", "traindata", *a.size_train], "r5_traindata",
                timeout=3600, done_file="data/r5_train.jsonl"):
            step("train", [PY, "-m", "nyaya.lora", "--model", a.llm, "--data", "data/r5_train.jsonl", "--out", FT_DIR,
                           "--adapter_out", "results/r5_lora", "--max_minutes", str(a.train_minutes)],
                 "r5_train", gpu, timeout=int(a.train_minutes * 60 + 3600), done_file=f"{FT_DIR}/config.json")
    finally:
        ev_train.set()


def lane_serving(a, gpu, ev_retrieval, ev_train, ev_vllm_env, ev_sglang_env=None):
    ev_retrieval.wait()
    if not os.path.exists("results/r5_prompts.json"):
        log("no prompts: retrieval failed, nothing to serve")
        return
    # transformers baseline (and, if vLLM is unavailable, the slow fallback for every test)
    need_hf = not os.path.exists("results/r5_bench_hf.json") or (a.no_vllm and not os.path.exists("results/r5_gen_base.json"))
    if need_hf:
        s = start_hf(a.llm, a.llm, gpu)
        if s:
            step("bench_hf", [PY, "scripts/r5_steps.py", "bench", "--base", s.base, "--model", a.llm, "--tag", "hf",
                              "--levels", "1,4", "--requests", "8", "--gpu", str(gpu)], "r5_bench_hf",
                 timeout=3600, done_file="results/r5_bench_hf.json")
            if a.no_vllm:
                hf_generate(a, s, "base")
            s.stop()
        else:
            log("the transformers server did not start (see logs/hf_*.log)")
    if a.no_vllm:
        ev_train.wait()
        if os.path.exists(f"{FT_DIR}/config.json") and not os.path.exists("results/r5_gen_ft.json"):
            s = start_hf(FT_DIR, FT_DIR, gpu)
            if s:
                hf_generate(a, s, "ft", model=FT_DIR)
                s.stop()
        return
    ev_vllm_env.wait()
    if ev_sglang_env is not None:                 # benchmark only once no install competes for the CPU
        ev_sglang_env.wait(timeout=2400)
    served = "base"
    base_done = all(os.path.exists(f) for f in ("results/r5_bench_vllm.json", "results/r5_gen_base.json"))
    s = None if base_done else start_vllm(a.llm, served, gpu)       # a re-run with earlier results skips this
    if s is None and base_done:
        log("base-model results already present; skipping to the fine-tuned model")
    elif s is None:
        log("vLLM did not start with any setting; falling back to the transformers server (smaller test sets)")
        a.no_vllm = True
        return lane_serving(a, gpu, ev_retrieval, ev_train, ev_vllm_env, ev_sglang_env)
    if s is not None:
        step("bench_vllm", [PY, "scripts/r5_steps.py", "bench", "--base", s.base, "--model", served, "--tag", "vllm",
                            "--gpu", str(gpu)], "r5_bench_vllm", timeout=3600, done_file="results/r5_bench_vllm.json")
        step("generate_base", [PY, "scripts/r5_steps.py", "generate", "--base", s.base, "--model", served, "--tag", "base"],
             "r5_gen_base", timeout=4 * 3600, done_file="results/r5_gen_base.json")
        s.stop()
    # SGLang on the same GPU, model and prompts, while fine-tuning runs on the other GPU
    if ev_sglang_env is not None and not a.no_sglang and not os.path.exists("results/r5_bench_sglang.json"):
        ev_sglang_env.wait()
        py = a.sglang_py
        sg = start_sglang(py, a.llm, served, gpu) if py else None
        if sg:
            step("bench_sglang", [PY, "scripts/r5_steps.py", "bench", "--base", sg.base, "--model", served, "--tag", "sglang",
                                  "--gpu", str(gpu)], "r5_bench_sglang", timeout=3600, done_file="results/r5_bench_sglang.json")
            sg.stop()
        else:
            log("SGLang not available on this machine; vLLM results stand alone")
    ev_train.wait()
    if os.path.exists("results/r5_gen_ft.json") and (os.path.exists("results/r5_bench_vllm_ft.json") or a.no_vllm):
        return
    if not os.path.exists(f"{FT_DIR}/config.json"):
        log("no fine-tuned model; skipping its evaluation")
        return
    s = start_vllm(FT_DIR, "ft", gpu)
    if s is None:
        log("vLLM did not start for the fine-tuned model; using the transformers server (smaller test sets)")
        s = start_hf(FT_DIR, FT_DIR, gpu)
        if s:
            hf_generate(a, s, "ft", model=FT_DIR)
            s.stop()
        return
    step("generate_ft", [PY, "scripts/r5_steps.py", "generate", "--base", s.base, "--model", "ft", "--tag", "ft"],
         "r5_gen_ft", timeout=4 * 3600, done_file="results/r5_gen_ft.json")
    step("bench_vllm_ft", [PY, "scripts/r5_steps.py", "bench", "--base", s.base, "--model", "ft", "--tag", "vllm_ft",
                           "--levels", "1,16", "--gpu", str(gpu)], "r5_bench_vllm_ft", timeout=3600,
         done_file="results/r5_bench_vllm_ft.json")
    s.stop()


def record_environment():
    """What a later reader needs to reproduce the numbers: GPU and driver, Python, every installed
    package, and the exact model revisions downloaded (the snapshot hash is part of each path)."""
    env = {"python": sys.version.split()[0], "cuda_device_order": os.environ.get("CUDA_DEVICE_ORDER")}
    try:
        env["gpus"] = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
                                     capture_output=True, text=True, timeout=30).stdout.strip().splitlines()
    except Exception:
        env["gpus"] = []
    if os.path.exists("data/prep_status.json"):
        env["downloads"] = json.load(open("data/prep_status.json"))
    if os.path.isdir("data/MultiConIR/.git"):                       # exact benchmark version
        env["multiconir_commit"] = subprocess.run(["git", "-C", "data/MultiConIR", "rev-parse", "HEAD"],
                                                  capture_output=True, text=True).stdout.strip()
    json.dump(env, open("results/r5_environment.json", "w"), indent=1)
    with open("results/r5_pip_freeze.txt", "w") as f:
        subprocess.run([PY, "-m", "pip", "freeze"], stdout=f, stderr=subprocess.DEVNULL)


def freeze_env(py, name):
    with open(f"results/r5_pip_freeze_{name}.txt", "w") as f:
        subprocess.run([PY, "-m", "uv", "pip", "freeze", "--python", py], stdout=f, stderr=subprocess.DEVNULL)


def hf_generate(a, s, tag, model=None):
    step(f"generate_{tag}", [PY, "scripts/r5_steps.py", "generate", "--base", s.base, "--model", model or a.llm,
                             "--tag", tag, "--concurrency", "2", "--limit", str(a.hf_limit)],
         f"r5_gen_{tag}", timeout=4 * 3600, done_file=f"results/r5_gen_{tag}.json")


# --------------------------------------------------------------------------- summary

def _load(name):
    p = f"{ROOT}/results/{name}.json"
    return json.load(open(p)) if os.path.exists(p) else None


def _boot(x, n=5000, seed=0):
    import numpy as np
    x = np.asarray([v for v in x if v is not None], dtype=float)
    if len(x) == 0:
        return None
    m = x[np.random.default_rng(seed).integers(0, len(x), size=(n, len(x)))].mean(1)
    return float(x.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def _paired(a, b, n=5000, seed=0):
    import numpy as np
    pairs = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
    if not pairs:
        return None
    d = np.asarray([x - y for x, y in pairs], dtype=float)
    m = d[np.random.default_rng(seed).integers(0, len(d), size=(n, len(d)))].mean(1)
    return float(d.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def _fmt(ci, pct=False):
    if ci is None:
        return "–"
    k = 100 if pct else 1
    f = "{:+.1f}" if pct else "{:.3f}"
    if pct:
        return f"{ci[0] * k:+.1f} [{ci[1] * k:+.1f}, {ci[2] * k:+.1f}]"
    return f"{f.format(ci[0])} [{f.format(ci[1])}, {f.format(ci[2])}]"


def _rate(ci):
    return "–" if ci is None else f"{100 * ci[0]:.1f}% [{100 * ci[1]:.1f}, {100 * ci[2]:.1f}]"


def answer_stats(g):
    A, U = g["arms"]["answer"], g["arms"]["unanswerable"]
    return {"EM": _boot([r["em"] for r in A]), "F1": _boot([r["f1"] for r in A]),
            "citation precision": _boot([r["cite_precision"] for r in A]),
            "citation recall": _boot([r["cite_recall"] for r in A]),
            "false abstention (answerable)": _boot([float(r["abstained"]) for r in A]),
            "abstention (gold withheld)": _boot([float(r["abstained"]) for r in U]),
            "well-formed replies": _boot([float(r["well_formed"]) for r in A + U]),
            "answer found in its cited sources": _boot([r.get("grounded") for r in A]),
            "answered without any valid citation": _boot([r.get("unsupported") for r in A]),
            "invalid citations removed (per reply)": _boot([r["invalid_citations"] for r in A + U]),
            "replies cut off at the token limit": _boot([float(r.get("truncated", False)) for r in A + U])}


def summary():
    L = ["# Round 5: the whole RAG system, served and measured", ""]
    st = status()
    rt = _load("r5_retrieval")
    if rt:
        lat = rt["latency"]
        L += ["## Retrieval (HotpotQA, Axon on LateOn)", "",
              f"{rt['questions']} questions, a corpus of {rt['corpus_paragraphs']} paragraphs, indexed in {rt['index_seconds']} s.", "",
              "| Measure | Value |", "|---|---|",
              f"| recall@5, both gold paragraphs (BM25 + ColBERT, RRF, Axon re-rank) | {rt['recall@5_both_gold']:.3f} |",
              f"| recall@5, both gold, BM25 + ColBERT fused without re-ranking | {rt.get('recall@5_both_gold_fused_only', float('nan')):.3f} |",
              f"| recall@5, at least one gold paragraph | {rt['recall@5_any_gold']:.3f} |"]
        for k in ("bm25", "exact", "plaid"):
            if f"{k}_recall@100" in rt:
                lab = {"bm25": "BM25", "exact": "ColBERT exact search", "plaid": "ColBERT PLAID (approximate)"}[k]
                L.append(f"| {lab} recall@100 | {rt[f'{k}_recall@100']:.3f} (median {lat[k]['median_ms']} ms, p95 {lat[k]['p95_ms']} ms per query) |")
        L.append(f"| full retrieval pipeline per query | median {lat['pipeline']['median_ms']} ms, p95 {lat['pipeline']['p95_ms']} ms, "
                 f"p99 {lat['pipeline'].get('p99_ms', '–')} ms |")
        rk = rt.get("ranking")
        if rk:
            L += ["", "| Ranking (gold = the 2 supporting paragraphs) | BM25 + ColBERT fused | + Axon re-rank | re-rank fused with BM25 again |",
                  "|---|---|---|---|"]
            for m in ("recall@5", "recall@10", "recall@20", "both@5", "mrr@10", "ndcg@10"):
                L.append(f"| {m} | {rk['fused'][m]:.3f} | {rk['reranked'][m]:.3f} | {rk.get('reranked_bm25', {}).get(m, float('nan')):.3f} |")
            if rt.get("generator_ranking"):
                L.append(f"\nThe generator was given the top 5 of **{rt['generator_ranking']}**, chosen beforehand on the 200 "
                         "held-out questions (both gold paragraphs in the top 5), not on the 500 graded ones.")
        ti = rt.get("tenant_isolation")
        if ti:
            L += ["", f"Tenant isolation (corpus split between two tenants, {ti['queries']} queries as tenant t0): "
                      f"{ti['cross_tenant_results']} cross-tenant results among {ti['results_returned']} returned; "
                      f"a query without a tenant was {'refused' if ti['query_without_tenant_refused'] else 'NOT refused'}."]
        if rt.get("plaid_error"):
            L.append(f"| PLAID | not built: {rt['plaid_error'][:120]} |")
        L.append("")
    header = False
    for tag, title in (("hf", "transformers server, one request at a time"), ("vllm", "vLLM"), ("sglang", "SGLang"),
                       ("vllm_ft", "vLLM, fine-tuned model")):
        b = _load(f"r5_bench_{tag}")
        if not b:
            continue
        if not header:
            header = True
            rerun = os.path.exists(f"{ROOT}/results/r5_bench_vllm_run1.json")
            toks = [r["mean_prompt_tokens"] for r in b["rows"]]
            L += [f"## Q9 Serving (T4, fp16, Qwen3-4B-Instruct-2507, ~{round(sum(toks) / len(toks), -1):.0f}-token RAG prompts, "
                  "128-token outputs)", "",
                  "Every request uses a prompt not sent before in that benchmark, so prefix caching helps only with the "
                  "shared system prompt. " +
                  ("vLLM and SGLang were measured back to back in a separate serving-only run, with nothing running on "
                   "the other GPU (the transformers baseline is from the first run, measured while GPU 1 was "
                   "fine-tuning); the first run's figures are kept in results/r5_bench_*_run1.json."
                   if rerun else "The base-model benchmarks ran on GPU 0 while GPU 1 was fine-tuning."), ""]
        ver = st.get("vllm_ft" if tag == "vllm_ft" else "vllm_base", {}).get("version", "") if tag.startswith("vllm") else \
            st.get("sglang_base", {}).get("version", "") if tag == "sglang" else ""
        note = "" if all(r.get("fixed_length", True) for r in b["rows"]) else ", outputs not fixed-length"
        L += [f"**{title}**" + (f" ({ver}{note})" if ver or note else ""), "",
              "| Concurrency | TTFT p50 / p99 ms | ITL p50 / p99 ms | TPOT ms | e2e p50 ms | requests/s | output tokens/s | GPU memory MiB | GPU util % | SM clock MHz | max temp °C | failed |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in b["rows"]:
            L.append(f"| {r['concurrency']} | {r['ttft_ms_p50']} / {r['ttft_ms_p99']} | {r['itl_ms_p50']} / {r['itl_ms_p99']} | "
                     f"{r['tpot_ms_mean']} | {r['e2e_ms_p50']} | {r['requests_per_s']} | {r['output_tokens_per_s']} | "
                     f"{r.get('gpu_mem_peak_mib', '–')} | {r.get('gpu_util_mean_pct', '–')} | {r.get('gpu_sm_clock_mean_mhz', '–')} | "
                     f"{r.get('gpu_temp_max_c', '–')} | {r['failed']} |")
        srv = [(k, v) for r in b["rows"][-1:] for k, v in r["server"].items()
               if any(s in k for s in ("time_to_first_token", "inter_token", "time_per_output", "e2e_request", "prefix_cache"))]
        if srv:
            L += ["", "Server-side (Prometheus /metrics) at the highest concurrency: " +
                  ", ".join(f"`{k}` {v * 1000:.1f} ms" if "seconds" in k else f"`{k}` {v:g}" for k, v in srv[:8])]
        L.append("")
    base, ft = _load("r5_gen_base"), _load("r5_gen_ft")
    if base:
        L += ["## Q10 Answers, citations, abstention", "", "| Measure | base | fine-tuned | fine-tuned − base (paired) |", "|---|---|---|---|"]
        sb, sf = answer_stats(base), (answer_stats(ft) if ft else {})
        keys = {"EM": "em", "F1": "f1", "citation precision": "cite_precision", "citation recall": "cite_recall",
                "false abstention (answerable)": "abstained", "answer found in its cited sources": "grounded",
                "answered without any valid citation": "unsupported"}
        for k, v in sb.items():
            d = "–"
            if ft and k in keys:
                fk = keys[k]
                d = _fmt(_paired([float(r[fk]) if r[fk] is not None else None for r in ft["arms"]["answer"]],
                                 [float(r[fk]) if r[fk] is not None else None for r in base["arms"]["answer"]]), pct=True)
            if ft and k == "abstention (gold withheld)":
                d = _fmt(_paired([float(r["abstained"]) for r in ft["arms"]["unanswerable"]],
                                 [float(r["abstained"]) for r in base["arms"]["unanswerable"]]), pct=True)
            L.append(f"| {k} | {_fmt(v)} | {_fmt(sf.get(k)) if ft else '–'} | {d} |")
        for tag2, g in (("base", base), ("fine-tuned", ft)):
            if g and (g.get("errors") or 0):
                L.append(f"\n{tag2}: {g['errors']} requests failed after 3 attempts ({g.get('errors_by_arm', {})}); "
                         "they count as wrong answers.")
        L += ["", "**With the no-citation-no-answer rule.** The rule was added after the first round 5 run, so it is reported "
                  "separately, not in place of the pre-registered results. It is applied to the same replies, "
                  "and any answer that cites no source counts as a refusal.", "",
              "| Measure | base | fine-tuned | fine-tuned − base (paired) |", "|---|---|---|---|"]

        def ruled(g):
            A, U = g["arms"]["answer"], g["arms"]["unanswerable"]
            return {"EM": [0.0 if r.get("unsupported") else r["em"] for r in A],
                    "F1": [0.0 if r.get("unsupported") else r["f1"] for r in A],
                    "false abstention (answerable)": [float(r["abstained"] or bool(r.get("unsupported"))) for r in A],
                    "abstention (gold withheld)": [float(r["abstained"] or bool(r.get("unsupported"))) for r in U]}
        rb, rf = ruled(base), (ruled(ft) if ft else {})
        for k2, vb in rb.items():
            L.append(f"| {k2} | {_fmt(_boot(vb))} | {_fmt(_boot(rf[k2])) if ft else '–'} | "
                     f"{_fmt(_paired(rf[k2], vb), pct=True) if ft else '–'} |")
        L += ["", "## Q11 Prompt injection (attack success rate, lower is better)", "",
              "| Prompt | base | fine-tuned | EM under attack (base) |", "|---|---|---|---|"]
        for arm, lab in (("undefended", "undefended (plain concatenation)"), ("defended", "defended (escaped sources, data-not-instructions rule)"),
                         ("scanned", "defended + input scanner")):
            rb = base["arms"].get(f"injection_{arm}", [])
            rf = ft["arms"].get(f"injection_{arm}", []) if ft else []
            L.append(f"| {lab} | {_rate(_boot([float(r['attacked']) for r in rb]))} | "
                     f"{_rate(_boot([float(r['attacked']) for r in rf])) if rf else '–'} | {_fmt(_boot([r['em'] for r in rb]))} |")
        if rt and "scanner" in rt:
            sc = rt["scanner"]
            L += ["", f"Input scanner: detected {sc['detected']}/{sc['injected']} planted injections "
                      f"(by template: {sc['detected_by_template']}); flagged {sc['false_positives']} of {sc['clean_paragraphs']} "
                      f"clean paragraphs (false-positive rate {100 * sc['false_positive_rate']:.2f}%)."]
            rb = base["arms"].get("injection_defended", [])
            if rb:
                by_t = {}
                for r in rb:
                    by_t.setdefault(r["template"], []).append(float(r["attacked"]))
                from nyaya.rag_eval import TEMPLATE_NAMES
                L.append("Defended prompt, base model, success by attack: " +
                         ", ".join(f"{TEMPLATE_NAMES[t]} {100 * sum(v) / len(v):.0f}%" for t, v in sorted(by_t.items())) + ".")
        L += ["", "## Q12 Condition check: LLM alone vs LLM + Axon certificate", ""]
        ca, cc = base["arms"].get("condition_alone", []), base["arms"].get("condition_cert", [])
        if ca and cc:
            def split(rows, want_all):
                return [r["correct"] for r in rows if (r["truth"] == "ALL") == want_all]
            L += ["| Documents | LLM alone | LLM + certificate | certificate − alone (paired) |", "|---|---|---|---|"]
            for lab, sel in ((f"all {len(ca)}", None), ("satisfy every condition (truth ALL)", True), ("break one condition", False)):
                xa = [r["correct"] for r in ca] if sel is None else split(ca, sel)
                xc = [r["correct"] for r in cc] if sel is None else split(cc, sel)
                L.append(f"| {lab} | {_rate(_boot(xa))} | {_rate(_boot(xc))} | {_fmt(_paired(xc, xa), pct=True)} |")
            if ft and ft["arms"].get("condition_cert"):
                fc, fa = ft["arms"]["condition_cert"], ft["arms"]["condition_alone"]
                L.append(f"| fine-tuned model, all {len(fa)} | {_rate(_boot([r['correct'] for r in fa]))} | "
                         f"{_rate(_boot([r['correct'] for r in fc]))} | {_fmt(_paired([r['correct'] for r in fc], [r['correct'] for r in fa]), pct=True)} |")
            ax = axon_alone()
            if ax:
                L += ["", f"Axon alone (no LLM; ALL if the weakest condition's relative score ≥ {ax['threshold']:.3f}, "
                          f"threshold fixed on {ax['n_calib']} calibration documents): {_rate(ax['ci'])}."]
            unparsed = sum(1 for r in ca + cc if r["verdict"] is None)
            L.append(f"Replies without a parsable verdict: {unparsed} of {len(ca) + len(cc)}.")
    lr = _load("r5_lora/train_report")
    if lr:
        L += ["", "## Q13 Fine-tuning run", "",
              f"LoRA r={lr['r']}, alpha={lr['alpha']}, lr {lr['lr']}, {lr['examples']} examples "
              f"({lr['skipped_too_long']} too long), {lr['steps']}/{lr['planned_steps']} steps in {lr['minutes']:.0f} min, "
              f"{lr['trainable_params'] / 1e6:.1f} M trainable parameters; loss {lr['first_loss']:.3f} → {lr['last_loss']:.3f}; "
              f"{lr['non_finite_losses']} non-finite losses skipped."]
    bad = [k for k, v in st.items() if isinstance(v, dict) and v.get("ok") is False and not k.endswith("_run1")]
    L += ["", f"Steps that failed: {', '.join(bad)} (see logs/)." if bad else "All steps succeeded.", ""]
    txt = "\n".join(L) + "\n"
    open(f"{ROOT}/results/ROUND5.md", "w").write(txt)
    return txt


def axon_alone():
    P = None
    p = f"{ROOT}/results/r5_prompts.json"
    if os.path.exists(p):
        P = json.load(open(p))
    if not P or not P.get("condition_calib"):
        return None

    def verdicts(rows, thr):
        return [("ALL" if min(r["relative"]) >= thr else r["weakest"]) for r in rows]
    cal = P["condition_calib"]
    cands = sorted({min(r["relative"]) for r in cal})
    best = max(cands, key=lambda t: sum(v == r["truth"] for v, r in zip(verdicts(cal, t), cal)))
    test = P["condition"]
    ok = [float(v == r["truth"]) for v, r in zip(verdicts(test, best), test)]
    return {"threshold": best, "n_calib": len(cal), "ci": _boot(ok)}


# --------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", default=LLM)
    ap.add_argument("--retriever", default=RETRIEVER)
    ap.add_argument("--summary_only", action="store_true")
    ap.add_argument("--no_vllm", action="store_true", help="serve with nyaya.hf_server only")
    ap.add_argument("--skip_train", action="store_true")
    ap.add_argument("--no_sglang", action="store_true", help="skip the SGLang comparison")
    ap.add_argument("--rebench", action="store_true", help="serving benchmarks again, reusing every other result")
    ap.add_argument("--train_minutes", type=float, default=100)
    ap.add_argument("--hf_limit", type=int, default=100, help="test items per arm when only the slow server is available")
    ap.add_argument("--smoke", action="store_true", help="tiny sizes for a local check")
    a = ap.parse_args()
    os.chdir(ROOT)
    # number GPUs the same way in CUDA and nvidia-smi, so GPU memory is read from the right card
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    for d in ("results", "logs", "data"):
        os.makedirs(d, exist_ok=True)
    if a.summary_only:
        print(summary())
        return
    if a.rebench:
        if not os.path.exists("results/r5_gen_ft.json"):
            raise SystemExit("--rebench needs an earlier run's results/ folder (copy it in first)")
        for tag in ("vllm", "vllm_ft", "sglang"):
            old, keep = f"results/r5_bench_{tag}.json", f"results/r5_bench_{tag}_run1.json"
            if os.path.exists(old) and not os.path.exists(keep):
                os.replace(old, keep)
        st = status()
        if "sglang_base" in st:                    # the earlier failed attempt is history, not this run's result
            st["sglang_base_run1"] = st.pop("sglang_base")
            json.dump(st, open(STATUS, "w"), indent=1)
    a.size_retrieve = (["--n_answer", "12", "--n_unanswerable", "6", "--n_injection", "10", "--n_condition", "10",
                        "--n_calib", "6"] if a.smoke else [])
    a.size_train = ["--n_train", "16"] if a.smoke else []
    import torch
    n = torch.cuda.device_count()
    log(f"GPUs: {n}; generator {a.llm}; retriever {a.retriever}")
    if shutil_which("nvidia-smi"):
        subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv"])
    if not a.smoke:
        sh([PY, "scripts/prep_data.py", "--data", "data", "--only", "model", "multiconir", "--model", a.llm, a.retriever],
           "r5_prep")
    else:
        sh([PY, "scripts/prep_data.py", "--data", "data", "--only", "multiconir"], "r5_prep")
    record_environment()
    ev_r, ev_t, ev_v = threading.Event(), threading.Event(), threading.Event()

    ev_s = threading.Event()
    a.sglang_py = None

    def install():
        try:
            if not a.no_vllm:
                vllm_python(VLLM_PIN)
        finally:
            ev_v.set()
        try:
            if not a.no_sglang and not os.path.exists("results/r5_bench_sglang.json"):
                a.sglang_py = sglang_python()
        finally:
            ev_s.set()
    log(f"disk free: {free_gb():.0f} GB")
    status("disk", free_gb=round(free_gb(), 1))
    threading.Thread(target=install, daemon=True).start()
    gA, gB = (0, 1) if n >= 2 else (0, 0)
    if gA == gB:                                    # one GPU: run the lanes one after the other
        lane_retrieval_and_training(a, gB, ev_r, ev_t)
        lane_serving(a, gA, ev_r, ev_t, ev_v, ev_s)
    else:
        tb = threading.Thread(target=lane_retrieval_and_training, args=(a, gB, ev_r, ev_t))
        tb.start()
        lane_serving(a, gA, ev_r, ev_t, ev_v, ev_s)
        tb.join()
    print("\n" + summary(), flush=True)


def shutil_which(x):
    import shutil
    return shutil.which(x)


if __name__ == "__main__":
    main()
