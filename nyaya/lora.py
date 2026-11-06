"""LoRA fine-tuning of a causal language model, written out in full (no PEFT).

  W x  ->  W x + (alpha / r) * B (A x)        A: r x in (random), B: out x r (zeros), W frozen

Training runs in fp16 on GPUs without bf16 (T4): frozen base weights in fp16, adapter weights in
fp32, autocast + loss scaling. The loss covers only the assistant's reply, and logits are computed
only at those positions (a 150k-word vocabulary over the whole prompt would not fit in memory).
After training the adapters are merged into W, so the result is an ordinary checkpoint that any
server (vLLM, SGLang, nyaya.hf_server) loads with no extra code.

    python -m nyaya.lora --model Qwen/Qwen3-4B-Instruct-2507 --data train.jsonl --out ft_model
    # train.jsonl: {"messages": [...], "target": "..."} per line
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int = 16, alpha: float = 32.0, dropout: float = 0.05):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        dev = base.weight.device
        self.A = nn.Parameter(torch.empty(r, base.in_features, device=dev, dtype=torch.float32))
        self.B = nn.Parameter(torch.zeros(base.out_features, r, device=dev, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.scale = alpha / r
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        y = self.base(x)
        z = F.linear(F.linear(self.drop(x), self.A.to(x.dtype)), self.B.to(x.dtype))
        return y + z * self.scale

    @torch.no_grad()
    def merged(self) -> nn.Linear:
        w = self.base.weight
        w.copy_((w.float() + self.scale * (self.B @ self.A)).to(w.dtype))
        return self.base


def inject(model: nn.Module, targets=TARGETS, r=16, alpha=32.0, dropout=0.05) -> list[LoRALinear]:
    for p in model.parameters():
        p.requires_grad_(False)
    added = []
    for name, mod in list(model.named_modules()):
        for child_name, child in list(mod.named_children()):
            if child_name in targets and isinstance(child, nn.Linear):
                lo = LoRALinear(child, r, alpha, dropout)
                setattr(mod, child_name, lo)
                added.append(lo)
    if not added:
        raise ValueError(f"no linear layers named {targets}")
    return added


def merge(model: nn.Module) -> int:
    n = 0
    for mod in list(model.modules()):
        for child_name, child in list(mod.named_children()):
            if isinstance(child, LoRALinear):
                setattr(mod, child_name, child.merged())
                n += 1
    return n


def load_adapter(model: nn.Module, path: str, r: int, alpha: float) -> int:
    """Put a saved adapter (adapter.safetensors) back into a freshly loaded base model; every
    adapter tensor must match a layer, so a wrong base model or rank fails loudly."""
    from safetensors.torch import load_file
    state = load_file(path)
    inject(model, r=r, alpha=alpha, dropout=0.0)
    layers = {n: m for n, m in model.named_modules() if isinstance(m, LoRALinear)}
    if len(state) != 2 * len(layers):
        raise ValueError(f"adapter has {len(state)} tensors, model has {len(layers)} LoRA layers")
    with torch.no_grad():
        for key, t in state.items():
            name, w = key.rsplit(".", 1)
            p = getattr(layers[name], w)
            if p.shape != t.shape:
                raise ValueError(f"{key}: adapter {tuple(t.shape)} vs model {tuple(p.shape)}")
            p.copy_(t.float())
    return len(layers)


def adapter_state(model: nn.Module) -> dict:
    return {f"{n}.{w}": getattr(m, w).detach().half().cpu().contiguous()
            for n, m in model.named_modules() if isinstance(m, LoRALinear) for w in ("A", "B")}


def encode_example(tok, messages, target, max_len):
    prompt = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True)["input_ids"]
    reply = tok(target + (tok.eos_token or ""), add_special_tokens=False)["input_ids"]
    if len(prompt) + len(reply) > max_len:
        return None
    return prompt, reply


def dtype_kw(dtype):
    """`dtype=` on recent transformers, `torch_dtype=` on older ones (otherwise weights load in fp32)."""
    import transformers
    major, minor = (int(x) for x in transformers.__version__.split(".")[:2])
    return {"dtype": dtype} if (major, minor) >= (4, 56) else {"torch_dtype": dtype}


def _decoder(model):
    return model.model if hasattr(model, "model") else model.get_decoder()


def reply_loss(model, prompt, reply, device):
    ids = torch.tensor([prompt + reply], device=device)
    h = _decoder(model)(input_ids=ids).last_hidden_state                 # [1, T, H]
    P, R = len(prompt), len(reply)
    logits = model.lm_head(h[:, P - 1:P + R - 1]).float()                # only where the reply is predicted
    return F.cross_entropy(logits.view(-1, logits.shape[-1]), ids[0, P:P + R])


def train(model, tok, examples, out_dir=None, lr=1e-4, epochs=1, accum=8, r=16, alpha=32.0, dropout=0.05,
          max_len=2048, warmup=20, seed=0, log=print, max_minutes=None):
    """examples: [{"messages": [...], "target": "..."}]. Returns a training report."""
    device = next(model.parameters()).device
    fp16 = device.type == "cuda"
    random.seed(seed)
    torch.manual_seed(seed)
    data = [e for e in (encode_example(tok, x["messages"], x["target"], max_len) for x in examples) if e]
    skipped_long = len(examples) - len(data)
    layers = inject(model, r=r, alpha=alpha, dropout=dropout)
    params = [p for m in layers for p in (m.A, m.B)]
    n_train = sum(p.numel() for p in params)
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if hasattr(model, "config"):
        model.config.use_cache = False
    model.train()
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    steps = math.ceil(len(data) * epochs / accum)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warmup) *
                                              0.5 * (1 + math.cos(math.pi * min(1.0, s / max(1, steps)))))
    scaler = torch.amp.GradScaler("cuda", enabled=fp16)
    hist, bad, step, t0 = [], 0, 0, time.time()
    order = [i for _ in range(epochs) for i in random.sample(range(len(data)), len(data))]
    window = []
    oom = 0
    for k, i in enumerate(order):
        try:
            with torch.autocast("cuda", dtype=torch.float16, enabled=fp16):
                loss = reply_loss(model, *data[i], device)
            if not torch.isfinite(loss):
                bad += 1
                continue
            scaler.scale(loss / accum).backward()
        except torch.cuda.OutOfMemoryError:                   # one over-long example must not end the run
            oom += 1
            loss, window = None, []                              # drop this accumulation window entirely
            for p in params:
                p.grad = None
            torch.cuda.empty_cache()
            continue
        window.append(loss.item())
        if window and ((k + 1) % accum == 0 or k == len(order) - 1):
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            sched.step()
            step += 1
            hist.append(sum(window) / len(window))
            window = []
            if step % 10 == 0 or step == steps:
                log(f"step {step}/{steps}  loss {sum(hist[-10:]) / len(hist[-10:]):.4f}  "
                    f"{(time.time() - t0) / 60:.1f} min")
            if max_minutes and (time.time() - t0) / 60 > max_minutes:
                log(f"time budget reached at step {step}")
                break
    model.eval()
    report = {"examples": len(data), "skipped_too_long": skipped_long, "non_finite_losses": bad, "out_of_memory": oom, "steps": step,
              "planned_steps": steps, "trainable_params": n_train, "first_loss": hist[0] if hist else None,
              "last_loss": sum(hist[-10:]) / len(hist[-10:]) if hist else None, "minutes": (time.time() - t0) / 60,
              "r": r, "alpha": alpha, "lr": lr, "accum": accum, "epochs": epochs}
    if out_dir:
        from safetensors.torch import save_file
        os.makedirs(out_dir, exist_ok=True)
        save_file(adapter_state(model), os.path.join(out_dir, "adapter.safetensors"))
        with open(os.path.join(out_dir, "train_report.json"), "w", encoding="utf-8") as f:
            json.dump(report, f, indent=1)
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", default=None, help="training data (jsonl); not needed with --apply_adapter")
    ap.add_argument("--apply_adapter", default=None, help="rebuild a merged model from a saved adapter, no training")
    ap.add_argument("--out", required=True, help="merged checkpoint directory")
    ap.add_argument("--adapter_out", default=None, help="where to keep the adapter and report (small)")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--r", type=int, default=16)
    ap.add_argument("--max_len", type=int, default=2048)
    ap.add_argument("--max_minutes", type=float, default=None)
    ap.add_argument("--revision", default=None, help="model commit hash (pins the exact base weights)")
    a = ap.parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(a.model, revision=a.revision)
    model = AutoModelForCausalLM.from_pretrained(a.model, revision=a.revision, attn_implementation="sdpa",
                                                 **dtype_kw(torch.float16 if dev == "cuda" else torch.float32)).to(dev)
    if a.apply_adapter:
        n = load_adapter(model, a.apply_adapter, r=a.r, alpha=2.0 * a.r)
        merge(model)
        model.config.use_cache = True
        model.save_pretrained(a.out, safe_serialization=True)
        tok.save_pretrained(a.out)
        print(json.dumps({"rebuilt_from": a.apply_adapter, "lora_layers": n}), flush=True)
        return
    if not a.data:
        ap.error("--data is required unless --apply_adapter is given")
    with open(a.data, encoding="utf-8") as f:
        examples = [json.loads(line) for line in f]
    rep = train(model, tok, examples, out_dir=a.adapter_out or a.out, lr=a.lr, epochs=a.epochs, accum=a.accum,
                r=a.r, alpha=2.0 * a.r, max_len=a.max_len, max_minutes=a.max_minutes,
                log=lambda m: print(m, flush=True))
    n = merge(model)
    model.config.use_cache = True
    model.save_pretrained(a.out, safe_serialization=True)
    tok.save_pretrained(a.out)
    print(json.dumps({**rep, "merged_layers": n}), flush=True)


if __name__ == "__main__":
    main()
