"""Claim-level evidence checking: is every sentence of an answer supported by the sources it cites?

Citation checks (nyaya.guards) establish that an answer cites sources it was given. That is not
the same as the cited source supporting what the sentence says: a model can cite a relevant
passage while stating something the passage does not say. This module checks each claim against
its own cited passages with a natural-language-inference (NLI) model:

    claims("Answer: 1922\\nEvidence: The war ended in October 1922 [c3].")
        -> [Claim(text="The war ended in October 1922.", cited=["c3"])]
    Verifier().check(answer_text, sources)
        -> per-claim entailment probability, and whether the answer is supported

A claim is supported when the probability that its cited passages entail it is at least the
threshold (chosen on a calibration set, never on the test set). An answer is supported when every
claim is, and the answer string itself appears in a cited passage. An uncited claim is not
supported. Abstentions ("INSUFFICIENT") make no claims and pass.

`judge_with_llm` asks the generator itself instead (a common alternative), for comparison.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from . import guards

_CITE = re.compile(r"\[(c\d+)\]")
_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")
_ANSWER = re.compile(r"^\s*\**\s*answer\s*\**\s*:\s*(.+)$", re.I | re.M)
DEFAULT_MODEL = "cross-encoder/nli-deberta-v3-large"
FALLBACK_MODELS = ("cross-encoder/nli-deberta-v3-base", "microsoft/deberta-large-mnli")


@dataclass
class Claim:
    text: str                          # the sentence with citation markers removed
    cited: list[str]                   # source ids it cites
    score: float | None = None         # probability that its cited passages entail it
    supported: bool | None = None


@dataclass
class Verdict:
    supported: bool
    abstained: bool
    claims: list[Claim] = field(default_factory=list)
    answer_in_sources: bool | None = None
    reason: str = ""


def _norm(s: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", s.lower()))


def answer_value(reply: str) -> str | None:
    m = _ANSWER.search(reply)
    return _CITE.sub("", m.group(1)).strip().strip("*").strip() if m else None


def claims(reply: str) -> list[Claim]:
    """Every factual sentence of the reply, except the short "Answer:" line, with the citations it
    carries. Citations after a run of sentences cover the whole run ("A. B. [c1]" cites c1 for both
    A and B), the usual convention; sentences after the last citation of a line stay uncited. An abstention
    makes no claims: its explanation of what is missing is not a statement about the world."""
    ans = answer_value(reply)
    if ans and guards.INSUFFICIENT.lower() in ans.lower():
        return []
    out = []
    for line in reply.splitlines():
        line = line.strip()
        if not line or _ANSWER.match(line):
            continue
        line = re.sub(r"^\s*\**\s*evidence\s*\**\s*:\s*", "", line, flags=re.I)
        pending = []                                   # sentences waiting for the next citation in the line
        for sent in _SENT.split(line):
            cited = list(dict.fromkeys(_CITE.findall(sent)))
            text = re.sub(r"\s+([.,;:!?])", r"\1", " ".join(_CITE.sub("", sent).split()))
            if len(text.split()) >= 3 and guards.INSUFFICIENT.lower() not in text.lower():
                out.append(Claim(text=text, cited=cited))
                if not cited:
                    pending.append(out[-1])
            if cited:
                for c in pending:
                    c.cited = list(cited)
                pending = []
    return out


class Verifier:
    """NLI-based checker. `threshold` is the entailment probability needed; calibrate it with
    `nyaya.verify.calibrate` on held-out labelled pairs before relying on it."""

    def __init__(self, model: str = DEFAULT_MODEL, device: str | None = None, threshold: float = 0.5,
                 max_length: int = 512, batch_size: int = 16, revision: str | None = None):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        last = None
        for name in (model, *[m for m in FALLBACK_MODELS if m != model]):
            try:
                rev = revision if name == model else None          # a pinned commit applies to the chosen model only
                self.tok = AutoTokenizer.from_pretrained(name, revision=rev)
                self.model = AutoModelForSequenceClassification.from_pretrained(name, revision=rev).to(self.device).eval()
                self.name = name
                break
            except Exception as e:                       # next candidate
                last = e
        else:
            raise RuntimeError(f"no NLI model could be loaded: {last!r}")
        if torch.device(self.device).type == "cuda":
            self.model.half()
        labels = {v.lower(): int(k) for k, v in self.model.config.id2label.items()}
        self.entail = next(i for lab, i in labels.items() if lab.startswith("entail"))
        self.threshold, self.max_length, self.batch_size = threshold, max_length, batch_size

    def entailment(self, pairs: list[tuple[str, str]]) -> list[float]:
        """P(premise entails hypothesis) for each (premise, hypothesis) pair."""
        import torch
        out = []
        with torch.no_grad():
            for i in range(0, len(pairs), self.batch_size):
                prem, hyp = zip(*pairs[i:i + self.batch_size])
                enc = self.tok(list(prem), list(hyp), truncation="only_first", max_length=self.max_length,
                               padding=True, return_tensors="pt").to(self.device)
                p = self.model(**enc).logits.float().softmax(-1)[:, self.entail]
                out += p.cpu().tolist()
        return out

    def check(self, reply: str, sources: list[dict]) -> Verdict:
        """sources: [{"id", "text"}] as shown to the model."""
        text_of = {s["id"]: s["text"] for s in sources}
        chk = guards.check_answer(reply, list(text_of))
        if chk.abstained:
            return Verdict(supported=True, abstained=True, reason="abstained")
        cs = claims(reply)
        todo = [(i, " ".join(text_of[c] for c in cl.cited if c in text_of), cl.text)
                for i, cl in enumerate(cs) if any(c in text_of for c in cl.cited)]
        for (i, _, _), p in zip(todo, self.entailment([(prem, hyp) for _, prem, hyp in todo])):
            cs[i].score = p
        for cl in cs:
            cl.supported = cl.score is not None and cl.score >= self.threshold
        cited_text = " ".join(text_of[c] for c in dict.fromkeys(_CITE.findall(reply)) if c in text_of)
        ans = answer_value(reply)
        in_src = (_norm(ans) in _norm(cited_text)) if ans and ans.lower() not in ("yes", "no") else None
        if not cs:                                     # nothing but "Answer: X [c#]": X must be in what it cites
            ok = in_src is True
            why = "" if ok else "no evidence sentence, and the answer is not in a cited source"
        else:
            ok = all(c.supported for c in cs) and in_src is not False
            why = ("uncited or unsupported claim" if not all(c.supported for c in cs)
                   else "answer not in cited sources" if in_src is False else "")
        return Verdict(supported=ok, abstained=False, claims=cs, answer_in_sources=in_src, reason=why)


def calibrate(scores: list[float], labels: list[bool], target_precision: float | None = None) -> float:
    """Threshold on held-out labelled pairs: the one with the best F1 for "supported", or the lowest
    one reaching `target_precision` (fewer unsupported claims let through, more good ones withheld)."""
    pairs = sorted(zip(scores, labels), reverse=True)
    tp, fp, pos = 0, 0, sum(labels)
    best, best_t = -1.0, 0.5
    for s, y in pairs:
        tp, fp = tp + y, fp + (not y)
        prec, rec = tp / (tp + fp), tp / max(pos, 1)
        if target_precision is not None:
            if prec >= target_precision:
                best_t = s
            continue
        f1 = 2 * prec * rec / max(prec + rec, 1e-9)
        if f1 > best:
            best, best_t = f1, s
    return best_t


class LLMJudge(Verifier):
    """The same claim check, with an LLM deciding support (YES/NO) instead of an NLI model. In round 7
    it was more accurate on labelled pairs (F1 0.967 vs 0.883) but slower (82 vs 17 ms a claim) and
    uses the generator's GPU. `threshold` is fixed at 0.5: the judge answers yes or no."""

    def __init__(self, workers: int = 8, **llm_kw):
        self.name, self.threshold, self.workers, self.llm_kw = "llm-judge", 0.5, workers, llm_kw

    def entailment(self, pairs):
        from concurrent.futures import ThreadPoolExecutor

        def one(p):
            try:
                return 1.0 if judge_with_llm(p[0], p[1], **self.llm_kw) else 0.0
            except Exception:                      # a judge that cannot answer does not vouch for the claim
                return 0.0
        with ThreadPoolExecutor(self.workers) as ex:
            return list(ex.map(one, pairs))


JUDGE_PROMPT = ("Passage:\n{premise}\n\nClaim: {claim}\n\nDoes the passage support every part of the claim? "
                "Answer with one word: YES or NO.")


def judge_with_llm(premise: str, claim: str, **llm_kw) -> bool | None:
    """The LLM-as-judge alternative: True/False, or None if the reply is neither."""
    from .llm import chat
    r = chat([{"role": "user", "content": JUDGE_PROMPT.format(premise=premise, claim=claim)}],
             temperature=0.0, max_tokens=4, **llm_kw).strip().upper()
    return True if r.startswith("YES") else False if r.startswith("NO") else None
