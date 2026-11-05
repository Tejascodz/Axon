"""Defences against corpus poisoning: documents planted so that they are retrieved for a question and
carry a false answer (PoisonedRAG-style attacks). Round 9 measures both, including what they cost on
clean questions.

echo filter       A planted passage usually restates the question to rank first for it. Passages
                  that contain most of the question's word sequences are withheld. Cheap, but an
                  attacker who paraphrases the question avoids it.
isolate and vote  Answer from each retrieved passage alone, then keep the answer only if it agrees
                  with what most of the passages say on their own (as in RobustRAG). A few planted
                  passages cannot outvote the genuine ones, unless they outnumber them in the
                  top k. Costs one extra LLM call per passage.
"""
from __future__ import annotations

import re
import string
from collections import Counter

from . import guards


def _words(s: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", s.lower())


def normalize(s: str) -> str:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def echoes_question(text: str, question: str, n: int = 3, share: float = 0.5) -> bool:
    """True when the passage contains at least `share` of the question's word n-grams."""
    q = _words(question)
    if len(q) < n + 1:
        return False
    grams = {tuple(q[i:i + n]) for i in range(len(q) - n + 1)}
    t = _words(text)
    have = {tuple(t[i:i + n]) for i in range(len(t) - n + 1)}
    return len(grams & have) / len(grams) >= share


def drop_echoes(sources: list[dict], question: str):
    kept, dropped = [], []
    for s in sources:
        (dropped if echoes_question(s["text"], question) else kept).append(s)
    return kept, dropped


def agree(a: str, b: str) -> bool:
    a, b = normalize(a), normalize(b)
    return bool(a) and bool(b) and (a == b or a in b or b in a)


def isolate_and_vote(question: str, sources: list[dict], min_votes: int = 2, **llm_kw) -> dict:
    """Answer from each source alone; the majority answer, if one has at least `min_votes` votes."""
    from .llm import chat
    answers = []
    for s in sources:
        reply = chat(guards.build_messages(question, [s]), max_tokens=64, **llm_kw)
        chk = guards.check_answer(reply, [s["id"]])
        answers.append(None if chk.abstained else chk.answer)
    groups: list[list[str]] = []
    for a in (x for x in answers if x):
        for g in groups:
            if agree(a, g[0]):
                g.append(a)
                break
        else:
            groups.append([a])
    groups.sort(key=len, reverse=True)
    top = groups[0] if groups else []
    clear = bool(top) and len(top) >= min_votes and (len(groups) == 1 or len(top) > len(groups[1]))
    return {"majority": Counter(top).most_common(1)[0][0] if clear else None, "votes": [len(g) for g in groups],
            "per_source": answers}
