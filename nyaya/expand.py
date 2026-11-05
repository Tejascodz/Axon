"""Query rewriting with the answering LLM, for questions retrieval alone handles badly. Both are off by
default and each costs LLM calls before retrieval; round 10 measures what they buy.

expansion   For long, reasoning-heavy questions (forum posts asking why something happens), the
            words of the question are often not the words of the passage that answers it. The LLM
            first writes what the answer would involve, and that text is searched for as well
            (query2doc, Wang et al. 2023; HyDE, Gao et al. 2022; BRIGHT's reasoning queries, Su et
            al., ICLR 2025, where LLM-written reasoning raised BM25 by several nDCG points).
              mode "bm25"     keyword search gets question + expansion; vector search and re-rank
                              keep the question
              mode "append"   every stage gets question + expansion
              mode "replace"  every stage gets the expansion alone
follow-up   For questions that chain facts (A's director's birthplace), the first search finds A, and
            the passage naming the director's birthplace is not described by the question. After
            each search the LLM writes the next search query, naming the entity found so far rather
            than a pronoun (IRCoT, Trivedi et al. 2023; entity completion as in ChainRAG, 2025), or
            says nothing more is needed.
"""
from __future__ import annotations

import re

MODES = ("bm25", "append", "replace")

EXPAND_PROMPT = """A user asked the question below. Before searching a document collection for it:
1. State in one sentence the essential problem the user needs solved.
2. Think step by step about what concepts, mechanisms, terms and facts a document that answers it would contain.
3. Write a short draft answer, using the technical terms such a document would use.
Write plain prose, at most 200 words, no headings.

Question:
{q}"""

FOLLOW_UP_PROMPT = """Question: {q}

Passages found so far:
{passages}

To answer the question, which single fact is still missing from these passages? Write one short search query
for it. Name people, places and works explicitly, using names found in the passages, never pronouns such as
"he", "it" or "this film". If the passages already hold everything needed, write DONE.
Reply with the query or DONE only, on one line."""


def expand(question: str, max_tokens: int = 320, **llm_kw) -> str:
    """The LLM's reasoning about what an answering document contains ('' if the call fails)."""
    from .llm import chat
    try:
        out = chat([{"role": "user", "content": EXPAND_PROMPT.format(q=question[:6000])}], temperature=0.0,
                   max_tokens=max_tokens, **llm_kw)
    except Exception:
        return ""
    return " ".join(out.split())


def queries(q: str, expansion: str | None, mode: str = "bm25") -> tuple[str, str, str]:
    """(vector-search query, keyword query, re-rank query) for an expansion and mode."""
    if not expansion:
        return q, q, q
    if mode not in MODES:
        raise ValueError(f"expand mode must be one of {MODES}")
    both = q + "\n" + expansion
    if mode == "bm25":
        return q, both, q
    if mode == "append":
        return both, both, both
    return expansion, expansion, expansion


_DONE = re.compile(r"^\W*done\W*$", re.I)


def parse_follow_up(reply: str, question: str) -> str | None:
    """The query from the LLM's reply; None for DONE, empty, or a repeat of the question."""
    line = next((x.strip() for x in (reply or "").splitlines() if x.strip()), "")
    line = re.sub(r"^(search\s+)?query\s*:\s*", "", line, flags=re.I).strip().strip('"').strip()
    if not line or _DONE.match(line) or line.lower().startswith("done"):
        return None
    if re.match(r"^\W*(answer|evidence)\s*:", line, re.I) or "insufficient" in line.lower():
        return None                                   # it answered instead of asking: nothing to search for
    if " ".join(line.lower().split()) == " ".join(question.lower().split()):
        return None
    return line[:300]


def follow_up(question: str, passages: list[str], max_tokens: int = 48, **llm_kw) -> str | None:
    from .llm import chat
    shown = "\n".join(f"[{n}] {' '.join(p.split())[:600]}" for n, p in enumerate(passages, 1))
    try:
        reply = chat([{"role": "user", "content": FOLLOW_UP_PROMPT.format(q=question, passages=shown)}],
                     temperature=0.0, max_tokens=max_tokens, **llm_kw)
    except Exception:
        return None
    return parse_follow_up(reply, question)
