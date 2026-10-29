"""Split a free-form query into conditions.

Rule-based splitting (nyaya.model.split_clauses) handles numbered lists, semicolons, new lines and
sentences. Free-form queries such as "red cotton kurta under 2000 for a wedding" need a language
model. `llm_conditions` asks any OpenAI-compatible endpoint (a local server or a hosted one) for
the conditions, and only accepts an answer whose conditions are taken from the query itself:
every content word of every condition must appear in the query, so the planner cannot add
requirements the user never asked for. Anything else falls back to the rule-based split.

Environment: NYAYA_LLM_BASE_URL, NYAYA_LLM_MODEL, NYAYA_LLM_API_KEY (see nyaya.rag).
"""
from __future__ import annotations

import functools
import json
import re

from .model import split_clauses

PROMPT = """Split the search request into its separate conditions.
Return only a JSON list of strings. Copy each condition's words from the request; do not add,
infer or rephrase anything. Keep a condition's alternatives together ("linen or cotton").
Request: {query}"""

_WORD = re.compile(r"[a-z0-9]+")


def _words(t):
    return set(_WORD.findall(t.lower()))


def valid(query: str, conditions) -> bool:
    if not isinstance(conditions, list) or not 1 <= len(conditions) <= 12:
        return False
    qw = _words(query)
    return all(isinstance(c, str) and c.strip() and _words(c) <= qw for c in conditions)


def rule_conditions(query: str) -> list[str]:
    parts = split_clauses(query)
    if len(parts) == 1 and parts[0] == query:
        parts = split_clauses(query, mode="sentences")
    return parts


@functools.lru_cache(maxsize=4096)
def _ask(query: str) -> tuple[str, ...] | None:
    from .rag import chat                                   # https / localhost checks live there
    try:
        text = chat([{"role": "user", "content": PROMPT.format(query=query)}], temperature=0.0)
        m = re.search(r"\[.*\]", text, re.S)
        conds = json.loads(m.group(0)) if m else None
    except Exception:
        return None
    conds = [c.strip() for c in conds] if isinstance(conds, list) and all(isinstance(c, str) for c in conds) else conds
    return tuple(conds) if valid(query, conds) else None


def llm_conditions(query: str) -> list[str]:
    """Conditions from the LLM if it answers with a valid split, otherwise the rule-based split."""
    conds = _ask(query)
    return list(conds) if conds else rule_conditions(query)
