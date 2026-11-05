"""The Porter stemmer (M.F. Porter, 1980, "An algorithm for suffix stripping"), as used by Lucene's
English analyzer, so BM25 can match "founded" with "founding" and "cities" with "city"."""
from __future__ import annotations

from functools import lru_cache

# Lucene's English stopword set (EnglishAnalyzer.ENGLISH_STOP_WORDS_SET)
STOPWORDS = frozenset("a an and are as at be but by for if in into is it no not of on or such that the their then "
                      "there these they this to was will with".split())

_V = set("aeiou")


def _cons(w, i):
    if w[i] in _V:
        return False
    if w[i] == "y":
        return i == 0 or not _cons(w, i - 1)
    return True


def _m(stem):
    """Number of vowel-consonant sequences: [C](VC)^m[V]."""
    n, i, L = 0, 0, len(stem)
    while i < L and _cons(stem, i):
        i += 1
    while i < L:
        while i < L and not _cons(stem, i):
            i += 1
        if i >= L:
            break
        while i < L and _cons(stem, i):
            i += 1
        n += 1
    return n


def _has_vowel(stem):
    return any(not _cons(stem, i) for i in range(len(stem)))


def _double_c(w):
    return len(w) >= 2 and w[-1] == w[-2] and _cons(w, len(w) - 1)


def _cvc(w):
    if len(w) < 3 or not (_cons(w, len(w) - 3) and not _cons(w, len(w) - 2) and _cons(w, len(w) - 1)):
        return False
    return w[-1] not in "wxy"


def _replace(w, suffix, repl, cond):
    if w.endswith(suffix):
        stem = w[: len(w) - len(suffix)]
        if cond(stem):
            return stem + repl, True
        return w, True                                   # suffix matched: stop trying others in this step
    return w, False


@lru_cache(maxsize=200_000)
def stem(w: str) -> str:
    if len(w) <= 2 or not w.isalpha():
        return w
    # step 1a
    if w.endswith("sses"):
        w = w[:-2]
    elif w.endswith("ies"):
        w = w[:-2]
    elif w.endswith("ss"):
        pass
    elif w.endswith("s"):
        w = w[:-1]
    # step 1b
    extra = False
    if w.endswith("eed"):
        if _m(w[:-3]) > 0:
            w = w[:-1]
    else:
        for suf in ("ed", "ing"):
            if w.endswith(suf) and _has_vowel(w[: -len(suf)]):
                w, extra = w[: -len(suf)], True
                break
    if extra:
        if w.endswith(("at", "bl", "iz")):
            w += "e"
        elif _double_c(w) and w[-1] not in "lsz":
            w = w[:-1]
        elif _m(w) == 1 and _cvc(w):
            w += "e"
    # step 1c
    if w.endswith("y") and _has_vowel(w[:-1]):
        w = w[:-1] + "i"
    # step 2
    for suf, rep in (("ational", "ate"), ("tional", "tion"), ("enci", "ence"), ("anci", "ance"), ("izer", "ize"),
                     ("bli", "ble"), ("alli", "al"), ("entli", "ent"), ("eli", "e"), ("ousli", "ous"),
                     ("ization", "ize"), ("ation", "ate"), ("ator", "ate"), ("alism", "al"), ("iveness", "ive"),
                     ("fulness", "ful"), ("ousness", "ous"), ("aliti", "al"), ("iviti", "ive"), ("biliti", "ble"),
                     ("logi", "log")):
        w, hit = _replace(w, suf, rep, lambda s: _m(s) > 0)
        if hit:
            break
    # step 3
    for suf, rep in (("icate", "ic"), ("ative", ""), ("alize", "al"), ("iciti", "ic"), ("ical", "ic"), ("ful", ""),
                     ("ness", "")):
        w, hit = _replace(w, suf, rep, lambda s: _m(s) > 0)
        if hit:
            break
    # step 4
    for suf in ("al", "ance", "ence", "er", "ic", "able", "ible", "ant", "ement", "ment", "ent", "ion", "ou", "ism",
                "ate", "iti", "ous", "ive", "ize"):
        if w.endswith(suf):
            s = w[: -len(suf)]
            if suf == "ion":
                if _m(s) > 1 and s and s[-1] in "st":
                    w = s
            elif _m(s) > 1:
                w = s
            break
    # step 5a
    if w.endswith("e"):
        s = w[:-1]
        if _m(s) > 1 or (_m(s) == 1 and not _cvc(s)):
            w = s
    # step 5b
    if _m(w) > 1 and _double_c(w) and w.endswith("l"):
        w = w[:-1]
    return w
