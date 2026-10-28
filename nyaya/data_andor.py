"""Generator for AND-of-OR compositional retrieval data (faceted product search).

Each document is a product with exactly one value per category.
Each query picks `width` categories and allows a set of values in each:
    relevant(doc)  <=>  for every chosen category, doc's value is in the allowed set
This is the Minsky-Papert (AND of ORs) relevance pattern studied by the ANDOR
benchmark (Agarwal et al., 2026). Values have several surface forms, and queries
are rendered with a different form than documents where possible, so pure
keyword overlap is not enough.

Hard negatives satisfy every clause but one.
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass, field

VOCAB: dict[str, list[list[str]]] = {
    "color": [["black", "jet black"], ["white", "off white"], ["navy", "dark blue"], ["red", "scarlet"],
              ["green", "olive green"], ["brown", "chocolate"], ["grey", "charcoal"], ["beige", "sand"],
              ["purple", "violet"], ["yellow", "mustard"], ["pink", "blush"], ["orange", "rust"]],
    "material": [["cotton", "pure cotton"], ["linen", "flax linen"], ["wool", "merino"], ["silk", "mulberry silk"],
                 ["denim", "jean fabric"], ["polyester", "poly blend"], ["leather", "genuine leather"],
                 ["velvet", "plush velvet"], ["nylon", "ripstop nylon"], ["rayon", "viscose"],
                 ["cashmere", "fine cashmere"], ["khadi", "handspun khadi"]],
    "occasion": [["office", "workwear"], ["wedding", "wedding guest"], ["party", "night out"], ["gym", "workout"],
                 ["beach", "seaside"], ["travel", "on the go"], ["festival", "festive"], ["hiking", "trekking"],
                 ["lounge", "at home"], ["date night", "dinner date"], ["interview", "formal meeting"],
                 ["campus", "college"]],
    "season": [["summer", "hot weather"], ["winter", "cold weather"], ["monsoon", "rainy season"],
               ["spring", "springtime"], ["autumn", "fall season"], ["all season", "year round"]],
    "fit": [["slim fit", "tailored"], ["regular fit", "classic fit"], ["relaxed fit", "loose"],
            ["oversized", "boxy"], ["skinny", "body hugging"], ["straight fit", "straight cut"]],
    "pattern": [["solid", "plain"], ["striped", "stripes"], ["checked", "plaid"], ["floral", "flower print"],
                ["polka dot", "dotted"], ["paisley", "mango print"], ["geometric", "abstract print"],
                ["camouflage", "camo"], ["embroidered", "threadwork"], ["tie dye", "dip dyed"]],
    "sleeve": [["sleeveless", "no sleeves"], ["short sleeve", "half sleeve"], ["long sleeve", "full sleeve"],
               ["three quarter sleeve", "elbow length sleeve"], ["cap sleeve", "tiny sleeve"]],
    "neckline": [["round neck", "crew neck"], ["v neck", "vee neckline"], ["collared", "shirt collar"],
                 ["mandarin collar", "band collar"], ["boat neck", "bateau neckline"], ["hooded", "with hood"],
                 ["turtle neck", "high neck"]],
    "origin": [["made in india", "indian made"], ["made in italy", "italian made"],
               ["made in japan", "japanese made"], ["made in portugal", "portuguese made"],
               ["made in vietnam", "vietnamese made"], ["made in turkey", "turkish made"]],
    "care": [["machine wash", "washing machine safe"], ["hand wash", "wash by hand"],
             ["dry clean", "dry clean only"], ["cold wash", "cold water wash"]],
    "closure": [["buttons", "button front"], ["zip", "zipper"], ["pull on", "slip on"],
                ["drawstring", "tie cord"], ["hook and eye", "hook closure"], ["snap", "press stud"]],
    "length": [["cropped", "crop length"], ["hip length", "regular length"], ["knee length", "to the knee"],
               ["ankle length", "full length"], ["mid thigh", "short length"]],
    "weight": [["lightweight", "featherlight"], ["midweight", "medium weight"], ["heavyweight", "thick"]],
    "style": [["minimal", "clean look"], ["vintage", "retro"], ["streetwear", "urban"],
              ["bohemian", "boho"], ["preppy", "ivy league"], ["ethnic", "traditional"],
              ["sporty", "athleisure"], ["formal", "dressy"]],
    "audience": [["men", "mens"], ["women", "womens"], ["unisex", "gender neutral"],
                 ["boys", "kids boys"], ["girls", "kids girls"]],
    "eco": [["organic", "organically grown"], ["recycled", "upcycled fibre"],
            ["vegan", "animal free"], ["conventional", "standard production"]],
}

CATEGORY_PHRASE = {
    "color": "colour", "material": "fabric", "occasion": "occasion", "season": "season",
    "fit": "fit", "pattern": "pattern", "sleeve": "sleeves", "neckline": "neckline",
    "origin": "origin", "care": "care", "closure": "closure", "length": "length",
    "weight": "weight", "style": "style", "audience": "for", "eco": "production",
}

BRAND_A = ["cedar", "moss", "velvet", "harbor", "indigo", "monsoon", "saffron", "granite", "lotus", "tide"]
BRAND_B = ["and mill", "arc", "studio", "works", "lane", "atelier", "co", "house", "line", "thread"]
ITEMS = ["shirt", "kurta", "dress", "jacket", "top", "trousers", "hoodie", "blazer", "skirt", "tee"]

DOC_TEMPLATES = [
    "{brand} {item}. {facts}",
    "{item} by {brand}: {facts}",
    "New from {brand}: a {item}. {facts}",
]
CLAUSE_TEMPLATES = [
    "{cat} must be {vals}",
    "{vals} for {cat}",
    "in terms of {cat}: {vals}",
    "any of {vals} ({cat})",
]
QUERY_TEMPLATES = [
    "I need a product where {clauses}. Every requirement is mandatory.",
    "Find items: {clauses}.",
    "Looking for something with {clauses}; all conditions apply.",
]


@dataclass
class Doc:
    doc_id: str
    values: dict[str, int]
    text: str


@dataclass
class Query:
    query_id: str
    clauses: dict[str, list[int]]
    text: str
    positives: list[str] = field(default_factory=list)
    hard_negatives: list[str] = field(default_factory=list)


def _or(words: list[str]) -> str:
    return words[0] if len(words) == 1 else ", ".join(words[:-1]) + " or " + words[-1]


class AndOrWorld:
    def __init__(self, seed: int = 0, categories: list[str] | None = None, paraphrase: bool = True):
        self.rng = random.Random(seed)
        self.cats = categories or list(VOCAB)
        self.paraphrase = paraphrase

    def _form(self, cat: str, v: int, for_query: bool) -> str:
        forms = VOCAB[cat][v]
        if not self.paraphrase:
            return forms[0]
        # documents mostly use the first form, queries mostly use the second
        if for_query:
            return forms[-1] if self.rng.random() < 0.7 else forms[0]
        return forms[0] if self.rng.random() < 0.7 else forms[-1]

    def render_doc(self, doc_id: str, values: dict[str, int]) -> Doc:
        facts = [f"{CATEGORY_PHRASE[c]}: {self._form(c, v, False)}" for c, v in values.items()]
        self.rng.shuffle(facts)
        tmpl = self.rng.choice(DOC_TEMPLATES)
        brand = f"{self.rng.choice(BRAND_A)} {self.rng.choice(BRAND_B)}"
        text = tmpl.format(brand=brand, item=self.rng.choice(ITEMS), facts="; ".join(facts) + ".")
        return Doc(doc_id, dict(values), text)

    def random_values(self) -> dict[str, int]:
        return {c: self.rng.randrange(len(VOCAB[c])) for c in self.cats}

    def sample_query_spec(self, width: int, options: tuple[int, int] = (2, 4)) -> dict[str, list[int]]:
        cats = self.rng.sample(self.cats, width)
        spec = {}
        for c in cats:
            n = len(VOCAB[c])
            k = min(self.rng.randint(*options), n - 1)
            spec[c] = sorted(self.rng.sample(range(n), k))
        return spec

    def render_query(self, query_id: str, spec: dict[str, list[int]]) -> str:
        parts = []
        for c, vals in spec.items():
            words = [self._form(c, v, True) for v in vals]
            parts.append(self.rng.choice(CLAUSE_TEMPLATES).format(cat=CATEGORY_PHRASE[c], vals=_or(words)))
        self.rng.shuffle(parts)
        return self.rng.choice(QUERY_TEMPLATES).format(clauses="; ".join(parts))

    @staticmethod
    def satisfies(values: dict[str, int], spec: dict[str, list[int]]) -> bool:
        return all(values[c] in allowed for c, allowed in spec.items())

    def make_split(self, n_docs: int, n_queries: int, widths: list[int], n_pos: int = 5,
                   n_hard: int = 10, prefix: str = "") -> tuple[list[Doc], list[Query]]:
        """Corpus with planted positives and one-clause-off hard negatives per query."""
        docs: list[Doc] = []
        specs = []
        for qi in range(n_queries):
            w = widths[qi % len(widths)]
            spec = self.sample_query_spec(w)
            specs.append(spec)
            for _ in range(n_pos):
                vals = self.random_values()
                for c, allowed in spec.items():
                    vals[c] = self.rng.choice(allowed)
                docs.append(self.render_doc(f"{prefix}d{len(docs)}", vals))
            for _ in range(n_hard):
                vals = self.random_values()
                for c, allowed in spec.items():
                    vals[c] = self.rng.choice(allowed)
                broken = self.rng.choice(list(spec))
                vals[broken] = self.rng.choice([v for v in range(len(VOCAB[broken])) if v not in spec[broken]])
                docs.append(self.render_doc(f"{prefix}d{len(docs)}", vals))
        while len(docs) < n_docs:
            docs.append(self.render_doc(f"{prefix}d{len(docs)}", self.random_values()))
        # relevance is computed exactly against the full corpus
        queries = []
        for qi, spec in enumerate(specs):
            q = Query(f"{prefix}q{qi}", spec, self.render_query(f"{prefix}q{qi}", spec))
            for d in docs:
                if self.satisfies(d.values, spec):
                    q.positives.append(d.doc_id)
                elif sum(d.values[c] not in a for c, a in spec.items()) == 1:
                    q.hard_negatives.append(d.doc_id)
            queries.append(q)
        return docs, queries


def save_beir(docs: list[Doc], queries: list[Query], out_dir: str) -> None:
    """Write corpus.jsonl / queries.jsonl / qrels.tsv (BEIR-style) for reuse."""
    import os
    os.makedirs(out_dir, exist_ok=True)
    with open(f"{out_dir}/corpus.jsonl", "w") as f:
        for d in docs:
            f.write(json.dumps({"_id": d.doc_id, "text": d.text, "values": d.values}) + "\n")
    with open(f"{out_dir}/queries.jsonl", "w") as f:
        for q in queries:
            f.write(json.dumps({"_id": q.query_id, "text": q.text, "width": len(q.clauses),
                                "clauses": q.clauses, "hard_negatives": q.hard_negatives[:50]}) + "\n")
    with open(f"{out_dir}/qrels.tsv", "w") as f:
        f.write("query-id\tcorpus-id\tscore\n")
        for q in queries:
            for p in q.positives:
                f.write(f"{q.query_id}\t{p}\t1\n")


if __name__ == "__main__":
    w = AndOrWorld(seed=1)
    docs, qs = w.make_split(n_docs=200, n_queries=3, widths=[2, 3, 4])
    print(docs[0].text)
    for q in qs:
        print(len(q.clauses), q.text, "| positives:", len(q.positives), "| hard:", len(q.hard_negatives))
