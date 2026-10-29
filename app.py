"""Interactive demo (Gradio). Runs locally or as a Hugging Face Space.

  pip install "axon-retrieval[demo]"
  python app.py                         # AXON_MODEL selects the ColBERT model (default lightonai/LateOn)

Type a query with several conditions and a few candidate documents. The demo shows the plain
ColBERT ranking next to Axon's, and for every Axon result which passage matched each condition.
"""
import html
import os

import gradio as gr

from nyaya.axon import Axon

MODEL = os.environ.get("AXON_MODEL", "lightonai/LateOn")
axon = Axon(MODEL)                      # one model; the method is chosen per call

EXAMPLE_QUERY = "Find a jacket that: 1. is waterproof, 2. is lightweight, 3. is made in India, 4. has a hood"
EXAMPLE_DOCS = """Trail shell jacket, fully waterproof with taped seams and an adjustable hood. Lightweight and packable. Made in India.

Waterproof parka with a detachable hood, very warm and heavy. Made in India.

Lightweight windbreaker with a hood, water resistant only, not waterproof. Made in India.

Lightweight waterproof rain jacket with hood, manufactured in Vietnam.

Light cotton hoodie made in India. Not suitable for rain."""


def _bar(x):
    return f"{max(0.0, min(1.0, x)) * 100:.0f}%"


def run(query, docs_text):
    docs = [d.strip() for d in docs_text.split("\n\n") if d.strip()][:50]
    if not query.strip() or not docs:
        return "Enter a query and at least one document.", ""
    index = axon.index(docs)
    p = axon.rerank(query, index, method="colbert")
    a = axon.rerank(query, index, method="grouped")
    left = "\n".join(f"{i + 1}. {html.escape(r.text[:120])}" for i, r in enumerate(p))
    right = []
    for i, r in enumerate(a):
        right.append(f"**{i + 1}. {html.escape(r.text[:120])}**")
        for j, c in enumerate(r.conditions):
            ev = "; ".join(f"“{html.escape(e[2])}”" for e in c.evidence) or "no clear passage"
            flag = " ← weakest" if j == r.weakest else ""
            right.append(f"- {html.escape(c.condition)}: {_bar(c.relative)}{flag} — {ev}")
        right.append("")
    return left, "\n".join(right)


with gr.Blocks(title="Axon") as demo:
    gr.Markdown("# Axon\nCondition-aware re-ranking with evidence for every condition.")
    with gr.Row():
        q = gr.Textbox(label="Query (number the conditions or separate them with ;)", value=EXAMPLE_QUERY)
    d = gr.Textbox(label="Candidate documents (blank line between documents)", value=EXAMPLE_DOCS, lines=12)
    go = gr.Button("Rank")
    with gr.Row():
        out_plain = gr.Markdown(label="ColBERT")
        out_axon = gr.Markdown(label="Axon")
    go.click(run, [q, d], [out_plain, out_axon])

if __name__ == "__main__":
    demo.launch()
