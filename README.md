# Axon: condition-aware late-interaction retrieval

*Package name: `nyaya` (Sanskrit: logic, reasoning).*

Late-interaction retrievers such as ColBERT score a query by averaging, over all query tokens, each token's best match in the document. For a query with many conditions ("a book by a war reporter, a collection of essays, critical of military intervention, …") that average lets a document that matches most conditions very well outrank one that matches all of them.

Axon splits the query into its conditions and scores each condition separately. Each condition is encoded on its own, its tokens' MaxSim scores are averaged, and the conditions are averaged with equal weight. The model's weights are not changed: it is a scoring change that plugs into any released ColBERT-style model.

**Main result.** On [MultiConIR](https://github.com/EIT-NLP/MultiConIR), a benchmark of real books, movies, people, medical cases and legal documents with queries of 1–10 conditions, condition-aware scoring improves three released late-interaction models. Every 95% interval excludes zero:

| Model | ColBERT scoring | Nyaya scoring | Δ (95% interval) | Δ on 10-condition queries |
| --- | --- | --- | --- | --- |
| [LateOn](https://huggingface.co/lightonai/LateOn) | 0.691 | **0.737** | +0.045 [+0.039, +0.051] | +0.062 [+0.043, +0.079] |
| [GTE-ModernColBERT-v1](https://huggingface.co/lightonai/GTE-ModernColBERT-v1) | 0.712 | **0.763** | +0.050 [+0.045, +0.055] | +0.067 [+0.051, +0.084] |
| [mLateOn](https://huggingface.co/lightonai/mLateOn) | 0.813 | **0.847** | +0.031 [+0.026, +0.036] | +0.112 [+0.092, +0.132] |

(MultiConIR Task 1 win rate: how often the relevant document outranks a hard negative that breaks some conditions, averaged over queries with 1–10 conditions. Intervals are paired bootstrap over 2,307 documents.)

**What the gain is made of.** A control run later (round 4) changes how the table should be read. Plain ColBERT with only the query's lead-in removed ("Find a book that:") recovers much of the T1 gain:
- **LateOn:** all of it (+0.046 vs +0.045).
- **GTE-ModernColBERT:** half (+0.025 vs +0.050).
- **mLateOn:** most of it (+0.027 vs +0.031).

The part that only condition-aware scoring adds shows up in two places:
- **Descriptive, natural-language queries (T3):** +0.020 / +0.077 / +0.091 over lead-in-removed ColBERT on the three models, every interval above zero.
- **10-condition queries:** +0.033 on GTE-ModernColBERT and +0.036 on mLateOn.

On LateOn, the benefit comes from encoding each condition on its own, not from weighting conditions equally; equal weighting is very slightly worse. Two practical lessons follow:
- Strip boilerplate from multi-condition queries before scoring them with ColBERT.
- Split descriptive queries into their conditions.

**Main negative results.**
- **Competitive MaxSim** makes conditions compete for each document token so that one piece of evidence cannot satisfy two conditions. It is worse than plain grouping on every model: T1 −0.077, −0.121 and −0.106, every interval below zero. The softer the competition, the smaller the loss, so on this benchmark sharing evidence between conditions helps rather than hurts.
- **A fuzzy AND** (soft minimum) is much better than averaging on clean synthetic data, but worse on real text, for all three models, with or without calibration.

Details below.

## Contents

- [Use it](#use-it)
- [The full RAG system](#the-full-rag-system)
- [How it works](#how-it-works)
- [Results](#results)
- [What did not work](#what-did-not-work)
- [Relation to prior work](#relation-to-prior-work)
- [Scope and limitations](#scope-and-limitations)
- [Reproduce](#reproduce)
- [Repository layout](#repository-layout)
- [Security notes](#security-notes)
- [Experiment log](#experiment-log)

## Use it

```bash
pip install "git+https://github.com/Tejascodz/Axon"            # add [server] or [demo] for those extras
```

```python
from nyaya.axon import Axon

ax = Axon("lightonai/LateOn")                    # any PyLate ColBERT checkpoint; loads once
index = ax.index(documents)                      # encode candidates once, reuse for many queries
for r in ax.rerank("Find a jacket that: 1. is waterproof, 2. is lightweight, 3. has a hood", index, top_k=5):
    print(f"{r.score:.3f}  {r.text[:60]}")
    for c in r.conditions:                       # evidence certificate: one entry per condition
        print(f"    {c.condition}: {c.relative:.0%} of a perfect match, evidence {[e[2] for e in c.evidence]}")
```

- **Simple queries never change.** A query with one condition is scored exactly like the underlying ColBERT model.
- **Conditions** come from numbered lists, semicolons, new lines or sentences. Pass `conditions=[...]` yourself, or use `nyaya.planner.llm_conditions(query)`. It asks any OpenAI-compatible model to split a free-form query and rejects any split that adds words the user never typed.
- **Evidence certificate.** Every result lists, per condition:
  - its score
  - its score relative to the condition matched against its own text
  - the document passages it matched, taken from ColBERT's own token alignments
  - `weakest`, naming the condition the document satisfies least
- **Methods.** `method="grouped"` is the default; `"colbert"` is the baseline. Competitive MaxSim is not offered, because it lost in round 4. Its scoring code stays in `nyaya/scoring.py` so that result can be reproduced.
- **How far to trust the certificate.**
  - When a document can be compared with one known to satisfy every condition, the condition whose score dropped most is the broken one 54–82% of the time (chance 10%).
  - From a single document alone, the `weakest` flag is right only 24–38% of the time, about as often as simple word overlap (26%). Treat it as a hint, not a verdict.
- **HTTP service:** `AXON_MODEL=lightonai/LateOn uvicorn nyaya.server:app`, then `POST /rerank`, or `POST /ask` for full answers. It has optional API-key auth (`AXON_API_KEY`), a per-client rate limit, request size caps and Prometheus metrics at `/metrics`, and it binds to localhost.
- **Demo:** `python app.py` shows ColBERT and Axon side by side with evidence. It runs as a Hugging Face Space.

## The full RAG system

Every stage is written in this repository; the only outside pieces are the ColBERT model, the language model and the server that runs it.

| Stage | What it does | Where |
|---|---|---|
| Ingestion | .txt, .md and .pdf files. Size and page limits are enforced, a PDF must carry a real PDF signature, symbolic links are not followed, and unreadable files are skipped and reported | `nyaya/rag.py`, `check_file` |
| Chunking | Paragraph-first, 220-word chunks with a 40-word overlap | `nyaya/rag.py` |
| Indexing | ColBERT token vectors for every chunk (no pickle; tensors only), BM25, and an optional PLAID index for approximate search over large collections | `nyaya/rag.py`, `build_axon` |
| Retrieval | BM25 and ColBERT search fused with reciprocal-rank fusion, then Axon re-ranking with per-condition evidence | `AxonRAG.retrieve` |
| Tenant isolation | In a multi-tenant index (one folder per tenant), the tenant filter is applied inside both searches, before fusion and re-ranking, and checked again on every result. A query without a tenant is refused. On the server, each tenant has its own API key | `AxonRAG.tenant_mask`, `nyaya/server.py` |
| Guardrails, input | Retrieved text is escaped, stripped of invisible characters and marked as data, never instructions. A scanner withholds passages that look like instructions aimed at the model, including system-prompt and key exfiltration attempts. It first normalises Unicode, so zero-width characters and look-alike letters do not hide an instruction | `nyaya/guards.py` |
| Generation | Any OpenAI-compatible server: vLLM, SGLang (both benchmarked in round 5), a hosted API, or `nyaya.hf_server` (one request at a time, the baseline). Answers stream; each must cite its sources or reply INSUFFICIENT. For multi-condition questions, Axon's evidence certificate is passed to the model | `nyaya/llm.py`, `AxonRAG.answer` |
| Guardrails, output | Citations to sources that were never shown are removed; sentences without a citation are reported | `guards.check_answer` |
| Monitoring | Time to first token, inter-token latency, stage timings and guardrail events per answer, in Prometheus format at `/metrics` | `nyaya/metrics.py` |
| Load testing | Concurrency sweep with TTFT, inter-token latency, time per output token, throughput, GPU memory and utilisation, plus the server's own metrics | `nyaya/llm_bench.py` |
| Fine-tuning | LoRA written out in full (no PEFT), fp16 with loss scaling for T4s, loss on the reply only, merged back into the weights so any server can load the result | `nyaya/lora.py` |
| Evaluation | It measures:<ul><li>retrieval: Recall@5/10/20, MRR@10, nDCG@10, p50/p95/p99 latency</li><li>answers on HotpotQA: exact match, F1</li><li>citations: precision and recall</li><li>grounding: is the answer in the sources it cites, checked without an LLM judge</li><li>abstention when the evidence is withheld</li><li>attack success for 8 injection styles</li><li>cross-tenant leakage</li><li>the MultiConIR condition check</li></ul> | `nyaya/rag_eval.py`, `scripts/round5.py` |

```bash
python -m nyaya.rag build --src docs/ --out index/ --model lightonai/LateOn --plaid
vllm serve Qwen/Qwen3-4B-Instruct-2507 --dtype half --max-model-len 4096 &     # or any OpenAI-compatible server
export NYAYA_LLM_BASE_URL=http://localhost:8000/v1 NYAYA_LLM_MODEL=Qwen/Qwen3-4B-Instruct-2507
python -m nyaya.rag ask --index index/ "linen or cotton shirt; for summer; made in India"
```

### Measured (round 5)

The full system was measured on Kaggle T4 GPUs with Qwen3-4B-Instruct-2507 as the generator and LateOn as the retriever. Questions, test sets and pass/fail rules were fixed in `scripts/round5.py` before running. Intervals are 95% bootstrap intervals. Full tables are in `results/ROUND5.md`, and exact package versions and model revisions in `results/r5_environment.json` and `results/r5_pip_freeze*.txt`.

**Retrieval.** 700 HotpotQA questions over a corpus of 6,852 paragraphs; gold = the two supporting paragraphs.

| | BM25 + ColBERT, fused | + Axon re-rank |
|---|---|---|
| both gold paragraphs in the top 5 | 0.699 | **0.824** |
| Recall@10 | 0.931 | **0.960** |
| MRR@10 | 0.919 | **0.964** |
| nDCG@10 | 0.845 | **0.910** |

- The full pipeline takes 63 ms median per query, 78 ms p95, 123 ms p99.
- HotpotQA questions have a single condition, so the re-ranking gain here is ColBERT's, not condition grouping's.
- On this corpus PLAID matches exact search (recall@100 0.993 vs 0.992) but is slower (65 vs 32 ms); approximate search pays off only on much larger collections.

**Serving.** T4, fp16, about 850-token RAG prompts and fixed 128-token outputs. Every request uses a prompt not sent before in that benchmark. vLLM and SGLang were measured back to back on the same GPU, model and prompts.

| Engine | 1 user: TTFT p50 | 1 user: time per output token | 16 users: output tokens/s | 16 users: ITL p99 | 32 users: output tokens/s |
|---|---|---|---|---|---|
| transformers, one request at a time | 854 ms | 65 ms | 14 (4 users) | – | – |
| vLLM 0.33.0 | 556 ms | 39 ms | 105 | 1,459 ms | 113 |
| SGLang 0.5.23 | **306 ms** | **37 ms** | **193** | **107 ms** | **226** |

- **SGLang led at every level here, about 2× vLLM's throughput** under load, and inter-token latency stayed stable where vLLM's p99 rose to about 1.5 s.
- **This comparison is specific to T4s (compute capability 7.5).** vLLM 0.33 falls back to its Triton attention kernels, because FlashAttention 2 needs compute capability 8.0. SGLang compiles FlashInfer kernels for the T4 on first start (about 4 minutes). The result may well differ on newer GPUs.
- **The T4 ran near its 70 W power limit** under both engines (66–68 W on average). SGLang ran at lower clocks (671–972 MHz versus 1,101–1,350 MHz for vLLM) and was still faster.
- **The fine-tuned model serves at the same speed as the base model** (39.6 vs 39.1 ms per token). A slower figure in the first run (59.5 ms) did not reproduce once GPU clock and temperature were recorded.
- **The transformers baseline was measured while the other GPU was fine-tuning.**

**Answers** (500 answerable questions, plus 200 with the gold paragraphs withheld)

| | Base model | Fine-tuned (LoRA) | Difference (paired) |
|---|---|---|---|
| Exact match | 0.324 | **0.506** | +18.2 [+14.4, +22.2] |
| F1 | 0.420 | **0.627** | +20.8 [+17.2, +24.5] |
| Citation precision | 0.932 | 0.913 | −1.8 [−4.0, +0.4] |
| Citation recall | 0.481 | **0.808** | +32.7 [+28.7, +36.5] |
| Answer found in the sources it cites | 0.885 | **0.972** | +9.3 [+5.8, +13.1] |
| Wrong refusals (answerable) | 0.436 | **0.160** | −27.6 [−31.8, −23.4] |
| Correct refusals (gold withheld) | **0.950** | 0.915 | −3.5 [−7.0, +0.0] |

- The base model refuses too often: 37% of the time even when both answer paragraphs are in front of it.
- Fine-tuning (1,600 examples, 83 minutes on one T4, 33 M trainable parameters) fixes most of that, at a small, borderline cost in correct refusals.
- No answer from either model lacked a valid citation. Of the 700 graded answers, one base-model reply reached the token limit and no fine-tuned reply did. Across every test, 10 of the base model's 2,500 replies and none of the fine-tuned model's reached it.
- The pre-registered success test for fine-tuning was **met in part**. Citation recall rose, exact match rose rather than fell, and wrong refusals dropped. But citation precision did not improve, and correct refusals fell slightly.

**Prompt injection** (200 questions, 8 attack styles, one per question; attack success rate)

| Prompt | Base model | Fine-tuned |
|---|---|---|
| undefended | 10.5% | 2.5% |
| defended (escaped sources, data-not-instructions rule) | 2.0% | 0.0% |
| defended + input scanner | **0.5%** | **0.0%** |

- **Scanner:** it caught all 150 planted attacks that use its known phrasings, including those hidden with zero-width characters or look-alike letters. It missed the 50 attacks worded to avoid it, and flagged **0 of 6,852** clean paragraphs.
- **Tenant isolation:** 0 cross-tenant results among 2,000 returned, and a query without a tenant was refused.

**Condition check** (600 MultiConIR documents: half satisfy all ten conditions, half break exactly one)

| | LLM alone | LLM + Axon certificate | Difference (paired) |
|---|---|---|---|
| Base model, all documents | 60.8% | **64.2%** | +3.3 [+0.7, +6.2] |
| Base model, documents that break one condition | 25.0% | 29.3% | +4.3 [−1.0, +9.3] |
| Fine-tuned model, all documents | 60.7% | 58.8% | −1.8 [−4.2, +0.5] |

- For the base model, the certificate helps; that was the pre-registered test, and it passed. But the gain is small.
- Naming the broken condition stays hard for every method: the LLM manages 25–29%, Axon alone 50% overall.
- The fine-tuned model leans toward "satisfies everything" (544 of 600 verdicts with the certificate) and gains nothing from it. Training it to refuse less appears to carry over to this task.

**Condition-level abstention.** For a multi-condition question, the generator sees, for every source, how strongly each condition was matched and by which passage. This lets it say which condition no source satisfies instead of answering anyway. The condition check above is the measurement of this idea.

**What this does not show.**
- A Kaggle notebook is a test environment, not a production deployment.
- The results cover one GPU type, one generator, and Wikipedia-style questions.
- The injection scanner is a heuristic; the defences reduce attacks, they don't prevent them.
- The "no citation, no answer" rule (`guards.require_citation`) is now on by default, but it was added after the first round 5 run; in this run no answer needed it.

## How it works

1. **Split the query into conditions** (`nyaya/model.py`, `split_clauses`). Numbered lists ("Find a book that: 1. …, 2. …") give one clause per item, with the lead-in dropped. Semicolons, new lines, "and also" and "plus" also separate clauses. For descriptive prose, sentence splitting can be switched on. An LLM query planner can replace this rule-based splitter.
2. **Encode each condition** with the model's own query encoder, which produces token vectors $q_{c,1..L_c}$ for condition $c$.
3. **Score** a document with token vectors $d_{1..K}$:

$$
s(q, d) = \frac{1}{C}\sum_{c=1}^{C} \frac{1}{L_c} \sum_{t=1}^{L_c} \max_{k} \langle q_{c,t}, d_k \rangle
$$

Standard ColBERT scoring is the same inner MaxSim, averaged over all query tokens at once, so long or wordy conditions get more weight. Documents are encoded and indexed exactly as before. Only the query side and the final aggregation change (`nyaya/scoring.py`, `score_grouped`).

**Competitive MaxSim (tested; it does not work).** In ColBERT, any number of query tokens may use the same document token as their best match, so a document that lacks one condition can still score well on it by reusing evidence that belongs to another condition. Competitive MaxSim shares every document token out among the query's conditions:
- Each token's share is a softmax over conditions of how well it fits each one, with temperature τ.
- A query token may only use its own condition's share of a document token.
- With one condition the share is always 1, so the rule is exactly MaxSim.
- Evidence that serves two conditions is split between them; distinct evidence costs nothing (`tests/test_scoring.py`).
- It needs no training and costs about the same as MaxSim (`nyaya/scoring.py`, `competitive_maxsim`).
- **Result:** with τ fixed in advance at 0.1, it is worse than plain grouping on all three models. The loss shrinks steadily as τ grows: τ 0.03 → −0.111, 0.1 → −0.077, 0.3 → −0.011 on LateOn. As τ grows, the rule approaches plain grouping. On MultiConIR, conditions share vocabulary and evidence for good reasons, so forcing them to split it removes real signal.

## Results

All numbers come from released weights with no training, 128-token queries (nothing truncated) and fp16 inference on Kaggle T4 GPUs.

**MultiConIR, three models** (T1 = win rate against a condition-breaking hard negative; T2 = whether documents that break more conditions are ranked lower; T3 = T1-style test with descriptive, natural-language queries split into sentences):

| Model | Scoring | T1 mean | T1, 5 conditions | T1, 10 conditions | T2 monotonicity | T3 |
| --- | --- | --- | --- | --- | --- | --- |
| LateOn | ColBERT | 0.691 | 0.754 | 0.427 | 0.685 | 0.409 |
| | **Nyaya** | **0.737** | **0.800** | **0.490** | **0.725** | **0.442** |
| GTE-ModernColBERT-v1 | ColBERT | 0.712 | 0.771 | 0.471 | 0.684 | 0.436 |
| | **Nyaya** | **0.763** | **0.811** | **0.540** | **0.733** | **0.505** |
| mLateOn | ColBERT | 0.813 | 0.852 | 0.670 | 0.782 | 0.650 |
| | **Nyaya** | **0.847** | **0.854** | **0.790** | **0.805** | **0.744** |

Paired differences (Nyaya − ColBERT, 95% bootstrap interval over documents):

| Model | T1 mean | T1, 10 conditions | T2 monotonicity | T3 |
| --- | --- | --- | --- | --- |
| LateOn | +0.045 [+0.039, +0.051] | +0.062 [+0.043, +0.079] | +0.039 [+0.032, +0.045] | +0.032 [+0.015, +0.048] |
| GTE-ModernColBERT-v1 | +0.050 [+0.045, +0.055] | +0.067 [+0.051, +0.084] | +0.049 [+0.043, +0.055] | +0.068 [+0.051, +0.085] |
| mLateOn | +0.031 [+0.026, +0.036] | +0.112 [+0.092, +0.132] | +0.022 [+0.017, +0.028] | +0.090 [+0.071, +0.108] |

The gain grows with the number of conditions. It is near zero for single-condition queries, where both scorings are identical, and largest at 10 conditions.

**Where the gain comes from** (round 4; paired differences, 95% intervals; full tables in `results/ROUND4.md`):

| Model | Comparison | T1 mean | T1, 10 conditions | T2 monotonicity | T3 |
| --- | --- | --- | --- | --- | --- |
| LateOn | lead-in removed − ColBERT | +0.046 [+0.041, +0.050] | +0.058 [+0.043, +0.074] | +0.030 [+0.025, +0.036] | +0.011 [+0.001, +0.022] |
| | grouping − lead-in removed | −0.001 [−0.006, +0.004] | +0.003 [−0.013, +0.020] | +0.008 [+0.002, +0.014] | +0.020 [+0.004, +0.036] |
| GTE-ModernColBERT-v1 | lead-in removed − ColBERT | +0.025 [+0.021, +0.029] | +0.035 [+0.023, +0.046] | +0.018 [+0.014, +0.022] | −0.009 [−0.019, +0.002] |
| | grouping − lead-in removed | +0.025 [+0.020, +0.029] | +0.033 [+0.018, +0.048] | +0.031 [+0.026, +0.037] | +0.077 [+0.060, +0.094] |
| mLateOn | lead-in removed − ColBERT | +0.027 [+0.023, +0.031] | +0.076 [+0.060, +0.093] | +0.018 [+0.014, +0.022] | −0.001 [−0.010, +0.008] |
| | grouping − lead-in removed | +0.004 [+0.000, +0.008] | +0.036 [+0.020, +0.051] | +0.004 [−0.001, +0.009] | +0.091 [+0.072, +0.109] |

On LateOn, the two halves of grouping are separated further:
- **Encoding each condition on its own, tokens weighted as in ColBERT,** versus lead-in removed: +0.003 T1 (interval includes zero), +0.012 T2 and +0.029 T3, both above zero.
- **Then weighting conditions equally:** −0.003 T1, −0.004 T2 and −0.008 T3.

So on LateOn the useful part is the separate encoding.

**Which condition does a near-miss break?** On 1,455 documents that break exactly one known condition (chance 0.10):

| Model | Grouping, comparing with the good document | Grouping, near-miss alone | Word overlap, near-miss alone | Competitive MaxSim, comparing / alone |
| --- | --- | --- | --- | --- |
| LateOn | 0.535 [0.510, 0.561] | 0.249 [0.227, 0.271] | 0.256 [0.234, 0.278] | 0.379 / 0.183 |
| GTE-ModernColBERT-v1 | 0.544 [0.519, 0.570] | 0.377 [0.352, 0.402] | 0.256 | 0.337 / 0.258 |
| mLateOn | 0.821 [0.801, 0.840] | 0.241 [0.219, 0.264] | 0.256 | 0.576 / 0.124 |

**Cost.** Re-ranking 100 pre-encoded candidates on a T4, including query encoding, takes a median of:

| Scoring | Median per query |
| --- | --- |
| ColBERT | 29.4 ms |
| Grouping | 27.7 ms |
| Competitive MaxSim | 31.2 ms |

Condition-aware scoring adds no measurable cost.

**Why condition-aware scoring is needed.** On a synthetic AND-of-OR product collection (20,000 documents, hard negatives that break one condition), released LateOn's Recall@10 falls from 0.98 with 2 conditions to 0.11 with 8 conditions, even with the full query visible.

## What did not work

These are reported because they are informative, and because the obvious next ideas are among them.

1. **Fuzzy AND across conditions (soft minimum).** Theory says a minimum over conditions should beat an average for AND-of-OR relevance (Agarwal et al. 2026; `scripts/simulate_margin.py` reproduces the margin argument).
   - **On synthetic data after fine-tuning it is excellent:** 8-condition Recall@10 rises from 0.511 to 0.968.
   - **On MultiConIR it is worse than plain averaging:** LateOn T1 0.706 vs 0.737 (−0.032 [−0.035, −0.028]). Softer temperatures help monotonically, approaching the average.
   - **Calibration closes most of the gap but never wins.** Dividing each condition's score by its score against its own text recovers most of the loss (−0.006 for LateOn), but it never beats plain averaging, and it is slightly worse on monotonicity for all three models (pre-registered test).
   - **Why:** real conditions are paraphrased and partly satisfied, so the weakest condition's score is often noise rather than a missing condition.
2. **Compressing documents into a few learned "facet" vectors** (ConstBERT-style, with competitive slot attention). Trained from scratch on CPU, it scored below a single-vector dense model (Recall@10 0.30 vs 0.47).
3. **Competitive MaxSim** (conditions compete for each document token; see How it works). It is worse than plain grouping on all three models, and fixed in advance as the round 4 test (T1 −0.077 / −0.121 / −0.106). It also localises the broken condition worse.
4. **Naming the broken condition from one document alone.** Each condition's score is divided by its score against its own text, and the weakest is reported. It is no better than word overlap on LateOn (0.249 vs 0.256) or mLateOn (0.241), and better only on GTE-ModernColBERT (0.377).
5. **Fine-tuning on synthetic multi-condition data** (LateOn, 1,500 steps, mixed 50/50 with MS MARCO, 2 seeds). It hurt ordinary search (NanoBEIR nDCG@10 0.602 → 0.49–0.53) and did not beat training-free Nyaya scoring on MultiConIR. The released weights are the better starting point.

## Relation to prior work

| Work | Relation |
| --- | --- |
| ColBERT / ColBERTv2; LateOn, GTE-ModernColBERT, mLateOn (LightOn, 2025–26) | The models and scoring rule this builds on; Nyaya changes only how query-token scores are aggregated. |
| MultiConIR (EMNLP Findings 2025) | The benchmark. Its paper evaluates BM25, dense embedders and rerankers, not late-interaction models. |
| Yin, Lu, Qi, Shen, *When Should Queries Be Decomposed?* (arXiv 2606.08577, 2026) | Closest work. One sub-query per condition with summed scores, for dense embedders and cross-encoder rerankers: it hurts first-stage dense retrieval and helps reranking (at about 2× cost). Nyaya's scoring equals that decomposition applied to a late-interaction model when each sub-query's score is length-normalised (summing raw ColBERT sums would instead weight every token equally), computed in a single pass; this repository is the late-interaction counterpart of that study. |
| LIMIT (Weller et al., ICLR 2026); Agarwal et al., *Retrieval Needs Multivectors* (2026) | Theory and benchmarks showing single vectors fail on combination queries; motivated the AND experiments. |
| Killingback et al. (2026) | MaxSim as an aggregation of soft-ORs; Signed MaxSim for negation. |
| EXCISE (Ali et al., 2026) | Query-side exclusion (NOT) for late interaction; complementary to conditions combined by AND. |
| Slot Attention (Locatello et al., 2020); capsule routing | The mechanism behind Competitive MaxSim: a softmax over the competitors (slots there, conditions here), so they compete for each input. We found no prior application across a query's conditions inside late-interaction scoring. Tested here, it does not help (round 4). |
| AligneR, *Multi-Vector Retrieval as Sparse Alignment* (Qian et al., 2022) | Learned, sparse query-token-to-document-token alignments in multi-vector retrieval; per token and trained, not a competition between conditions. |
| Word Mover's Distance (Kusner et al., 2015); optimal-transport text matching; coverage penalties in translation (Tu et al., 2016) | The same intuition (evidence should not be spent twice) in other settings. |
| Shin et al., *Unbalanced Optimal Transport for Efficient Visual Document Retrieval* (arXiv 2610.23929, 2026) | Optimal transport used to compress document pages before standard MaxSim; query-independent and aimed at efficiency, so unrelated to evidence sharing between conditions. |
| Edy et al., *Working Notes on Late Interaction Dynamics* (ECIR 2026 workshop) | Observes that MaxSim uses only the single best token per query token and leaves a better operator to future work; no multi-condition evaluation. |

## Scope and limitations

- **What is tested is ranking, not first-stage search.** MultiConIR's official tasks compare the relevant document against its own hard negatives, a reranking-style setting. Whether condition-aware scoring also helps first-stage retrieval over a whole collection has not been tested on real data; on the synthetic first-stage test it was slightly below plain LateOn (Recall@10 0.475 vs 0.495). Yin et al. report that decomposition hurts first-stage retrieval for dense models.
- **Much of the templated-query gain is boilerplate removal** (see "What the gain is made of"). Condition-aware scoring adds a clear, separate gain on descriptive queries and on 10-condition queries for two of three models.
- **Numbers and ranges are not understood.** "Under 2 kg" or "after 2010" are matched as words, not compared as quantities, so filter numeric constraints with structured metadata.
- **Query splitting is rule-based** unless the optional LLM planner is used. It works for numbered or semicolon-separated conditions and, with sentence splitting, for descriptive queries; free-form queries need an LLM planner.
- **The guardrails are heuristics.** The injection scanner matches common phrasings. Round 5 measures how many planted attacks it catches and how many clean passages it flags, and some test attacks are worded to avoid it. The defended prompt lowers attack success but does not prevent it.
- **Fine-tuning data are from HotpotQA** (gold and distractor paragraphs from the dataset, not this retriever's output), so the fine-tuned generator is tuned to that kind of question.
- **One run per configuration.** Scoring is deterministic given the weights; the intervals reflect variation over documents, not over training runs.

## Reproduce

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest -q tests
```

**Main result** (training-free, three models; about 2 h on Kaggle T4 x2, internet on):

```bash
python scripts/round2.py            # writes results/ROUND2.md
```

**Everything else**

```bash
python scripts/simulate_margin.py                                   # margin simulation (numpy)
python scripts/cpu_experiment.py --steps 2000 --no_paraphrase       # from-scratch CPU comparison
python scripts/kaggle_run.py --plan quick                           # LateOn: training-free + fine-tuning, all suites
python scripts/eval_suite.py --zero_shot lightonai/LateOn --variant colbert_grouped \
    --query_length 128 --suites multiconir --out results/lateon_grouped.json
```

Datasets (MultiConIR, LIMIT, NanoBEIR, MS MARCO triples) and models are downloaded at run time by `scripts/prep_data.py`; their own licenses apply.

**Round 4** (Competitive MaxSim, localization, lead-in control, latency; about 2–2.5 h) and **round 5** (the full RAG system; about 3 h, plus 45 min for `--rebench`, the serving-only re-measurement), Kaggle T4 x2, internet on:

```bash
python scripts/round4.py            # writes results/ROUND4.md
python scripts/round5.py            # writes results/ROUND5.md
python scripts/round5.py --rebench  # serving benchmarks again from an earlier run's results/
```

Checkpoints trained with `nyaya.train` still work with the RAG commands through `--ckpt`.

## Repository layout

```
nyaya/axon.py         Axon API: index, rerank, per-condition evidence certificates
nyaya/planner.py      condition splitting: rules, or an LLM with a no-invented-requirements check
nyaya/server.py       HTTP service: /rerank, /ask, /metrics; API key, rate limit, size caps
nyaya/rag.py          chunking, indexing (exact or PLAID), hybrid retrieval, guarded generation
nyaya/guards.py       prompt format, prompt-injection scanner, citation checker
nyaya/llm.py          OpenAI-compatible client with streaming and per-request timing
nyaya/llm_bench.py    serving load test: TTFT, inter-token latency, throughput, GPU use
nyaya/metrics.py      Prometheus metrics, no dependencies
nyaya/hf_server.py    one-request-at-a-time transformers server (baseline and fallback)
nyaya/lora.py         LoRA fine-tuning and merging, fp16 on T4
nyaya/rag_eval.py     HotpotQA, injection and condition-check evaluation; fine-tuning data
nyaya/scoring.py      MaxSim, ColBERT mean, soft-min, condition-grouped and (negative result) Competitive MaxSim scoring
nyaya/model.py        backbones (tiny, Hugging Face, PyLate ColBERT checkpoints), clause splitter, retriever
nyaya/benchmarks.py   MultiConIR (official win-rate metrics, paired outputs), NanoBEIR, localization, latency
nyaya/data_andor.py   synthetic AND-of-OR collections with condition-breaking hard negatives
nyaya/evaluate.py     exact late-interaction search, recall / nDCG / MRR, BEIR and LIMIT loaders
nyaya/train.py        contrastive training (CPU or GPU, fp16), all variants
nyaya/bm25.py         BM25
scripts/round2.py     main experiment: three models, paired bootstrap intervals
scripts/round4.py     Competitive MaxSim, violated-condition localization, lead-in control, latency
scripts/round5.py     the full RAG system: vLLM serving, answers, guardrails, fine-tuning (one run)
scripts/r5_steps.py   round 5 steps: retrieve, training data, generate, benchmark
app.py                demo: ColBERT vs Axon with evidence (Gradio)
pyproject.toml        package (pip install), extras for server, demo, evaluation
scripts/kaggle_run.py multi-GPU job queue for unattended runs (training-free + fine-tuning)
scripts/eval_suite.py one evaluation entry point for checkpoints and released models
scripts/prep_data.py  dataset and model downloads
tests/                scoring, evidence, planner, guardrail, metrics, LoRA and server tests
.github/workflows/    CI: lint and tests on CPU
```

## Security notes

- Checkpoints and indexes are loaded with `torch.load(..., weights_only=True)`, so they can hold only tensors and plain data. The BM25 index is rebuilt from text, and no pickled objects are written or read. Keep `torch>=2.6`: earlier versions have a known flaw in `weights_only` loading.
- `NYAYA_LLM_API_KEY` is read from the environment only, is sent only when set, and is refused over plain http to non-local hosts.
- Retrieved text is HTML-escaped inside `<source>` tags before it reaches the LLM, and the system prompt treats sources as data. LLM output is printed, never executed. As with any RAG system, indexed documents can still try to influence the answer, so index only content you trust.
- Retrieved passages that look like instructions to the model are withheld by an input scanner, and citations to sources that were never shown are removed from answers (`nyaya/guards.py`). Round 5 measures attack success with and without these defences.
- The HTTP service supports an API key (`AXON_API_KEY`) or one key per tenant (`AXON_TENANT_KEYS`), compared in constant time. It also has a per-client rate limit and request size caps. Error replies do not expose internals. It binds to localhost by default; put it behind TLS before exposing it.
- In a multi-tenant index, documents are filtered by tenant inside retrieval, never by the prompt. A query without a tenant is refused, and there is no cache that could leak across tenants. Round 5 counts cross-tenant results on the real index.
- Ingestion enforces file-size and PDF-page limits and checks the PDF signature instead of trusting the extension. Symbolic links and linked folders are not followed.
- Fine-tuned weights are saved as safetensors (no pickle).
- The experiment runner quotes every user-supplied value before it reaches the shell, and runs servers and installs without a shell.

## Experiment log

<details>
<summary>Chronological record of every experiment, including dead ends</summary>

**Margin simulation** (no learning; noisy clause scores; recall of 10 relevant documents among 200 hard and 2,000 random negatives):

| noise σ | clauses m | mean (ColBERT) | soft-min |
| --- | --- | --- | --- |
| 0.05 | 8 | 0.934 | 1.000 |
| 0.10 | 4 | 0.645 | 0.982 |
| 0.10 | 8 | 0.513 | 0.981 |
| 0.20 | 8 | 0.191 | 0.428 |

**CPU, tiny encoder from scratch, facet-slot design** (2,000 steps, exact value words, 5,000 documents; Recall@10 / nDCG@10): ColBERT 0.754 / 0.781, dense 0.473 / 0.487, clause-mean over facet slots 0.297 / 0.314, soft AND over facet slots 0.207 / 0.217.

Training lessons from these runs:
- Query and document vectors must share one projection.
- Slot attention collapses without unit-variance slot queries and a diversity penalty.
- A hard minimum from step one trains slowly, so the temperature is annealed from 1.0 to 0.05.

**CPU, condition-grouped token scoring** (same setup), Recall@10 at 2 / 6 / 8 conditions: ColBERT 0.996 / 0.670 / 0.360; grouped 1.000 / 1.000 / 0.976; grouped + soft AND 1.000 / 1.000 / 1.000. The setting saturates.

**Kaggle T4, bge-small fine-tuned** (1,000 steps, paraphrased values, 20,000 documents), Recall@10 overall / 8 conditions: dense 0.687 / 0.324; ColBERT 0.733 / 0.376; grouped 0.999 / 0.990; grouped + soft AND 1.000 / 1.000. LIMIT-small R@2: 0.155 / 0.713 / 0.761 / 0.785. The soft-AND checkpoints in this run were evaluated at τ = 0.1 instead of 0.05 because of a bug that was fixed later.

**Kaggle T4 x2, LateOn, training-free** (synthetic, Recall@10 at 2 / 4 / 6 / 8 conditions):
- as released, 32-token queries: 0.976 / 0.240 / 0.060 / 0.008
- full query: 0.978 / 0.662 / 0.280 / 0.106
- grouped: 0.946 / 0.626 / 0.246 / 0.100
- grouped + soft AND: 0.834 / 0.484 / 0.168 / 0.076

Full LIMIT (50k documents): R@2 0.744, R@10 0.898. NanoBEIR (SciFact, NFCorpus, FiQA, SCIDOCS, NQ) mean nDCG@10: 0.602.

**Soft-AND temperature on MultiConIR** (LateOn, T1 mean): τ 0.02 → 0.654, 0.05 → 0.706, 0.1 → 0.725, 0.2 → 0.732; plain grouping 0.737.

**Kaggle T4 x2, LateOn fine-tuned** (1,500 steps, synthetic + MS MARCO 50/50, 2 seeds; synthetic R@10 overall / 8 conditions; MultiConIR T1 mean; NanoBEIR; LIMIT R@2):

| Variant | Synthetic | MultiConIR T1 | NanoBEIR | LIMIT R@2 |
| --- | --- | --- | --- | --- |
| ColBERT | 0.867 / 0.511 | 0.683 | 0.534 | 0.885 |
| grouped | 0.924 / 0.665 | 0.694 | 0.491 | 0.700 |
| grouped + soft AND | 0.993 / 0.968 | 0.625 | 0.494 | 0.679 |

**MultiConIR, calibrated variants** (T1 mean; LateOn / GTE-ModernColBERT / mLateOn): grouped with calibrated clauses 0.736 / 0.763 / 0.847; soft AND 0.706 / 0.761 / 0.844; calibrated soft AND 0.731 / 0.762 / 0.847.

**Round 4, Kaggle T4 x2** (training-free; three models; fixed in advance):
- Competitive MaxSim, τ 0.1: worse than grouping on all three models.
- τ sensitivity, LateOn only: 0.03 worse, 0.3 close to grouping.
- Lead-in control and separate-encoding ablation: see Results.
- Localization: see Results.
- Latency, 100 candidates: about 28–31 ms per query on a T4.

Full tables are in `results/ROUND4.md`.

MultiConIR's "T2 Positive > HN1" uses the same document pairs as "T1 with 10 conditions", so it is reported once.

</details>

## License

MIT, see [LICENSE](LICENSE). Models and datasets used in the experiments are not included and keep their own licenses.
