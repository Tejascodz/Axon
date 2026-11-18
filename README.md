<div align="center">

# Axon

**A retrieval-augmented generation (RAG) system built on late-interaction (ColBERT) retrieval, run as a service.**

Axon answers questions from your own documents, cites the passages it used, declines when they do not support an answer, and keeps instructions hidden in documents away from the model.

[![CI](https://github.com/Tejascodz/Axon/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/Tejascodz/Axon/actions/workflows/ci.yml)
[![Python 3.11 | 3.12](https://img.shields.io/badge/python-3.11%20%7C%203.12-3776AB)](pyproject.toml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue)](LICENSE)
[![Version 1.0.0](https://img.shields.io/badge/version-1.0.0-informational)](pyproject.toml)
[![LLM API: OpenAI-compatible](https://img.shields.io/badge/LLM%20API-OpenAI--compatible-555)](#configuration)

[Overview](#overview) · [Quickstart](#quickstart) · [How it works](#how-it-works) · [Results](#results) · [Usage](#usage) · [Related work](#comparison-with-related-work) · [Limitations](#limitations) · [Development](#reproducing-results-and-development) · [Security](#security)

</div>

<p align="center">
  <img src="docs/figures/results.svg" width="100%" alt="Two bar charts comparing a single search with the full Axon pipeline. Retrieval quality, higher is better: HotpotQA both gold passages in the top 5 rises from 0.826 to 0.882, MuSiQue all supporting passages in the top 10 from 0.378 to 0.532, and BRIGHT nDCG@10, mean of three held-out domains, from 0.155 to 0.258. Attack success, lower is better: unseen LLM-written prompt injections fall from 5.0% to 0.25%, hand-written injections from 4.5% to 0.5%, and false answers from planted documents that restate the question from 21% to 0%. Error bars are 95% Wilson intervals.">
</p>

<table align="center">
<tr>
<td align="center" width="33%"><h3>0.38 → 0.53</h3>multi-hop questions with every supporting passage retrieved<br><sub>MuSiQue, all supporting passages in top 10, 500 questions: 0.378 → 0.532</sub></td>
<td align="center" width="33%"><h3>0.16 → 0.26</h3>nDCG@10 on reasoning-heavy questions<br><sub>BRIGHT, mean of economics, psychology, robotics (305 queries): 0.155 → 0.258</sub></td>
<td align="center" width="33%"><h3>5% → 0.25%</h3>prompt-injection attacks that succeed<br><sub>40 injections no detector was tuned on, 400 attacks: 20 → 1</sub></td>
</tr>
</table>

<sub>Each pair compares one search with the default re-rank against the full pipeline, on the same index, with Qwen3-4B-Instruct-2507 on an NVIDIA T4 and LateOn as the retriever; one run per setting. Settings were chosen on separate data (HotpotQA dev questions, BRIGHT biology) before the test data was scored. Costs: follow-up searches double answer latency (5.2 s → 10.8 s p50 at 8 concurrent requests); the echo filter lowers HotpotQA exact match on clean questions from 0.334 to 0.312; and on MuSiQue the 4B generator still answers only 5% of questions exactly, so better retrieval has not yet become better answers there.</sub>

## Overview

Late-interaction retrievers such as ColBERT score a document by matching every query token against every document token. That makes them accurate, but a question that chains facts ("where was the director of this film born?") or is phrased very differently from its answer still misses the passage that matters, and anything retrieved goes straight into the model's prompt. Axon adds the missing pieces around the retriever: searches that follow the evidence, re-ranking that scores each condition of a question separately, and checks on both the passages going into the model and the answer coming out.

- **Hybrid retrieval with condition-aware re-ranking.** ColBERT and BM25 candidates are fused by reciprocal rank, then re-ranked with each condition of the question scored on its own, so a passage cannot win by matching most conditions very well and ignoring one.
- **Multi-hop search.** A second hop follows documents named in the top passages; optional LLM-written follow-up searches name the fact that is still missing.
- **Defences in the request path.** A pattern scanner and a prompt-injection classifier withhold suspicious passages, an echo filter drops planted passages that restate the question, sources reach the model escaped and marked as data, and answers that cite nothing, or cite sources the model never saw, are withheld.
- **An index built for change.** Memory-mapped vectors, incremental updates, compaction, optional PLAID for large collections, and versioned atomic commits that the server reloads while it serves.
- **Isolation and operations.** One key per tenant with tenant filtering inside both searches, rate limits, back-pressure, Prometheus metrics, Grafana dashboards and alert rules.

Ships as a Python library (`nyaya`), a command-line tool, an HTTP service with a small web page, and a Docker Compose deployment with SGLang, Prometheus and Grafana.

## Quickstart

Requires Python 3.11 or 3.12. The first run downloads the LateOn retriever from Hugging Face; a GPU is used if there is one, otherwise the CPU.

<!-- quickstart:start -->
```bash
pip install "git+https://github.com/Tejascodz/Axon"

mkdir -p docs
echo "Unused items can be returned within 30 days of delivery for a full refund." > docs/returns.md
echo "Orders ship within two business days. Express delivery arrives the next business day." > docs/shipping.md
echo "Electronics carry a one-year warranty against manufacturing defects." > docs/warranty.md

python -m nyaya.rag build --src docs --out index
python -m nyaya.rag ask --index index --no_llm --k 2 "How long do I have to send something back?"
```
<!-- quickstart:end -->

Expected output (timings vary):

```text
QUICKSTART_OUTPUT
```

`--no_llm` stops after retrieval. To get answers, point Axon at any OpenAI-compatible server and drop the flag:

```bash
export NYAYA_LLM_BASE_URL=http://localhost:30000/v1   # SGLang, vLLM, or a hosted endpoint
export NYAYA_LLM_MODEL=Qwen/Qwen3-4B-Instruct-2507      # the model name that server expects
python -m nyaya.rag ask --index index "How long do I have to send something back?"
```

To run the full service (Axon, SGLang, Prometheus, Grafana) on one GPU host, see [Run the service](#run-the-service).

## How it works

<p align="center">
  <img src="docs/figures/architecture.svg" width="92%" alt="Flow of one question through Axon. POST /ask checks the key, rate limit, body size and in-flight limit. Optional LLM query expansion. Hybrid first stage: ColBERT (exact or PLAID) and BM25, 100 candidates each, tenant-filtered, fused by reciprocal rank with k = 60. Condition-aware re-rank. Second hop through titles named in the top 2 passages, with 2 reserved places. Optional LLM follow-up searches. Source defences: injection scanner and classifier, echo filter. Guarded prompt with escaped sources. Generation on an OpenAI-compatible LLM server. Answer checks on citations and an optional claim check. Response with cited and withheld sources. The index feeds the four search stages.">
</p>

**Hybrid first stage.** ColBERT and BM25 each return 100 candidates, and reciprocal-rank fusion (k = 60) merges them. ColBERT search is exact over memory-mapped fp16 token vectors, or approximate with PLAID when an index has one; PLAID matched exact recall@100 at 100,000 passages and was faster from 150,000 on a T4. In a multi-tenant index the tenant filter runs inside both searches, before fusion, and every result is checked again before it is returned. Code: `AxonRAG.retrieve` in [`nyaya/rag.py`](nyaya/rag.py).

**Condition-aware re-rank.** The question is split into conditions (numbered items, semicolons, "and also"); each condition's token scores are averaged, then conditions are averaged with equal weight, and the per-condition scores are returned as evidence. Questions longer than 40 words can switch to plain ColBERT scoring (`method="auto"`), which did better on long forum-style questions. Code: [`nyaya/axon.py`](nyaya/axon.py), [`nyaya/scoring.py`](nyaya/scoring.py).

**Second hop.** For questions that chain two facts, the top 2 passages are scanned for titles of other documents; those documents are re-ranked and the best 2 take reserved places in the top k. Reserved places rather than a second fusion, because fusion favours passages both steps already agree on, which are rarely the missing one. On by default in the server (`AXON_HOPS=2`).

**Query expansion and follow-up searches** (off by default). Expansion asks the LLM what a document answering the question would contain and searches for that text instead of the question. Follow-up searches ask the LLM, after each search, for the one fact still missing, phrased with names rather than pronouns; at most 2 rounds, and passages the injection scanner flags are never shown to it. Code: [`nyaya/expand.py`](nyaya/expand.py).

**Source defences.** A pattern scanner (after Unicode normalisation, so invisible characters and look-alike letters do not hide an instruction) and, in the deployment, a DeBERTa-v3 prompt-injection classifier applied sentence by sentence withhold passages before the prompt is built. The echo filter withholds passages that contain half or more of the question's three-word sequences, the signature of documents planted to rank first for a question. Withheld passages are listed in the response with the reason, without their text. Code: [`nyaya/guards.py`](nyaya/guards.py), [`nyaya/injection.py`](nyaya/injection.py), [`nyaya/defend.py`](nyaya/defend.py).

**Generation and answer checks.** Sources are escaped and wrapped in `<source>` tags with an instruction to treat them as data. The answer must cite source ids; citations to ids that were not shown are removed, and an answer with no valid citation is replaced by a refusal. An optional verifier checks each claim against the passages it cites, with an NLI model or the LLM as judge. The LLM client retries transient errors with backoff, refuses redirects, caps reply size and opens a circuit breaker after repeated failures. Code: [`nyaya/llm.py`](nyaya/llm.py), [`nyaya/verify.py`](nyaya/verify.py).

**Index.** One folder: chunks (paragraph-first, 220 words, 40-word overlap), token vectors, BM25 statistics and an optional PLAID index. `update` re-encodes only files whose hash changed; `compact` drops deleted chunks without re-encoding. Every write goes to new files that are flushed and renamed into place, and `config.json`, written last, is the commit point; the server notices the new version and swaps it in between requests. Code: [`nyaya/index.py`](nyaya/index.py), [`nyaya/store.py`](nyaya/store.py).

## Results

**Setup.** Retriever: [LateOn](https://huggingface.co/lightonai/LateOn) at a pinned revision. Generator: Qwen3-4B-Instruct-2507 in fp16 on SGLang. Hardware: two NVIDIA T4 GPUs (16 GB each). One run per setting. Choices were made on held-out data first: HotpotQA dev questions for the second hop, the BRIGHT biology domain for the BRIGHT settings, clean dev passages for the injection classifier's threshold. Intervals are 95% Wilson intervals for proportions; per-question outputs were not kept for these runs, so differences are not paired-tested, and nDCG means have no interval. Raw results: [`results/`](results).

### Retrieval

| Benchmark (test questions) | One search | Axon | Interval (Axon) | Source |
|---|---|---|---|---|
| HotpotQA, both gold passages in top 5 (500) | 0.826 | **0.882** with the second hop | 0.851 – 0.907 | [r8.json](results/r8_run1/r8.json) |
| MuSiQue, all supporting passages in top 10 (500) | 0.378 | 0.418 with the second hop, **0.532** with follow-up searches | 0.488 – 0.575 | [r10.json](results/r10_run1/r10.json) |
| MuSiQue, all supporting passages in top 5 (500) | 0.244 | **0.396** with follow-up searches | | [r10.json](results/r10_run1/r10.json) |
| BRIGHT nDCG@10, economics (103) | 0.140 | 0.189 re-rank by length, **0.253** with expansion | | [r10.json](results/r10_run1/r10.json) |
| BRIGHT nDCG@10, psychology (101) | 0.178 | 0.214, **0.328** | | [r10.json](results/r10_run1/r10.json) |
| BRIGHT nDCG@10, robotics (101) | 0.147 | 0.147, **0.193** | | [r10.json](results/r10_run1/r10.json) |

Condition-aware re-ranking against plain ColBERT scoring on MultiConIR (nDCG@10, paired bootstrap intervals, [ROUND4.md](results/ROUND4.md)):

| Model | ColBERT scoring | Axon scoring | Difference (95% interval) | On 10-condition queries |
|---|---|---|---|---|
| [LateOn](https://huggingface.co/lightonai/LateOn) | 0.691 | **0.737** | +0.045 [+0.039, +0.051] | +0.062 [+0.043, +0.079] |
| [GTE-ModernColBERT-v1](https://huggingface.co/lightonai/GTE-ModernColBERT-v1) | 0.712 | **0.763** | +0.050 [+0.045, +0.055] | +0.067 [+0.051, +0.084] |
| [mLateOn](https://huggingface.co/lightonai/mLateOn) | 0.813 | **0.847** | +0.031 [+0.026, +0.036] | +0.112 [+0.092, +0.132] |

Scale (exact search, vectors on the GPU, [r6](results/r6_run1)):

| Passages | Recall@100 | Search p50 / p95 | PLAID recall@100, p50 |
|---|---|---|---|
| 6,852 | 0.992 | 32 / 35 ms | 0.993, 65 ms |
| 100,000 | 0.976 | 119 / 157 ms | 0.974, 119 ms |
| 150,000 | 0.971 | 168 / 224 ms | 0.971, 146 ms |
| 300,000 | 0.959 | 312 / 418 ms | not built |

### Answers

| HotpotQA, 500 test questions | Exact match | F1 | Declined | Latency p50, 8 concurrent |
|---|---|---|---|---|
| One search | 0.314 | 0.406 | 0.448 | 5.1 s |
| Second hop (server default) | 0.334 | 0.436 | 0.410 | 5.2 s |
| Second hop and follow-up searches | **0.350** | **0.449** | 0.402 | 10.8 s |

A LoRA fine-tune of the generator on the citation format (paired intervals, [ROUND5.md](results/ROUND5.md)) raised exact match from 0.324 to **0.506** (+18.2 [+14.4, +22.2]) and citation recall from 0.481 to **0.808**, and cut wrong refusals from 0.436 to 0.160; citation precision did not change significantly (−1.8 [−4.0, +0.4]).

### Security

| Attack | Without the defence | With it | Cost on clean questions | Source |
|---|---|---|---|---|
| 40 LLM-written injections, unseen by any detector (400 attacks) | 5.0% succeed | **0.25%** with the classifier | 1.1% of clean passages withheld; exact match unchanged (0.26) | [r10b.json](results/r10b_run1/r10b.json) |
| 30 hand-written injections (400 attacks) | 4.5% | **0.5%** | as above | [r10b.json](results/r10b_run1/r10b.json) |
| 3 planted passages restating the question with a false answer (100 questions) | 21% false answers | **0%** with the echo filter | exact match 0.334 → 0.312 (500 questions) | [r10.json](results/r10_run1/r10.json) |
| The same, paraphrased | 16% | 9% with echo and near-duplicate filters | exact match 0.29 → 0.26 | [r10.json](results/r10_run1/r10.json) |
| 440 queries written to reach another tenant's documents | | **0** of 4,400 results crossed tenants | | [r8.json](results/r8_run1/r8.json) |

Under load (one T4, 8 concurrent clients for 30 minutes, [r7b](results/r7b_run1)): 2,952 of 2,958 requests answered (99.8%), p50 / p95 4.5 / 8.6 s with generation, no latency drift, an index update applied while serving with no errors, and a burst of 48 clients refused with 503 within 197 ms (p99) at the 16-request limit.

### What did not work or is not significant

- **Answers on chained questions.** On MuSiQue the generator declines about 89% of questions even though all supporting passages are in the top 10 for 53% of them. Composing three or four facts is beyond the 4B model.
- **Claim-level checking does not predict correctness.** Both the NLI checker and the LLM judge catch altered claims (F1 0.883 and 0.967), but answers they mark as supported are not reliably more often right, so the check is off by default ([r7](results/r7_run1)).
- **Voting across passages** (answer from each passage alone, keep the majority) cut exact match on clean questions from 0.27 to 0.02 and was dropped.
- **BM25 fusion and English stemming** did not change the final BRIGHT ranking once re-ranking was applied.
- **The HotpotQA answer gain from follow-up searches** (0.334 → 0.350 exact match) is within the noise of 500 questions.
- **Paraphrased planted passages** still get through 9% of the time.

## Usage

### Command line

| Command | What it does |
|---|---|
| `python -m nyaya.rag build --src DIR --out INDEX [--plaid] [--tenants]` | Index `.md`, `.txt` and `.pdf` files; `--tenants` makes each top-level folder a tenant |
| `python -m nyaya.rag update --src DIR --index INDEX` | Re-index only added, changed and removed files |
| `python -m nyaya.rag compact --index INDEX` | Drop deleted chunks without re-encoding |
| `python -m nyaya.rag stats --index INDEX` | Model, version, chunk counts, PLAID state |
| `python -m nyaya.rag ask --index INDEX [--no_llm] [--k 5] [--tenant T] "question"` | Retrieve, then answer with citations |

### HTTP API

| Endpoint | Auth | Purpose |
|---|---|---|
| `POST /ask` `{"question": "...", "k": 5}` | API or tenant key | Answer with cited sources and withheld passages |
| `POST /rerank` `{"query": "...", "documents": [...], "conditions": [...]}` | API or tenant key | Re-rank documents, with per-condition evidence |
| `GET /` | none (static page, no data) | Web page for asking questions |
| `GET /health`, `GET /ready` | none | Liveness, and readiness once models and the index are loaded |
| `GET /metrics` | metrics key | Prometheus metrics |

```bash
curl -s http://localhost:8080/ask -H "Authorization: Bearer $AXON_API_KEY" \
     -H "Content-Type: application/json" -d '{"question": "What is the refund window?"}'
```

### Configuration

<details>
<summary>Server settings (environment variables)</summary>

| Variable | Default | Meaning |
|---|---|---|
| `AXON_INDEX` | required | Index folder |
| `AXON_API_KEY` / `AXON_API_KEY_FILE` | none | Key for `/ask` and `/rerank`; at least 16 characters |
| `AXON_TENANT_KEYS` / `AXON_TENANT_KEYS_FILE` | none | `tenant:key,tenant:key`; each key sees only its tenant |
| `AXON_METRICS_KEY` / `AXON_METRICS_KEY_FILE` | none | Key for `/metrics` |
| `AXON_REQUIRE_AUTH` | `1` in the image | Refuse to serve (503) when no key is configured |
| `AXON_HOPS` | `2` | `1` turns the second hop off |
| `AXON_ECHO_FILTER` | `1` | `0` turns the echo filter off |
| `AXON_INJECTION_DETECTOR` | off; `classifier` in `deploy/` | `classifier` or `llm` |
| `AXON_FOLLOW_UPS` | `0` | Up to 3 LLM-written follow-up searches |
| `AXON_EXPAND` | off | `bm25`, `append` or `replace` (the measured mode) |
| `AXON_VERIFY` | off | Claim-level check: `AXON_VERIFY_METHOD=nli` or `llm` |
| `AXON_RATE_PER_MIN` | `120` | Per tenant, or per client address without tenant keys |
| `AXON_MAX_INFLIGHT` | `16` | Concurrent `/ask` requests before 503 with `Retry-After` |
| `AXON_MAX_BODY_BYTES` | 16 MiB | Request bodies need a `Content-Length` no larger than this |
| `AXON_UI` | `1` | `0` turns the web page off |
| `NYAYA_LLM_BASE_URL` | `http://localhost:8000/v1` | OpenAI-compatible endpoint |
| `NYAYA_LLM_API_KEY` | none | Sent only to the configured host, never over plain HTTP to a remote one |
| `NYAYA_LLM_RETRIES`, `NYAYA_LLM_BREAKER_FAILURES`, `NYAYA_LLM_BREAKER_SECONDS` | 2, 5, 30 | Retries and circuit breaker |

</details>

<details>
<summary>Library</summary>

```python
from nyaya.rag import AxonRAG

rag = AxonRAG("index")
hits = rag.retrieve("Which jackets are waterproof and under 2 kg?", k=5, hops=2)
out = rag.answer("Which jackets are waterproof and under 2 kg?", retrieve_kw={"hops": 2}, echo_filter=True)
print(out["answer"], [s["id"] for s in out["sources"]], out["withheld"])
```

</details>

### Run the service

```bash
cd deploy
./make-secrets.sh                    # random keys and passwords in deploy/secrets/
cp .env.example .env                 # model, ports, limits; see the comments in the file
python -m nyaya.rag build --src /path/to/docs --out index
docker compose up -d                 # Axon on :8080, SGLang, Prometheus on :9090, Grafana on :3000
```

Operations, alerts, upgrades and rollback: [`deploy/RUNBOOK.md`](deploy/RUNBOOK.md).

## Comparison with related work

| Project | What it is | How Axon differs |
|---|---|---|
| [ColBERT](https://github.com/stanford-futuredata/ColBERT), [PLAID](https://arxiv.org/abs/2205.09707) | The late-interaction retriever and its fast search engine | Axon uses released ColBERT models and PLAID through [PyLate](https://github.com/lightonai/pylate), and adds per-condition scoring, hybrid BM25 fusion and a full answering pipeline |
| [RAGatouille](https://github.com/AnswerDotAI/RAGatouille), [PyLate](https://github.com/lightonai/pylate) | Libraries for training and using ColBERT models | Retrieval libraries; Axon is a served RAG system with generation, defences, tenancy and operations |
| [LlamaIndex](https://github.com/run-llama/llama_index), [Haystack](https://github.com/deepset-ai/haystack) | General RAG frameworks with many integrations | Far broader integrations; Axon is a single pipeline with each stage measured and its defaults chosen from those measurements |
| [IRCoT](https://arxiv.org/abs/2212.10509), [query2doc](https://arxiv.org/abs/2303.07678), [BRIGHT](https://arxiv.org/abs/2407.12883) | LLM-guided multi-step retrieval and query expansion | Axon's follow-up searches and expansion follow these methods; they are reproduced here on a 4B model, not new |
| [RobustRAG](https://arxiv.org/abs/2405.15556), [TrustRAG](https://arxiv.org/abs/2501.00879) | Defences against planted documents | Axon's near-duplicate filter is close to TrustRAG's first stage; isolate-and-vote in the style of RobustRAG cost too much accuracy here |
| [ProtectAI DeBERTa-v3 injection classifier](https://huggingface.co/protectai/deberta-v3-base-prompt-injection-v2) | An open prompt-injection classifier | Used as is, applied per sentence after normalisation, and measured end to end on injections it was not trained on |

## Limitations

- **The 4B generator limits answer quality.** It declines about 40% of HotpotQA questions that have an answer in the collection and most questions that chain three or more facts. Larger generators have not been measured.
- **One question at a time.** No conversation memory, and no summarising or rewriting of whole documents.
- **File types.** `.md`, `.txt` and `.pdf` with a text layer. Scanned PDFs need OCR first; Word and HTML need converting.
- **English.** The BM25 analyser, the injection classifier and every benchmark are English only.
- **Untested settings.** Query expansion was measured only on BRIGHT, not on short factual questions. Follow-up searches and expansion together were not measured. CPU-only serving was not benchmarked.
- **Single runs.** Each result is one run on one hardware setup; the per-question outputs of the multi-hop and security runs were not kept, so their differences are not paired-tested.
- **Hardware.** The generator needs a GPU with about 16 GB for the 4B model in fp16. CI covers Python 3.11 and 3.12 on Linux only.

## Reproducing results and development

```bash
git clone https://github.com/Tejascodz/Axon && cd Axon
pip install -e ".[server,eval,dev]" bandit
python -m pytest -q tests                  # unit and integration tests, CPU only
python -m pyflakes nyaya scripts tests     # lint
bandit -q -ll -r nyaya scripts             # security lint
python scripts/make_figures.py             # redraws docs/figures/results.svg from results/
```

The experiments run on two T4 GPUs (Kaggle "GPU T4 x2" with internet on). Each script writes JSON and a Markdown summary to `results/`; rounds 6 to 10 also take `--smoke`, which checks the script in a few minutes on a CPU with stand-in models.

| Results | Command |
|---|---|
| MultiConIR re-ranking | `python scripts/round4.py` |
| Answers, LoRA, serving, injection prompts | `python scripts/round5.py` |
| Scale and PLAID | `python scripts/round6.py` |
| Load, live updates, claim checking | `python scripts/round7.py` |
| Second hop, MuSiQue, BRIGHT, attacks, tenants | `python scripts/round8.py` |
| Expansion, follow-ups, defences | `python scripts/round10.py` |
| Unseen injections | `R10_OUT=r10b.json python scripts/round10.py --only injection` |

<details>
<summary>Project structure</summary>

```text
nyaya/            library
  rag.py          ingestion, AxonRAG pipeline, command line
  axon.py         condition-aware re-ranker on ColBERT models
  index.py        build, update, compact, versioned commits
  store.py        memory-mapped token-vector store
  bm25.py, stem.py  keyword search and Porter stemming
  expand.py       query expansion and follow-up searches
  guards.py       prompt construction, injection scanner, citation checks
  injection.py    learned injection detectors
  defend.py       echo filter, near-duplicate filter, voting
  verify.py       claim-level checking (NLI or LLM judge)
  llm.py          OpenAI-compatible client with retries and circuit breaker
  server.py       FastAPI service; webui.py is its web page
deploy/           Dockerfile, Compose, Prometheus, alerts, Grafana, runbook, CI fixtures
scripts/          experiments, figures, dashboard generator
results/          every committed result
tests/            pytest suite
```

</details>

## Security

- **Authentication is required by default** in the container image: without a key the service answers 503 rather than serving openly. Keys shorter than 16 characters are refused; `deploy/make-secrets.sh` writes 43-character random keys. Keys can be read from files (Docker secrets) and are compared in constant time.
- **Limits.** Per-tenant rate limits, a cap on concurrent requests, request bodies capped before they are read, and size limits on questions, documents and conditions.
- **Hardening.** The container runs as a non-root user on a read-only filesystem with all capabilities dropped; responses carry `nosniff`, `no-store` and `no-referrer`, and the web page a strict Content-Security-Policy. Compose publishes every port on 127.0.0.1 only; put a TLS proxy in front before exposing it.
- **Supply chain.** Model revisions are pinned, CI pins GitHub Actions to commits, and pip-audit, bandit and a Docker end-to-end test run on every push. Seven upstream advisories without a compatible fix are accepted and documented with their mitigations in [`deploy/RUNBOOK.md`](deploy/RUNBOOK.md).

To report a vulnerability, please use [private vulnerability reporting](https://github.com/Tejascodz/Axon/security/advisories/new) rather than a public issue.

## License and citation

MIT, see [LICENSE](LICENSE). If you use Axon in research, please cite it using [CITATION.cff](CITATION.cff).
