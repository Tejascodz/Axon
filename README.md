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

**Main negative result.** Combining conditions with a fuzzy AND (a soft minimum, as suggested by recent theory on conjunctive retrieval) is much better than averaging on clean synthetic data. On real text it is worse than plain averaging, for all three models, with or without calibration. Details below.

## Contents

- [How it works](#how-it-works)
- [Results](#results)
- [What did not work](#what-did-not-work)
- [Relation to prior work](#relation-to-prior-work)
- [Scope and limitations](#scope-and-limitations)
- [Reproduce](#reproduce)
- [Repository layout](#repository-layout)
- [Security notes](#security-notes)
- [Experiment log](#experiment-log)

## How it works

1. **Split the query into conditions** (`nyaya/model.py`, `split_clauses`). Numbered lists ("Find a book that: 1. …, 2. …") give one clause per item, with the lead-in dropped. Semicolons, new lines, "and also" and "plus" also separate clauses. For descriptive prose, sentence splitting can be switched on. An LLM query planner can replace this rule-based splitter.
2. **Encode each condition** with the model's own query encoder, which produces token vectors $q_{c,1..L_c}$ for condition $c$.
3. **Score** a document with token vectors $d_{1..K}$:

$$
s(q, d) = \frac{1}{C}\sum_{c=1}^{C} \frac{1}{L_c} \sum_{t=1}^{L_c} \max_{k} \langle q_{c,t}, d_k \rangle
$$

Standard ColBERT scoring is the same inner MaxSim, averaged over all query tokens at once, so long or wordy conditions get more weight. Documents are encoded and indexed exactly as before. Only the query side and the final aggregation change (`nyaya/scoring.py`, `score_grouped`).

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

**Why condition-aware scoring is needed.** On a synthetic AND-of-OR product collection (20,000 documents, hard negatives that break one condition), released LateOn's Recall@10 falls from 0.98 with 2 conditions to 0.11 with 8 conditions, even with the full query visible.

## What did not work

These are reported because they are informative, and because the obvious next ideas are among them.

1. **Fuzzy AND across conditions (soft minimum).** Theory says a minimum over conditions should beat an average for AND-of-OR relevance (Agarwal et al. 2026; `scripts/simulate_margin.py` reproduces the margin argument).
   - **On synthetic data after fine-tuning it is excellent:** 8-condition Recall@10 rises from 0.511 to 0.968.
   - **On MultiConIR it is worse than plain averaging:** LateOn T1 0.706 vs 0.737 (−0.032 [−0.035, −0.028]). Softer temperatures help monotonically, approaching the average.
   - **Calibration closes most of the gap but never wins.** Dividing each condition's score by its score against its own text recovers most of the loss (−0.006 for LateOn), but it never beats plain averaging, and it is slightly worse on monotonicity for all three models (pre-registered test).
   - **Why:** real conditions are paraphrased and partly satisfied, so the weakest condition's score is often noise rather than a missing condition.
2. **Compressing documents into a few learned "facet" vectors** (ConstBERT-style, with competitive slot attention). Trained from scratch on CPU, it scored below a single-vector dense model (Recall@10 0.30 vs 0.47).
3. **Fine-tuning on synthetic multi-condition data** (LateOn, 1,500 steps, mixed 50/50 with MS MARCO, 2 seeds). It hurt ordinary search (NanoBEIR nDCG@10 0.602 → 0.49–0.53) and did not beat training-free Nyaya scoring on MultiConIR. The released weights are the better starting point.

## Relation to prior work

| Work | Relation |
| --- | --- |
| ColBERT / ColBERTv2; LateOn, GTE-ModernColBERT, mLateOn (LightOn, 2025–26) | The models and scoring rule this builds on; Nyaya changes only how query-token scores are aggregated. |
| MultiConIR (EMNLP Findings 2025) | The benchmark. Its paper evaluates BM25, dense embedders and rerankers, not late-interaction models. |
| Yin, Lu, Qi, Shen, *When Should Queries Be Decomposed?* (arXiv 2606.08577, 2026) | Closest work. One sub-query per condition with summed scores, for dense embedders and cross-encoder rerankers: it hurts first-stage dense retrieval and helps reranking (at about 2× cost). Nyaya's scoring equals that decomposition applied to a late-interaction model when each sub-query's score is length-normalised (summing raw ColBERT sums would instead weight every token equally), computed in a single pass; this repository is the late-interaction counterpart of that study. |
| LIMIT (Weller et al., ICLR 2026); Agarwal et al., *Retrieval Needs Multivectors* (2026) | Theory and benchmarks showing single vectors fail on combination queries; motivated the AND experiments. |
| Killingback et al. (2026) | MaxSim as an aggregation of soft-ORs; Signed MaxSim for negation. |
| EXCISE (Ali et al., 2026) | Query-side exclusion (NOT) for late interaction; complementary to conditions combined by AND. |

## Scope and limitations

- **What is tested is ranking, not first-stage search.** MultiConIR's official tasks compare the relevant document against its own hard negatives, a reranking-style setting. Whether condition-aware scoring also helps first-stage retrieval over a whole collection has not been tested on real data; on the synthetic first-stage test it was slightly below plain LateOn (Recall@10 0.475 vs 0.495). Yin et al. report that decomposition hurts first-stage retrieval for dense models.
- **Two effects are not yet separated.** Encoding each condition on its own, and giving each condition equal weight, are not separated by an ablation.
- **Query splitting is rule-based.** It works for numbered or semicolon-separated conditions and, with sentence splitting, for descriptive queries; free-form queries need an LLM planner.
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

**RAG over your own files** (hybrid BM25 + Nyaya retrieval, answers with citations from any OpenAI-compatible endpoint):

```bash
python -m nyaya.rag build --src docs/ --ckpt runs/model.pt --out index/
export NYAYA_LLM_BASE_URL=http://localhost:8080/v1      # local server, or an https endpoint
export NYAYA_LLM_MODEL=<model name>                       # NYAYA_LLM_API_KEY only if the endpoint needs one
python -m nyaya.rag ask --index index/ --ckpt runs/model.pt "linen or cotton shirt; for summer; made in India"
```

## Repository layout

```
nyaya/scoring.py      MaxSim, ColBERT mean, soft-min, condition-grouped scoring
nyaya/model.py        backbones (tiny, Hugging Face, PyLate ColBERT checkpoints), clause splitter, retriever
nyaya/benchmarks.py   MultiConIR (official win-rate metrics, paired outputs), NanoBEIR
nyaya/data_andor.py   synthetic AND-of-OR collections with condition-breaking hard negatives
nyaya/evaluate.py     exact late-interaction search, recall / nDCG / MRR, BEIR and LIMIT loaders
nyaya/train.py        contrastive training (CPU or GPU, fp16), all variants
nyaya/bm25.py         BM25
nyaya/rag.py          indexing, hybrid search, cited answers
scripts/round2.py     main experiment: three models, paired bootstrap intervals
scripts/kaggle_run.py multi-GPU job queue for unattended runs (training-free + fine-tuning)
scripts/eval_suite.py one evaluation entry point for checkpoints and released models
scripts/prep_data.py  dataset and model downloads
tests/                scoring and splitter tests
```

## Security notes

- Checkpoints and indexes are loaded with `torch.load(..., weights_only=True)`, so they can hold only tensors and plain data. The BM25 index is rebuilt from text, and no pickled objects are written or read. Keep `torch>=2.6`: earlier versions have a known flaw in `weights_only` loading.
- `NYAYA_LLM_API_KEY` is read from the environment only, is sent only when set, and is refused over plain http to non-local hosts.
- Retrieved text is HTML-escaped inside `<source>` tags before it reaches the LLM, and the system prompt treats sources as data. LLM output is printed, never executed. As with any RAG system, indexed documents can still try to influence the answer, so index only content you trust.
- The experiment runner quotes every user-supplied value before it reaches the shell.

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

MultiConIR's "T2 Positive > HN1" uses the same document pairs as "T1 with 10 conditions", so it is reported once.

</details>

## License

MIT, see [LICENSE](LICENSE). Models and datasets used in the experiments are not included and keep their own licenses.
