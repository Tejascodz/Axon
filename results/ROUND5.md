# Round 5: the whole RAG system, served and measured

## Retrieval (HotpotQA, Axon on LateOn)

700 questions, a corpus of 6852 paragraphs, indexed in 197.1 s.

| Measure | Value |
|---|---|
| recall@5, both gold paragraphs (BM25 + ColBERT, RRF, Axon re-rank) | 0.824 |
| recall@5, both gold, BM25 + ColBERT fused without re-ranking | 0.699 |
| recall@5, at least one gold paragraph | 0.997 |
| BM25 recall@100 | 0.966 (median 0.66 ms, p95 0.96 ms per query) |
| ColBERT exact search recall@100 | 0.992 (median 31.95 ms, p95 35.26 ms per query) |
| ColBERT PLAID (approximate) recall@100 | 0.993 (median 65.02 ms, p95 69.8 ms per query) |
| full retrieval pipeline per query | median 62.69 ms, p95 78.14 ms, p99 122.59 ms |

| Ranking (gold = the 2 supporting paragraphs) | BM25 + ColBERT fused | + Axon re-rank | re-rank fused with BM25 again |
|---|---|---|---|
| recall@5 | 0.846 | 0.911 | 0.846 |
| recall@10 | 0.931 | 0.960 | 0.929 |
| recall@20 | 0.966 | 0.978 | 0.959 |
| both@5 | 0.699 | 0.824 | 0.700 |
| mrr@10 | 0.919 | 0.964 | 0.918 |
| ndcg@10 | 0.845 | 0.910 | 0.843 |

The generator was given the top 5 of **reranked**, chosen beforehand on the 200 held-out questions (both gold paragraphs in the top 5), not on the 500 graded ones.

Tenant isolation (corpus split between two tenants, 200 queries as tenant t0): 0 cross-tenant results among 2000 returned; a query without a tenant was refused.

## Q9 Serving (T4, fp16, Qwen3-4B-Instruct-2507, ~850-token RAG prompts, 128-token outputs)

Every request uses a prompt not sent before in that benchmark, so prefix caching helps only with the shared system prompt. vLLM and SGLang were measured back to back in a separate serving-only run, with nothing running on the other GPU (the transformers baseline is from the first run, measured while GPU 1 was fine-tuning); the first run's figures are kept in results/r5_bench_*_run1.json.

**transformers server, one request at a time**

| Concurrency | TTFT p50 / p99 ms | ITL p50 / p99 ms | TPOT ms | e2e p50 ms | requests/s | output tokens/s | GPU memory MiB | GPU util % | SM clock MHz | max temp °C | failed |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 853.5 / 1072.1 | 66.8 / 470.3 | 65.3 | 9176.4 | 0.11 | 14.0 | 8969.0 | 89.7 | – | – | 0 |
| 4 | 26837.7 / 27593.4 | 65.7 / 383.9 | 63.7 | 35171.9 | 0.113 | 14.4 | 8969.0 | 90.2 | – | – | 0 |

Server-side (Prometheus /metrics) at the highest concurrency: `hf_time_to_first_token_seconds_mean` 793.8 ms, `hf_e2e_request_seconds_mean` 8887.5 ms

**vLLM** (0.33.0)

| Concurrency | TTFT p50 / p99 ms | ITL p50 / p99 ms | TPOT ms | e2e p50 ms | requests/s | output tokens/s | GPU memory MiB | GPU util % | SM clock MHz | max temp °C | failed |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 556.3 / 743.2 | 39.3 / 79.9 | 39.1 | 5534.2 | 0.182 | 23.3 | 12951.0 | 99.9 | 1101 | 63.0 | 0 |
| 4 | 1511.7 / 2408.3 | 51.8 / 105.3 | 55.1 | 8773.7 | 0.467 | 59.8 | 13127.0 | 98.5 | 1279 | 65.0 | 0 |
| 16 | 3885.5 / 7754.4 | 92.2 / 1459.4 | 122.9 | 19637.2 | 0.82 | 105.0 | 13127.0 | 99.4 | 1302 | 69.0 | 0 |
| 32 | 4480.0 / 16925.0 | 159.7 / 1550.5 | 239.8 | 35983.2 | 0.883 | 113.0 | 13127.0 | 99.7 | 1350 | 72.0 | 0 |

Server-side (Prometheus /metrics) at the highest concurrency: `vllm:prefix_cache_queries_total{engine="0",model_name="base"}` 108718, `vllm:prefix_cache_hits_total{engine="0",model_name="base"}` 18432, `vllm:time_to_first_token_seconds_mean{engine="0",model_name="base"}` 5635.8 ms, `vllm:inter_token_latency_seconds_mean{engine="0",model_name="base"}` 239.9 ms, `vllm:request_time_per_output_token_seconds_mean{engine="0",model_name="base"}` 239.9 ms, `vllm:e2e_request_latency_seconds_mean{engine="0",model_name="base"}` 36098.8 ms

**SGLang** (0.5.23)

| Concurrency | TTFT p50 / p99 ms | ITL p50 / p99 ms | TPOT ms | e2e p50 ms | requests/s | output tokens/s | GPU memory MiB | GPU util % | SM clock MHz | max temp °C | failed |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 305.6 / 368.9 | 36.0 / 75.2 | 36.6 | 4868.5 | 0.202 | 25.9 | 13831.0 | 99.8 | 833 | 63.0 | 0 |
| 4 | 1013.1 / 1265.5 | 39.6 / 79.6 | 41.5 | 6183.6 | 0.653 | 83.6 | 13863.0 | 97.7 | 972 | 65.0 | 0 |
| 16 | 2547.4 / 4325.4 | 51.6 / 106.6 | 64.2 | 10782.6 | 1.507 | 192.9 | 13863.0 | 98.5 | 778 | 68.0 | 0 |
| 32 | 5013.3 / 8907.7 | 75.3 / 156.4 | 105.0 | 18225.0 | 1.762 | 225.5 | 13865.0 | 98.2 | 671 | 70.0 | 0 |

Server-side (Prometheus /metrics) at the highest concurrency: `sglang:time_to_first_token_seconds_mean{engine_type="unified",is_streaming="true",model_name="base"}` 4767.1 ms, `sglang:e2e_request_latency_seconds_mean{engine_type="unified",is_streaming="true",model_name="base"}` 18100.5 ms, `sglang:request_time_per_output_token_seconds_mean{engine_type="unified",is_streaming="true",model_name="base"}` 105.0 ms, `sglang:inter_token_latency_seconds_mean{engine_type="unified",model_name="base"}` 105.0 ms

**vLLM, fine-tuned model** (0.33.0)

| Concurrency | TTFT p50 / p99 ms | ITL p50 / p99 ms | TPOT ms | e2e p50 ms | requests/s | output tokens/s | GPU memory MiB | GPU util % | SM clock MHz | max temp °C | failed |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 569.8 / 742.4 | 39.7 / 80.8 | 39.6 | 5607.4 | 0.179 | 23.0 | 12951.0 | 99.8 | 1039 | 65.0 | 0 |
| 16 | 4343.9 / 8096.5 | 93.0 / 1460.2 | 124.5 | 19949.0 | 0.808 | 103.4 | 13103.0 | 99.7 | 1300 | 69.0 | 0 |

Server-side (Prometheus /metrics) at the highest concurrency: `vllm:prefix_cache_queries_total{engine="0",model_name="ft"}` 53434, `vllm:prefix_cache_hits_total{engine="0",model_name="ft"}` 9216, `vllm:time_to_first_token_seconds_mean{engine="0",model_name="ft"}` 3949.0 ms, `vllm:inter_token_latency_seconds_mean{engine="0",model_name="ft"}` 124.5 ms, `vllm:request_time_per_output_token_seconds_mean{engine="0",model_name="ft"}` 124.5 ms, `vllm:e2e_request_latency_seconds_mean{engine="0",model_name="ft"}` 19759.1 ms

## Q10 Answers, citations, abstention

| Measure | base | fine-tuned | fine-tuned − base (paired) |
|---|---|---|---|
| EM | 0.324 [0.284, 0.364] | 0.506 [0.462, 0.550] | +18.2 [+14.4, +22.2] |
| F1 | 0.420 [0.380, 0.461] | 0.627 [0.590, 0.666] | +20.8 [+17.2, +24.5] |
| citation precision | 0.932 [0.908, 0.955] | 0.913 [0.893, 0.932] | -1.8 [-4.0, +0.4] |
| citation recall | 0.481 [0.444, 0.520] | 0.808 [0.774, 0.840] | +32.7 [+28.7, +36.5] |
| false abstention (answerable) | 0.436 [0.392, 0.480] | 0.160 [0.128, 0.192] | -27.6 [-31.8, -23.4] |
| abstention (gold withheld) | 0.950 [0.915, 0.975] | 0.915 [0.875, 0.950] | -3.5 [-7.0, +0.0] |
| well-formed replies | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | – |
| answer found in its cited sources | 0.885 [0.844, 0.922] | 0.972 [0.954, 0.987] | +9.3 [+5.8, +13.1] |
| answered without any valid citation | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | +0.0 [+0.0, +0.0] |
| invalid citations removed (per reply) | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | – |
| replies cut off at the token limit | 0.001 [0.000, 0.004] | 0.000 [0.000, 0.000] | – |

**With the no-citation-no-answer rule.** The rule was added after the first round 5 run, so it is reported separately, not in place of the pre-registered results. It is applied to the same replies, and any answer that cites no source counts as a refusal.

| Measure | base | fine-tuned | fine-tuned − base (paired) |
|---|---|---|---|
| EM | 0.324 [0.284, 0.364] | 0.506 [0.462, 0.550] | +18.2 [+14.4, +22.2] |
| F1 | 0.420 [0.380, 0.461] | 0.627 [0.590, 0.666] | +20.8 [+17.2, +24.5] |
| false abstention (answerable) | 0.436 [0.392, 0.480] | 0.160 [0.128, 0.192] | -27.6 [-31.8, -23.4] |
| abstention (gold withheld) | 0.950 [0.915, 0.975] | 0.915 [0.875, 0.950] | -3.5 [-7.0, +0.0] |

## Q11 Prompt injection (attack success rate, lower is better)

| Prompt | base | fine-tuned | EM under attack (base) |
|---|---|---|---|
| undefended (plain concatenation) | 10.5% [6.5, 15.0] | 2.5% [0.5, 5.0] | 0.335 [0.270, 0.400] |
| defended (escaped sources, data-not-instructions rule) | 2.0% [0.5, 4.0] | 0.0% [0.0, 0.0] | 0.285 [0.225, 0.350] |
| defended + input scanner | 0.5% [0.0, 1.5] | 0.0% [0.0, 0.0] | 0.285 [0.225, 0.345] |

Input scanner: detected 150/200 planted injections (by template: [25, 25, 0, 0, 25, 25, 25, 25]); flagged 0 of 6852 clean paragraphs (false-positive rate 0.00%).
Defended prompt, base model, success by attack: direct override 8%, fake system tag 4%, fake answer note 4%, required prefix 0%, system override 0%, exfiltration 0%, zero-width hidden 0%, look-alike letters 0%.

## Q12 Condition check: LLM alone vs LLM + Axon certificate

| Documents | LLM alone | LLM + certificate | certificate − alone (paired) |
|---|---|---|---|
| all 600 | 60.8% [56.8, 64.7] | 64.2% [60.2, 68.0] | +3.3 [+0.7, +6.2] |
| satisfy every condition (truth ALL) | 96.7% [94.7, 98.7] | 99.0% [97.7, 100.0] | +2.3 [+0.7, +4.3] |
| break one condition | 25.0% [20.3, 30.0] | 29.3% [24.3, 34.7] | +4.3 [-1.0, +9.3] |
| fine-tuned model, all 600 | 60.7% [56.7, 64.5] | 58.8% [54.8, 62.8] | -1.8 [-4.2, +0.5] |

Axon alone (no LLM; ALL if the weakest condition's relative score ≥ 0.901, threshold fixed on 200 calibration documents): 50.0% [46.0, 54.0].
Replies without a parsable verdict: 0 of 1200.

## Q13 Fine-tuning run

LoRA r=16, alpha=32.0, lr 0.0001, 1600 examples (0 too long), 200/200 steps in 83 min, 33.0 M trainable parameters; loss 1.999 → 0.087; 0 non-finite losses skipped.

All steps succeeded.

