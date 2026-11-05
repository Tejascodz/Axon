# Round 6: the index at scale

Hardware: Tesla T4, 580.178.04, 15360 MiB, Tesla T4, 580.178.04, 15360 MiB; host memory 31.3 GB.

## Stored vectors vs fresh encoding (re-ranking)

100 questions x 20 candidates: max score difference 0.00e+00, same order 100%, same evidence 100.0%.

## Scale

| passages | build | load | vectors | exact recall@100 | exact p50 / p95 (GPU) | exact p50 (streamed) | PLAID recall@100 | PLAID p50 / p95 | pipeline p50 / p95 (exact) | pipeline p50 / p95 (PLAID) | peak RAM |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 100,000 | 59.3 min | 15.4 s | 2.52 GB | 0.9757 | 119.0 / 157.1 ms | 693.9 ms | 0.9743 | 119.2 / 147.3 ms | 177.8 / 221.4 ms | 179.2 / 214.6 ms | 12.32 GB |
| 150,000 | 66.2 min | 21.5 s | 3.75 GB | 0.9714 | 167.5 / 224.4 ms | 1019.7 ms | 0.9707 | 145.7 / 187.2 ms | 235.5 / 294.1 ms | 214.6 / 263.2 ms | 17.75 GB |

Large size reduced from 300,000 to 151,876: the 100,000 build took 59 min.

## Incremental update (1% of the 100k index)

500 changed, 250 deleted, 250 added: update 36.98 s against a full build of 3556 s (96.1x). Compaction 2942.65 s.

* changed text found in the top 5: 94%
* deleted files returned in the top 20 of 100 queries for their own text: 0
* added files found in the top 5: 96%
* recall@100 before / after update / after compaction: 0.9757 / 0.9757 / 0.9757; PLAID after update 0.9743

## Dependency overrides (deploy/overrides.txt) vs PyLate's pins

encoding failed in the pinned environment (logs/r6_deps_env.log)

Not run in this round: the 30-minute load test through the server with SGLang (left out to save GPU quota).
