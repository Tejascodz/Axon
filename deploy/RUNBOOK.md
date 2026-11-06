# Axon operations runbook

How to deploy, run, update and debug the Axon stack. Every alert in `alerts.yml` links to a section below.

## The stack

| Service | What it does | Port (host) | Health |
|---|---|---|---|
| `sglang` or `vllm` | LLM server (one of them, picked by `--profile`) | none: backend network only | `/health` |
| `axon` | Retrieval, re-ranking, guardrails, `/ask`, `/rerank` | `127.0.0.1:8080` | `/health` (alive), `/ready` (model and index loaded) |
| `prometheus` | Scrapes Axon and the LLM server every 15 s, evaluates `alerts.yml` | `127.0.0.1:9090` | `/-/ready` |
| `grafana` | The "Axon" dashboard (provisioned, read-only) | `127.0.0.1:3000` | `/api/health` |

Every published port binds to localhost. To serve users, put a TLS reverse proxy (nginx, Caddy, a cloud
load balancer) in front of `axon:8080` only. Prometheus and Grafana stay private.

Which LLM server: on a T4, SGLang 0.5.23 gave 1.8x the throughput of vLLM 0.33.0 at 16 concurrent
requests and a p99 inter-token latency of 107 ms against 1,459 ms (round 5). The likely reason is that
vLLM falls back to Triton attention on GPUs older than compute capability 8.0. On A100, H100 or L4,
measure both before you choose.

## First deployment

```sh
cp deploy/.env.example deploy/.env           # set the model, image tags, limits
deploy/make-secrets.sh                       # random keys in deploy/secrets/ (git-ignored, dir mode 700)
python -m nyaya.rag build --src docs/ --out deploy/index [--plaid] [--tenants]
docker compose -f deploy/docker-compose.yml --profile sglang up -d --build
docker compose -f deploy/docker-compose.yml ps          # wait for "healthy"; the first start downloads models
curl -s localhost:8080/ready
curl -s localhost:8080/ask -H "Authorization: Bearer $(cat deploy/secrets/axon_api_key)" \
     -H 'Content-Type: application/json' -d '{"question": "..."}'
```

Grafana: `http://localhost:3000`. User `admin`, password in `deploy/secrets/grafana_admin_password`.

Answer-time options (`deploy/.env`; each was measured in round 10, `results/ROUND10.md`):

| Setting | Default here | What it buys | What it costs |
|---|---|---|---|
| `AXON_INJECTION_DETECTOR=classifier` | on | injection attacks that succeed 4.5% -> 0.5% (30 hand-written attacks) | 1.1% of clean passages withheld; +0.7 GB RAM; +0.03 s per question on a T4 (CPU not measured) |
| `AXON_INJECTION_DETECTOR=llm` | off | 0.75% attack success, 0.5% of clean passages withheld | one LLM call per question (+2.5 s) |
| `AXON_ECHO_FILTER=1` | on | planted passages that restate the question: false answers 21% -> 0% | HotpotQA exact match 0.334 -> 0.312 on clean questions |
| `AXON_FOLLOW_UPS=2` | off | MuSiQue all supporting passages in top 10: 0.42 -> 0.53; HotpotQA exact match 0.334 -> 0.350 | about twice the latency (5.2 s -> 10.8 s at 8 concurrent) |
| `AXON_EXPAND=replace` | off | BRIGHT nDCG@10 0.18 -> 0.26 on long reasoning questions | one LLM call before retrieval; not measured on short factual questions |

The classifier is downloaded on first start, at the commit in `AXON_INJECTION_REVISION`. If it cannot be
loaded, `/ready` stays false: the server does not run without the detector it was configured with.

GPU for Axon itself: the image is CPU-only by default. That is fine for re-ranking about 100
candidates per query. For large exact-search indexes, build with
`--build-arg TORCH_INDEX=https://download.pytorch.org/whl/cu126` and give the `axon` service a GPU
reservation, as the LLM services have.

## Keys

* `axon_api_key`: clients send `Authorization: Bearer <key>`.
* `axon_metrics_key`: only Prometheus has it, and it can read `/metrics` and nothing else.
  Client keys cannot read `/metrics`.
* Multi-tenant index (built with `--tenants`): set `AXON_TENANT_KEYS_FILE` to a file holding
  `tenant:key` pairs, one per line or comma-separated, each key at least 16 characters. The key decides
  the tenant. A malformed file makes Axon refuse every request (503); it never turns authentication off.
* The image sets `AXON_REQUIRE_AUTH=1`. If no key is configured, every request gets 503 instead of
  being served without authentication.

**Rotating a key.** Write the new value into the file. Axon re-reads a key file when it changes, with no
restart. Clients holding the old key get 401 from then on, so hand out the new key first. After rotating
the metrics key, restart Prometheus too (`docker compose restart prometheus`).

## Changing the documents

```sh
python -m nyaya.rag update --src docs/ --index deploy/index    # encodes only new and changed files
python -m nyaya.rag stats --index deploy/index
python -m nyaya.rag compact --index deploy/index               # when deleted chunks pile up
```

* `update` compares SHA-256 hashes. Removed and changed files are hidden from search at once
  (tombstones), and only new or changed files are encoded.
* Axon picks up the new version on its next `/ask`, with no restart. Watch `axon_index_version` in Grafana.
  Requests already in flight finish on the snapshot they started with.
* Writers take an exclusive lock, so two `update` runs at once fail fast instead of corrupting the index.
  Every file is written to a temporary name and renamed into place, `config.json` last, so a reader sees
  the old index or the new one, never a mix.
* Back up `deploy/index/` before a `compact`, or before a `build` that replaces an existing index. The
  folder is the whole state; Axon keeps nothing else.

## Upgrades and rollback

* Axon: `docker compose build axon && docker compose up -d axon`. To roll back, set `AXON_TAG` to the
  previous tag and run `up -d axon`. Index format 2 is read by every 1.x release. Format-1 (padded)
  indexes still load, but `update` asks you to rebuild them once.
* LLM server: change `SGLANG_IMAGE` or `VLLM_IMAGE` and run `up -d`. Re-run the serving benchmark
  (`scripts/round5.py --rebench`) before and after. Server versions have changed latency by 2x or more
  on the same GPU.
* Pin images by digest (`image@sha256:...`) in `.env` once a version is approved, so a re-pushed tag cannot
  change what runs.

## Dependency advisories accepted

`deploy/overrides.txt` lists advisories in PyLate 1.6.0's pinned dependencies that cannot be fixed by
upgrading: the fixed sentence-transformers and transformers releases stop LateOn from loading (tested
on Kaggle and offline). Their common path is loading a malicious model or file. Mitigations:
* models load only by pinned commit (`AXON_MODEL_REVISION`, `AXON_VERIFY_REVISION`), from repositories
  you trust, with `trust_remote_code` off (the default);
* the container's root filesystem is read-only, it runs as a non-root user with no capabilities;
* never point `AXON_MODEL` at a model you have not reviewed.
CI fails on any advisory not in that list. Revisit when PyLate supports newer sentence-transformers.

## Logs

Axon writes one JSON line per request to stdout: `ts`, `request_id`, `method`, `route`, `status`, `ms`,
`tenant`. Questions, documents, answers and keys are never logged. Each response carries
`X-Request-ID`, which a caller can also set (letters, digits, `.`, `_`, `-`, up to 64 characters).
To trace a complaint, ask for that ID:

```sh
docker compose -f deploy/docker-compose.yml logs axon | grep '"request_id": "<id>"'
```

Container logs rotate at 5 x 50 MB per service.

## Capacity (measured, round 5, one T4 per role)

* Retrieval, 7,000 passages: p50 63 ms, p95 78 ms, p99 123 ms for the first stage plus re-ranking.
  Exact search and PLAID gave the same recall@100 (0.992 and 0.993). Exact search was faster at that
  size (32 ms against 65 ms). Round 6 measures where PLAID starts to pay off.
* Generation, Qwen3-4B on SGLang, prompts of about 850 tokens: time to first token 306 ms at one request,
  37 ms per output token, 193 tokens/s at 16 concurrent requests.
* `AXON_MAX_INFLIGHT` should be about the LLM's `--max-running-requests`. Above that, requests only queue
  inside the LLM and time to first token climbs. Axon refuses the excess with 503 and `Retry-After: 1`
  instead.

---

## Alerts

### AxonDown
Prometheus cannot scrape Axon. Check `docker compose ps axon` and `docker compose logs --tail 100 axon`.
Other things to check:
* An OOM kill: `docker inspect axon-axon-1 --format '{{.State.OOMKilled}}'`. If true, raise
  `AXON_MEM_LIMIT` (vectors take about chunks x 200 tokens x 128 x 2 bytes).
* A changed metrics key: restart Prometheus.

### AxonNotReady
Axon has been running for 15 minutes without loading its model or index. The log line
`"event": "preload_failed"` has the reason. Common causes:
* No network to download the model. Pre-fill the `axon-data` volume, or mount a local model and point
  `AXON_MODEL` at it.
* `AXON_INDEX_DIR` points at an empty folder.
* The index was built with a different model. `stats` shows which one.

### LLMDown
The LLM server is not answering, so `/ask` returns 502 while `/rerank` keeps working. Check
`docker compose logs --tail 200 sglang` (or `vllm`). Usual causes:
* CUDA out of memory at start: lower `LLM_MAX_LEN` or `LLM_MAX_SEQS`.
* A driver mismatch with the image's CUDA version.
* A gated model without `HF_TOKEN`.

The first start can take 10 to 15 minutes, while the model downloads and CUDA graphs are captured.

### AxonHighErrorRate
More than 5% of `/ask` and `/rerank` requests return 5xx for 10 minutes. Split it by status on the
"Requests by route and status" panel:
* `502`: the LLM call failed. See LLMDown and the `"event": "ask_failed"` log lines, whose `error` field
  gives the exception type.
* `503`: either back-pressure (see AxonAskSaturated) or a configuration refusal (key file missing or
  malformed). The response body says which.
* `500`: a bug. Collect the request IDs and the logs.

### AxonAskSaturated
`/ask` has been at `AXON_MAX_INFLIGHT` for 10 minutes and is refusing requests. Steps:
* If the LLM queue is also long (LLMQueueBacklog), the GPU is the limit: add an LLM replica, use a
  smaller or quantized model, or shorten `max_tokens`.
* If the LLM is idle, raise `AXON_MAX_INFLIGHT`.

### AxonAskSlow
p95 `/ask` above 20 s. Look at "Pipeline stage p95" to find which stage is slow.
* `generate` slow: it is the LLM (see RAGTimeToFirstTokenHigh).
* `first_stage` or `rerank` slow: see RetrievalSlow.

### RAGTimeToFirstTokenHigh
p95 time to first token above 5 s. Compare the server-side TTFT panel with Axon's client-side TTFT:
* Server TTFT high too: the time goes to LLM queueing or prefill. Reduce concurrency or prompt size
  (`k`, the chunk size), or add capacity.
* Server TTFT low but Axon's high: the time is lost between Axon and the LLM (network, or a proxy).

### RetrievalSlow
p95 first stage or re-rank above 1 s.
* First stage: exact search grows linearly with the index. Rebuild with `--plaid`, or give Axon a GPU.
* Re-rank: check CPU throttling (`docker stats`). Candidate depth is fixed at 100.

### TenantFilterViolationBlocked
A result from another tenant got through retrieval and was caught by the final check before it was
returned. Nothing leaked, but a filter upstream failed, and that is a bug. Steps:
* Keep the logs and the index version.
* Run `pytest tests -k tenant` against a copy of the index.
* Until it is fixed, consider separate indexes per tenant.

### UncitedAnswersHigh
Over 10% of answers have no valid citation and are withheld ("no citation, no answer"). This usually
means the model or prompt changed: a new LLM image, a model without the fine-tune, or `max_tokens`
cutting answers short. In round 5, every uncited answer at 96 tokens turned out to be truncation. Check
`finish_reason` on a sample of requests.

### InjectionAttemptsSpike
The injection scanner quarantined over 50 retrieved sources in an hour. Find which documents were
recently added (`git log`, or the `update` output) and review them; one poisoned file can match many
queries. Quarantined sources are never shown to the LLM.

### IndexReloadFailing
Axon saw a new index version but could not load it, so it keeps serving the previous version.
Usual causes:
* An `update` crashed half-way. Run `update` again: it is idempotent.
* The disk is full.
* The index was rebuilt with a different model.

### IndexNeedsCompaction
Over 30% of the index is deleted chunks. They cost memory and search time but are never returned.
Run `python -m nyaya.rag compact --index deploy/index` at a quiet time. It renumbers chunks and rebuilds
PLAID, and Axon picks up the result live.

### AxonMemoryGrowing
Resident memory has grown by over about 70 MB an hour for 3 hours. Expected jumps:
* Index reloads: the new snapshot is built before the old one is released.
* The rate-limit table: bounded at 100,000 clients.

A steady climb with neither is a leak. Note the request mix, then restart Axon (`docker compose
restart axon`; the index is on disk, nothing is lost).

### LLMQueueBacklog
Over 16 requests are waiting inside the LLM server. Either lower `AXON_MAX_INFLIGHT`, so that Axon sheds
load with a fast 503 instead of slow answers, or add GPU capacity.
