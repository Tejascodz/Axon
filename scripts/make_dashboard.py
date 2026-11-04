"""Writes deploy/grafana/dashboards/axon.json (run after changing a panel; CI checks it is current)."""
import json
import os

DS = {"type": "prometheus", "uid": "axon-prom"}
LLM = '{__name__=~"(vllm|sglang):%s"}'
panels, y = [], 0


def row(title):
    global y
    panels.append({"type": "row", "title": title, "collapsed": False, "gridPos": {"h": 1, "w": 24, "x": 0, "y": y},
                   "id": len(panels) + 1, "panels": []})
    y += 1


def ts(title, targets, unit="short", w=8, x=0, desc=""):
    panels.append({"type": "timeseries", "title": title, "description": desc, "datasource": DS,
                   "id": len(panels) + 1, "gridPos": {"h": 8, "w": w, "x": x, "y": y},
                   "fieldConfig": {"defaults": {"unit": unit, "custom": {"lineWidth": 1, "fillOpacity": 0}}, "overrides": []},
                   "options": {"legend": {"displayMode": "list", "placement": "bottom"}, "tooltip": {"mode": "multi"}},
                   "targets": [{"datasource": DS, "expr": e, "legendFormat": leg, "refId": chr(65 + i)}
                               for i, (e, leg) in enumerate(targets)]})


def stat(title, expr, unit="short", w=4, x=0):
    panels.append({"type": "stat", "title": title, "datasource": DS, "id": len(panels) + 1,
                   "gridPos": {"h": 4, "w": w, "x": x, "y": y},
                   "fieldConfig": {"defaults": {"unit": unit}, "overrides": []},
                   "options": {"reduceOptions": {"calcs": ["lastNotNull"]}, "colorMode": "value", "graphMode": "none"},
                   "targets": [{"datasource": DS, "expr": expr, "refId": "A"}]})


def q(p, metric, extra=""):
    return f'histogram_quantile({p}, sum by (le{extra}) (rate({metric}[$__rate_interval])))'


row("Overview")
stat("Ready", 'max(axon_ready)', x=0)
stat("Requests / s", 'sum(rate(axon_requests_total{path=~"/ask|/rerank"}[$__rate_interval]))', "reqps", x=4)
stat("5xx share", 'sum(rate(axon_requests_total{path=~"/ask|/rerank",status=~"5.."}[$__rate_interval])) / '
     'sum(rate(axon_requests_total{path=~"/ask|/rerank"}[$__rate_interval]))', "percentunit", x=8)
stat("/ask in flight", 'max(axon_ask_inflight)', x=12)
stat("Index version", 'max(axon_index_version)', x=16)
stat("Live chunks", 'max(axon_index_live_chunks)', x=20)
y += 4

row("Axon API")
ts("Requests by route and status", [('sum by (path, status) (rate(axon_requests_total[$__rate_interval]))',
                                     "{{path}} {{status}}")], "reqps", x=0)
ts("/ask latency", [(q(0.5, 'axon_request_seconds_bucket{path="/ask"}'), "p50"),
                    (q(0.95, 'axon_request_seconds_bucket{path="/ask"}'), "p95"),
                    (q(0.99, 'axon_request_seconds_bucket{path="/ask"}'), "p99")], "s", x=8)
ts("/rerank latency", [(q(0.5, 'axon_request_seconds_bucket{path="/rerank"}'), "p50"),
                       (q(0.95, 'axon_request_seconds_bucket{path="/rerank"}'), "p95"),
                       (q(0.99, 'axon_request_seconds_bucket{path="/rerank"}'), "p99")], "s", x=16)
y += 8
ts("Pipeline stage p95", [(q(0.95, "rag_stage_seconds_bucket", ", stage"), "{{stage}}")], "s", x=0,
   desc="first_stage = BM25 + ColBERT/PLAID, rerank = Axon, generate = LLM call")
ts("Back-pressure", [("max(axon_ask_inflight)", "in flight"), ("max(axon_ask_capacity)", "capacity"),
                     ("sum(rate(axon_ask_rejected_total[$__rate_interval]))", "rejected / s")], x=8)
ts("Axon memory", [('process_resident_memory_bytes{job="axon"}', "resident")], "bytes", x=16)
y += 8

row("Generation (measured by Axon, client side)")
ts("Time to first token", [(q(0.5, "rag_time_to_first_token_seconds_bucket"), "p50"),
                           (q(0.95, "rag_time_to_first_token_seconds_bucket"), "p95")], "s", x=0)
ts("Inter-token latency", [(q(0.5, "rag_inter_token_latency_seconds_bucket"), "p50"),
                           (q(0.99, "rag_inter_token_latency_seconds_bucket"), "p99")], "s", x=8)
ts("Answers and guardrails", [('sum by (event) (rate(rag_events_total[$__rate_interval]))', "{{event}}")],
   "ops", x=16, desc="answered, abstained, uncited_answer_withheld, source_withheld (injection scanner), ...")
y += 8

row("LLM server (vLLM or SGLang)")
ts("Server TTFT", [(q(0.5, LLM % "time_to_first_token_seconds_bucket"), "p50"),
                   (q(0.95, LLM % "time_to_first_token_seconds_bucket"), "p95")], "s", x=0)
ts("Server inter-token latency", [(q(0.5, LLM % "inter_token_latency_seconds_bucket"), "p50"),
                                  (q(0.99, LLM % "inter_token_latency_seconds_bucket"), "p99")], "s", x=8)
ts("Tokens / s", [(f'sum(rate({LLM % "generation_tokens_total"}[$__rate_interval]))', "generated"),
                  (f'sum(rate({LLM % "prompt_tokens_total"}[$__rate_interval]))', "prompt")], "short", x=16)
y += 8
ts("Running and queued requests", [("max(vllm:num_requests_running) or max(sglang:num_running_reqs)", "running"),
                                   ("max(vllm:num_requests_waiting) or max(sglang:num_queue_reqs)", "queued")], x=0,
   desc="gauge names differ between versions: check the server's /metrics if empty")
ts("End-to-end request latency (server)", [(q(0.5, LLM % "e2e_request_latency_seconds_bucket"), "p50"),
                                           (q(0.95, LLM % "e2e_request_latency_seconds_bucket"), "p95")], "s", x=8)
ts("Index health", [("max(axon_index_live_chunks)", "live"), ("max(axon_index_dead_chunks)", "deleted"),
                    ('sum(increase(rag_events_total{event=~"index_reloaded|index_reload_failed"}[$__rate_interval]))',
                     "reloads")], x=16)
y += 8

dash = {"uid": "axon-overview", "title": "Axon", "tags": ["axon", "rag"], "timezone": "utc", "schemaVersion": 39,
        "version": 1, "editable": False, "refresh": "30s", "time": {"from": "now-6h", "to": "now"},
        "templating": {"list": []}, "annotations": {"list": []}, "panels": panels}
out = os.path.join(os.path.dirname(__file__), "..", "deploy", "grafana", "dashboards", "axon.json")
with open(out, "w") as f:
    json.dump(dash, f, indent=1)
    f.write("\n")
print(f"{len(panels)} panels -> {os.path.normpath(out)}")
