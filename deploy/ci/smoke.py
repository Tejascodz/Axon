"""CI: checks the running Compose stack end to end. Exits non-zero on the first failure.

    python deploy/ci/smoke.py --secrets deploy/secrets
"""
import argparse
import base64
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

AXON, PROM, GRAFANA = "http://127.0.0.1:8080", "http://127.0.0.1:9090", "http://127.0.0.1:3000"


def call(url, method="GET", body=None, headers=None, timeout=60):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method,
                                 headers={"Content-Type": "application/json", **(headers or {})})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:  # local stack only  # nosec B310
            raw = r.read()
            return r.status, dict(r.headers), raw
    except urllib.error.HTTPError as e:
        return e.code, dict(e.headers), e.read()
    except (urllib.error.URLError, ConnectionError, TimeoutError):
        return None, {}, b""


def check(ok, what):
    print(("PASS " if ok else "FAIL ") + what, flush=True)
    if not ok:
        sys.exit(1)


def wait(what, fn, seconds):
    t0 = time.time()
    while time.time() - t0 < seconds:
        if fn():
            print(f"PASS {what} ({time.time() - t0:.0f} s)", flush=True)
            return
        time.sleep(3)
    check(False, f"{what} within {seconds} s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--secrets", required=True)
    a = ap.parse_args()
    key = open(f"{a.secrets}/axon_api_key").read().strip()
    mkey = open(f"{a.secrets}/axon_metrics_key").read().strip()
    gpw = open(f"{a.secrets}/grafana_admin_password").read().strip()
    client, scraper = {"Authorization": f"Bearer {key}"}, {"Authorization": f"Bearer {mkey}"}

    wait("Axon ready (model and index loaded)", lambda: call(f"{AXON}/ready")[0] == 200, 1200)

    # authentication
    check(call(f"{AXON}/health")[0] == 200, "/health needs no key")
    check(call(f"{AXON}/rerank", "POST", {"query": "q", "documents": ["d"]})[0] == 401, "/rerank without a key: 401")
    check(call(f"{AXON}/rerank", "POST", {"query": "q", "documents": ["d"]}, scraper)[0] == 401,
          "the metrics key cannot call /rerank")
    check(call(f"{AXON}/metrics", headers=client)[0] == 401, "a client key cannot read /metrics")

    # re-ranking with the real model
    docs = ["Hiking boots with a steel shank.", "A waterproof trail jacket with an adjustable hood.",
            "Returns are accepted within 30 days."]
    st, hd, raw = call(f"{AXON}/rerank", "POST", {"query": "waterproof jacket with a hood", "documents": docs},
                       {**client, "X-Request-ID": "ci-rerank-1"})
    res = json.loads(raw)["results"] if st == 200 else []
    check(st == 200 and res and res[0]["index"] == 1, f"/rerank puts the jacket first ({[r['index'] for r in res]})")
    check(hd.get("x-request-id") == "ci-rerank-1", "X-Request-ID is echoed")

    # full RAG: retrieval, guardrails, streamed generation, citation check
    q = "How heavy is the Ridgeline trail jacket?"
    st, hd, raw = call(f"{AXON}/ask", "POST", {"question": q, "k": 5}, {**client, "X-Request-ID": "ci-ask-1"})
    out = json.loads(raw) if st == 200 else {}
    check(st == 200, f"/ask answers ({st}: {raw[:200]!r})")
    check(out.get("request_id") == "ci-ask-1", "/ask returns its request ID")
    check(out["sources"] and out["sources"][0]["source"] == "jackets.md",
          f"the jacket document is retrieved first ({[s['source'] for s in out['sources']]})")
    check(not out["abstained"] and out["check"]["cited"], f"the answer cites a source ({out['check']['cited']})")
    check("410 grams" in out["text"], "the answer comes from the retrieved source")
    check(out["timing"]["ttft"] is not None, "time to first token is measured")

    st, _, raw = call(f"{AXON}/ask", "POST", {"question": "What is the shipping update for stoves?"}, client)
    out = json.loads(raw) if st == 200 else {}
    withheld = [w["title"] for w in out.get("withheld", [])]
    check(st == 200 and withheld == ["notice.md"], f"the injected document is quarantined, never shown to the LLM ({withheld})")
    check("PWNED" not in out.get("text", ""), "the injection does not reach the answer")

    st, _, raw = call(f"{AXON}/metrics", headers=scraper)
    page = raw.decode()
    check(st == 200 and 'rag_events_total{event="answered"}' in page and "axon_ready 1.0" in page,
          "the metrics key reads /metrics with RAG events")
    check('rag_events_total{event="source_withheld"}' in page, "quarantined sources are counted")

    # Prometheus
    def targets_up():
        st, _, raw = call(f"{PROM}/api/v1/targets")
        if st != 200:
            return False
        up = {t["labels"]["job"]: t["health"] for t in json.loads(raw)["data"]["activeTargets"]}
        return up.get("axon") == "up" and up.get("llm") == "up"
    wait("Prometheus scrapes Axon (with the metrics key) and the LLM", targets_up, 120)

    def ready_series():
        st, _, raw = call(f"{PROM}/api/v1/query?" + urllib.parse.urlencode({"query": "axon_ready"}))
        res = json.loads(raw)["data"]["result"] if st == 200 else []
        return bool(res) and res[0]["value"][1] == "1"
    wait("axon_ready == 1 in Prometheus", ready_series, 60)
    st, _, raw = call(f"{PROM}/api/v1/rules")
    rules = [r for g in json.loads(raw)["data"]["groups"] for r in g["rules"]] if st == 200 else []
    check(len(rules) == 15 and all(r["health"] == "ok" for r in rules), f"15 alert rules loaded and healthy ({len(rules)})")

    # Grafana
    auth = {"Authorization": "Basic " + base64.b64encode(f"admin:{gpw}".encode()).decode()}
    wait("Grafana up", lambda: call(f"{GRAFANA}/api/health")[0] == 200, 120)
    check(call(f"{GRAFANA}/api/dashboards/uid/axon-overview")[0] in (401, 403), "Grafana requires a login")
    st, _, raw = call(f"{GRAFANA}/api/dashboards/uid/axon-overview", headers=auth)
    check(st == 200 and len(json.loads(raw)["dashboard"]["panels"]) == 25, "the Axon dashboard is provisioned")
    st, _, raw = call(f"{GRAFANA}/api/datasources/uid/axon-prom/health", headers=auth)
    check(st == 200 and json.loads(raw).get("status") == "OK", f"Grafana reaches Prometheus ({raw[:120]!r})")
    print("all checks passed")


if __name__ == "__main__":
    main()
