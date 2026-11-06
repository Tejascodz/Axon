# Axon API server (retrieval, re-ranking, guardrails, /ask). The LLM runs in its own container.
#
#   docker build -f deploy/Dockerfile -t axon:1.0 .
#   docker build -f deploy/Dockerfile --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cu126 -t axon:1.0-gpu .
#
# Dependencies: deploy/overrides.txt lifts ujson to its fixed release; the other advisories in
# PyLate's pins cannot be lifted without breaking the model (tested; see that file and RUNBOOK.md).
# --build-arg PYLATE_PINS=1 installs PyLate's own pins exactly.
# For production, pin the base image by digest (python:3.12-slim@sha256:...) so rebuilds are identical.
FROM python:3.12-slim

ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu
ARG PYLATE_PINS=0
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 \
    UV_NO_CACHE=1 HOME=/data XDG_CACHE_HOME=/data/cache HF_HOME=/data/hf \
    AXON_PRELOAD=1 AXON_REQUIRE_AUTH=1 MALLOC_ARENA_MAX=2

WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY nyaya ./nyaya
COPY deploy/overrides.txt /tmp/overrides.txt
RUN pip install "uv==0.13.7" \
 && uv pip install --system --index-url "$TORCH_INDEX" "torch==2.11.*" \
 && if [ "$PYLATE_PINS" = "1" ]; then uv pip install --system ".[server]" "pypdf>=5"; \
    else uv pip install --system --override /tmp/overrides.txt ".[server]" "pypdf>=5"; fi \
 && pip uninstall -y uv \
 && rm -f /tmp/overrides.txt \
 && useradd --uid 10001 --no-create-home --shell /usr/sbin/nologin axon \
 && mkdir -p /data && chown 10001 /data

USER 10001
EXPOSE 8080
# readiness: the model (and index, if AXON_INDEX is set) is loaded; the first start downloads the model
HEALTHCHECK --interval=30s --timeout=5s --start-period=600s --retries=3 \
  CMD ["python", "-c", "import sys, urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8080/ready', timeout=4).status == 200 else 1)"]
CMD ["uvicorn", "nyaya.server:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1", \
     "--timeout-graceful-shutdown", "30", "--no-server-header"]
