# ---------------------------------------------------------------------------------------------
# Production inference image: ONNX Runtime + FastAPI only (no PyTorch -> ~350 MB instead of ~2 GB)
#   docker build -f docker/Dockerfile -t defect-detection-api:1.0.0 .
#   docker run --rm -p 8000:8000 defect-detection-api:1.0.0
# ---------------------------------------------------------------------------------------------
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

COPY requirements/serve.txt /tmp/serve.txt
RUN pip install -r /tmp/serve.txt

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-deps . && rm -rf src

# The model is baked into the image so that image tag == model version (immutable, rollback = redeploy
# the previous tag). Override MODEL_DIR at build time to ship another model, or mount a volume and
# set DD_MODEL_DIR at run time.
ARG MODEL_DIR=models/efficientnet_b0
COPY ${MODEL_DIR} /app/model

RUN useradd --create-home --uid 10001 appuser && chown -R appuser /app
USER appuser

ENV DD_MODEL_DIR=/app/model \
    DD_LOG_JSON=true \
    DD_LOG_LEVEL=INFO \
    DD_INTRA_OP_THREADS=2 \
    WEB_CONCURRENCY=2 \
    PORT=8000

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import os,urllib.request,sys; sys.exit(0 if urllib.request.urlopen(f'http://127.0.0.1:{os.environ.get(\"PORT\",\"8000\")}/ready', timeout=4).status == 200 else 1)"

# Each uvicorn worker is a separate process with its own ORT session; size
# WEB_CONCURRENCY x DD_INTRA_OP_THREADS to the number of CPU cores of the node.
CMD ["sh", "-c", "exec uvicorn defect_detection.api.main:app --host 0.0.0.0 --port ${PORT} --workers ${WEB_CONCURRENCY} --no-access-log"]
