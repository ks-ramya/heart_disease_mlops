###############################################################################
# Heart Disease Prediction API — multi-stage Docker build
#
# Build:
#   docker build -t heart-disease-api:latest .
# Run:
#   docker run --rm -p 8080:8080 heart-disease-api:latest
###############################################################################

# ─── Stage 1: build deps + train model ────────────────────────────────
FROM python:3.10-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY src ./src
COPY data ./data

# Train model inside the image so the artifact is self-contained.
# Skip MLflow remote tracking by using local file backend (default).
RUN python -m src.data.download && python -m src.models.train

# ─── Stage 2: lean runtime image ──────────────────────────────────────
FROM python:3.10-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8080 \
    MODEL_PATH=/app/models/heart_disease_model.pkl

WORKDIR /app

# Copy installed packages from the builder stage
COPY --from=builder /usr/local/lib/python3.10/site-packages \
                    /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy source + trained model artifact
COPY --from=builder /build/src ./src
COPY --from=builder /build/models ./models

# Drop privileges
RUN groupadd -r app && useradd -r -g app app && chown -R app:app /app
USER app

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request,sys; \
        sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8080/health',timeout=3).status==200 else 1)"

CMD ["gunicorn", "src.api.app:app", \
     "--bind", "0.0.0.0:8080", \
     "--workers", "2", \
     "--worker-class", "uvicorn.workers.UvicornWorker", \
     "--timeout", "120", \
     "--access-logfile", "-"]
