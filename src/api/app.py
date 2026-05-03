"""FastAPI service for Heart Disease prediction.

Endpoints:
    GET  /              -> service info
    GET  /health        -> liveness/readiness probe
    GET  /metrics       -> Prometheus metrics (text)
    POST /predict       -> single prediction + confidence
    POST /predict/batch -> batch predictions

Model is loaded lazily at import time from ``MODEL_PATH`` (env-overridable).
Logs are emitted as structured JSON for observability.
"""

from __future__ import annotations

import os
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from src.api.logging_config import configure_logging
from src.api.schemas import (
    BatchRequest,
    BatchResponse,
    HealthResponse,
    HeartFeatures,
    PredictionResponse,
)
from src.config import API_TITLE, API_VERSION, MODEL_PATH_ENV

logger = configure_logging()
_model = None
_model_path = Path(MODEL_PATH_ENV)


def _load_model():
    global _model
    if not _model_path.exists():
        logger.warning("Model file missing", extra={"path": str(_model_path)})
        return None
    try:
        m = joblib.load(_model_path)
        logger.info("Model loaded", extra={"path": str(_model_path)})
        return m
    except Exception as exc:  # pragma: no cover
        logger.error("Model load failed", extra={"error": str(exc)})
        return None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model
    _model = _load_model()
    yield


app = FastAPI(title=API_TITLE, version=API_VERSION, lifespan=lifespan)

# ─── Static UI (served at /ui) ────────────────────────────────────────
_STATIC_DIR = Path(__file__).resolve().parent / "static"
if _STATIC_DIR.is_dir():
    app.mount("/ui", StaticFiles(directory=str(_STATIC_DIR), html=True), name="ui")

# ─── Prometheus instrumentation ───────────────────────────────────────
try:
    from prometheus_fastapi_instrumentator import Instrumentator  # type: ignore
    Instrumentator().instrument(app).expose(app, endpoint="/metrics", include_in_schema=False)
except ImportError:  # pragma: no cover
    logger.warning("prometheus_fastapi_instrumentator not installed; /metrics disabled")

# Custom model-level metrics (in addition to default HTTP histograms).
# Tolerate duplicate registration on module re-import (test fixtures reload src.*).
try:
    from prometheus_client import Counter, Histogram, REGISTRY  # type: ignore

    def _get_or_create(factory, name: str):
        existing = getattr(REGISTRY, "_names_to_collectors", {}).get(name)
        if existing is not None:
            return existing
        return factory()

    PREDICTIONS_TOTAL = _get_or_create(
        lambda: Counter(
            "predictions_total",
            "Number of /predict responses, labelled by predicted class.",
            ["label"],
        ),
        "predictions_total",
    )
    PREDICTION_CONFIDENCE = _get_or_create(
        lambda: Histogram(
            "prediction_confidence",
            "Confidence (max class probability) of /predict responses.",
            buckets=(0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0),
        ),
        "prediction_confidence",
    )
except ImportError:  # pragma: no cover
    PREDICTIONS_TOTAL = None
    PREDICTION_CONFIDENCE = None


@app.middleware("http")
async def request_logger(request: Request, call_next):
    """Per-request structured access log with latency + correlation id."""
    rid = request.headers.get("x-request-id") or str(uuid.uuid4())
    t0 = time.perf_counter()
    try:
        response = await call_next(request)
        return response
    finally:
        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.info(
            "request",
            extra={
                "request_id": rid,
                "method": request.method,
                "path": request.url.path,
                "client": request.client.host if request.client else None,
                "duration_ms": round(elapsed_ms, 2),
            },
        )


def _predict_one(features: HeartFeatures) -> PredictionResponse:
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    df = pd.DataFrame([features.model_dump()])
    pred_int = int(_model.predict(df)[0])
    proba = _model.predict_proba(df)[0]
    classes = [int(c) for c in _model.classes_]
    proba_map = {("disease" if c == 1 else "no_disease"): float(p)
                 for c, p in zip(classes, proba)}
    confidence = float(max(proba))
    label = "disease" if pred_int == 1 else "no_disease"
    return PredictionResponse(
        prediction=pred_int, label=label,
        confidence=confidence, probabilities=proba_map,
    )


# ─── Routes ───────────────────────────────────────────────────────────
@app.get("/")
def root(request: Request):
    """Serve the UI to browsers; return JSON metadata to API clients."""
    accept = request.headers.get("accept", "")
    if _STATIC_DIR.is_dir() and "text/html" in accept:
        return RedirectResponse(url="/ui/", status_code=307)
    return {
        "service": API_TITLE,
        "version": API_VERSION,
        "model_loaded": _model is not None,
        "ui": "/ui/" if _STATIC_DIR.is_dir() else None,
        "endpoints": ["/health", "/metrics", "/predict", "/predict/batch", "/docs", "/ui/"],
    }


@app.get("/health", response_model=HealthResponse)
def health():
    if _model is None:
        return JSONResponse(
            status_code=503,
            content=HealthResponse(
                status="unhealthy", model_loaded=False,
                model_path=str(_model_path), api_version=API_VERSION,
            ).model_dump(),
        )
    return HealthResponse(
        status="healthy", model_loaded=True,
        model_path=str(_model_path), api_version=API_VERSION,
    )


def _record_prediction_metrics(result: PredictionResponse) -> None:
    if PREDICTIONS_TOTAL is not None:
        PREDICTIONS_TOTAL.labels(label=result.label).inc()
    if PREDICTION_CONFIDENCE is not None:
        PREDICTION_CONFIDENCE.observe(result.confidence)


@app.post("/predict", response_model=PredictionResponse)
def predict(features: HeartFeatures):
    result = _predict_one(features)
    _record_prediction_metrics(result)
    logger.info("prediction", extra={
        "prediction": result.prediction,
        "label": result.label,
        "confidence": result.confidence,
    })
    return result


@app.post("/predict/batch", response_model=BatchResponse)
def predict_batch(req: BatchRequest):
    if not req.instances:
        raise HTTPException(status_code=400, detail="`instances` must not be empty")
    preds = [_predict_one(f) for f in req.instances]
    for p in preds:
        _record_prediction_metrics(p)
    logger.info("batch_prediction", extra={"count": len(preds)})
    return BatchResponse(predictions=preds, count=len(preds))


# Allow `python -m src.api.app` for quick local dev
if __name__ == "__main__":  # pragma: no cover
    import uvicorn
    uvicorn.run("src.api.app:app", host="0.0.0.0",
                port=int(os.getenv("API_PORT", "8080")), reload=False)
