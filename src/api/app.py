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
from fastapi.responses import JSONResponse

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

# ─── Prometheus instrumentation ───────────────────────────────────────
try:
    from prometheus_fastapi_instrumentator import Instrumentator  # type: ignore
    Instrumentator().instrument(app).expose(app, endpoint="/metrics", include_in_schema=False)
except ImportError:  # pragma: no cover
    logger.warning("prometheus_fastapi_instrumentator not installed; /metrics disabled")


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
def root():
    return {
        "service": API_TITLE,
        "version": API_VERSION,
        "model_loaded": _model is not None,
        "endpoints": ["/health", "/metrics", "/predict", "/predict/batch", "/docs"],
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


@app.post("/predict", response_model=PredictionResponse)
def predict(features: HeartFeatures):
    result = _predict_one(features)
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
    logger.info("batch_prediction", extra={"count": len(preds)})
    return BatchResponse(predictions=preds, count=len(preds))


# Allow `python -m src.api.app` for quick local dev
if __name__ == "__main__":  # pragma: no cover
    import uvicorn
    uvicorn.run("src.api.app:app", host="0.0.0.0",
                port=int(os.getenv("API_PORT", "8080")), reload=False)
