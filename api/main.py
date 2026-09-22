"""FastAPI inference service.

Exposes the trained model through a versioned REST API with a health probe,
request logging, basic input validation and Prometheus-style latency metrics.
This is the "integrate the model through an API" and "deployment/monitoring"
piece of the pipeline.

Run locally:
    export MODEL_CHECKPOINT=artifacts/densenet121_baseline/best_model.pt
    uvicorn api.main:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import os
import time

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from src.chestxray import __version__
from src.chestxray.inference import get_predictor
from src.chestxray.utils import get_logger

log = get_logger("api")

CHECKPOINT = os.environ.get("MODEL_CHECKPOINT", "artifacts/densenet121_baseline/best_model.pt")
IMAGE_SIZE = int(os.environ.get("IMAGE_SIZE", "224"))
MAX_BYTES = int(os.environ.get("MAX_UPLOAD_BYTES", str(10 * 1024 * 1024)))  # 10 MB
ALLOWED_TYPES = {"image/png", "image/jpeg"}

app = FastAPI(
    title="ChestX-ray CAD API",
    description="Multi-label chest X-ray pathology classifier.",
    version=__version__,
)

# Lightweight in-memory metrics (scrape-able; swap for prometheus_client in prod).
_METRICS = {"requests": 0, "errors": 0, "latency_ms_sum": 0.0}


def _predictor():
    try:
        return get_predictor(CHECKPOINT, IMAGE_SIZE)
    except FileNotFoundError:
        raise HTTPException(status_code=503, detail="Model checkpoint not available.")


@app.get("/health")
def health() -> dict:
    loaded = os.path.exists(CHECKPOINT)
    return {"status": "ok" if loaded else "degraded",
            "model_loaded": loaded, "model_version": __version__}


@app.get("/metrics")
def metrics() -> dict:
    avg = (_METRICS["latency_ms_sum"] / _METRICS["requests"]) if _METRICS["requests"] else 0.0
    return {**_METRICS, "latency_ms_avg": round(avg, 2)}


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> JSONResponse:
    _METRICS["requests"] += 1
    if file.content_type not in ALLOWED_TYPES:
        _METRICS["errors"] += 1
        raise HTTPException(status_code=415, detail=f"Unsupported type: {file.content_type}")

    raw = await file.read()
    if len(raw) > MAX_BYTES:
        _METRICS["errors"] += 1
        raise HTTPException(status_code=413, detail="File too large.")

    t0 = time.perf_counter()
    try:
        probs = _predictor().predict_from_bytes(raw)
    except ValueError as exc:
        _METRICS["errors"] += 1
        raise HTTPException(status_code=400, detail=str(exc))
    elapsed = (time.perf_counter() - t0) * 1000
    _METRICS["latency_ms_sum"] += elapsed

    ranked = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)
    log.info("predict ok | %d labels | %.1f ms", len(ranked), elapsed)
    return JSONResponse({
        "predictions": [{"label": k, "probability": round(v, 4)} for k, v in ranked],
        "model_version": __version__,
        "inference_ms": round(elapsed, 2),
    })
