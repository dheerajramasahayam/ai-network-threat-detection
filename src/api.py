"""
api.py
------
FastAPI REST endpoint for real-time threat detection.

Endpoints:
  POST /detect        - Classify a single network flow
  POST /detect/batch  - Classify a batch of flows
  GET  /health        - Health check
  GET  /model/info    - Loaded model metadata
  GET  /metrics       - Running detection statistics

Run with:
    uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

import os
import sys
import time
import glob
import logging
from collections import deque, Counter
from contextlib import asynccontextmanager
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from src.model import IntrusionDetectionModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# State (loaded at startup)
# ──────────────────────────────────────────────────────────────────────────────

class AppState:
    model: IntrusionDetectionModel | None = None
    model_path: str = ""
    start_time: float = time.time()
    recent_results: deque = deque(maxlen=500)   # last 500 detections
    counters: Counter = Counter()

state = AppState()


def _find_latest_model() -> str:
    """Auto-discover the most recently modified .joblib in models/."""
    pattern = os.path.join(BASE_DIR, 'models', '*.joblib')
    files = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)
    if not files:
        raise FileNotFoundError(
            f"No trained model found in {os.path.join(BASE_DIR, 'models')}. "
            "Run src/training.py first."
        )
    return files[0]


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── startup ───────────────────────────────────────────────────────
    try:
        model_path = os.environ.get('MODEL_PATH') or _find_latest_model()
        state.model = IntrusionDetectionModel.load(model_path)
        state.model_path = model_path
        logger.info(f"Model loaded: {model_path}")
    except FileNotFoundError as exc:
        logger.warning(str(exc))
        logger.warning("API will start in degraded mode — train a model first.")
    yield
    # ── shutdown ──────────────────────────────────────────────────────
    logger.info("API shutting down.")


# ──────────────────────────────────────────────────────────────────────────────
# App
# ──────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="AI Network Intrusion Detection API",
    description=(
        "Real-time threat classification for network flows. "
        "Powered by ensemble ML (RandomForest + XGBoost) trained on CICIDS2017."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ──────────────────────────────────────────────────────────────────────────────
# Request / Response schemas
# ──────────────────────────────────────────────────────────────────────────────

class FlowFeatures(BaseModel):
    """Feature vector for a single bidirectional IP flow."""
    features: dict[str, float] | list[float] = Field(
        ...,
        description=(
            "Either a dict mapping feature names → values, "
            "or a raw list of floats matching the model's feature order."
        ),
        examples=[{"Flow Bytes/s": 12345.6, "SYN Flag Count": 15.0}],
    )
    src_ip: str | None = Field(None, description="Source IP (for logging only)")
    dst_ip: str | None = Field(None, description="Destination IP (for logging only)")
    dst_port: int | None = Field(None, description="Destination port (for logging only)")
    threshold: float = Field(0.5, ge=0.0, le=1.0, description="Decision threshold")


class BatchFlowFeatures(BaseModel):
    flows: list[FlowFeatures]


class DetectionResponse(BaseModel):
    label: str
    is_attack: bool
    attack_probability: float
    confidence: float
    model_type: str
    latency_ms: float
    threshold_used: float
    metadata: dict[str, Any] = {}


class BatchDetectionResponse(BaseModel):
    results: list[DetectionResponse]
    total_flows: int
    attacks_detected: int
    benign_flows: int
    total_latency_ms: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_type: str | None
    uptime_seconds: float


class ModelInfoResponse(BaseModel):
    model_type: str
    model_path: str
    is_trained: bool
    feature_count: int
    feature_names: list[str]


class MetricsResponse(BaseModel):
    total_inspected: int
    attacks_detected: int
    benign_flows: int
    attack_rate_pct: float
    uptime_seconds: float


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _require_model():
    if state.model is None:
        raise HTTPException(
            status_code=503,
            detail="No model loaded. Run src/training.py to train models first.",
        )


def _classify(flow: FlowFeatures) -> DetectionResponse:
    _require_model()
    features = flow.features
    if isinstance(features, dict):
        fn = state.model.feature_names
        arr = np.array([[features.get(f, 0.0) for f in fn]]) if fn \
              else np.array([list(features.values())])
    else:
        arr = np.array([features])

    t0 = time.perf_counter()
    attack_prob = float(state.model.predict_proba(arr)[0])
    latency_ms = (time.perf_counter() - t0) * 1000

    is_attack = attack_prob >= flow.threshold
    label = "ATTACK" if is_attack else "BENIGN"
    confidence = attack_prob if is_attack else 1.0 - attack_prob

    # Update running counters
    state.counters["total"] += 1
    state.counters["attacks" if is_attack else "benign"] += 1

    result = DetectionResponse(
        label=label,
        is_attack=is_attack,
        attack_probability=round(attack_prob, 6),
        confidence=round(confidence, 6),
        model_type=state.model.model_type,
        latency_ms=round(latency_ms, 3),
        threshold_used=flow.threshold,
        metadata={
            k: v for k, v in [
                ("src_ip", flow.src_ip),
                ("dst_ip", flow.dst_ip),
                ("dst_port", flow.dst_port),
            ] if v is not None
        },
    )
    state.recent_results.append(result)
    return result


# ──────────────────────────────────────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["System"])
def health():
    return HealthResponse(
        status="ok" if state.model else "degraded",
        model_loaded=state.model is not None,
        model_type=state.model.model_type if state.model else None,
        uptime_seconds=round(time.time() - state.start_time, 1),
    )


@app.get("/model/info", response_model=ModelInfoResponse, tags=["System"])
def model_info():
    _require_model()
    return ModelInfoResponse(
        model_type=state.model.model_type,
        model_path=state.model_path,
        is_trained=state.model.is_trained,
        feature_count=len(state.model.feature_names),
        feature_names=state.model.feature_names,
    )


@app.get("/metrics", response_model=MetricsResponse, tags=["System"])
def metrics():
    total = state.counters.get("total", 0)
    attacks = state.counters.get("attacks", 0)
    benign = state.counters.get("benign", 0)
    return MetricsResponse(
        total_inspected=total,
        attacks_detected=attacks,
        benign_flows=benign,
        attack_rate_pct=round(attacks / total * 100, 2) if total else 0.0,
        uptime_seconds=round(time.time() - state.start_time, 1),
    )


@app.post("/detect", response_model=DetectionResponse, tags=["Detection"])
def detect(flow: FlowFeatures):
    """
    Classify a **single** network flow as BENIGN or ATTACK.

    Pass either a feature dict `{"Flow Bytes/s": 12345, ...}` or a raw list
    of floats matching the model's feature order.
    """
    return _classify(flow)


@app.post("/detect/batch", response_model=BatchDetectionResponse, tags=["Detection"])
def detect_batch(batch: BatchFlowFeatures):
    """Classify a **batch** of flows in one request."""
    if not batch.flows:
        raise HTTPException(status_code=400, detail="Batch must not be empty.")
    if len(batch.flows) > 10_000:
        raise HTTPException(status_code=400, detail="Max batch size is 10,000 flows.")

    t0 = time.perf_counter()
    results = [_classify(f) for f in batch.flows]
    total_ms = (time.perf_counter() - t0) * 1000

    attacks = sum(r.is_attack for r in results)
    return BatchDetectionResponse(
        results=results,
        total_flows=len(results),
        attacks_detected=attacks,
        benign_flows=len(results) - attacks,
        total_latency_ms=round(total_ms, 3),
    )


@app.get("/recent", tags=["Detection"])
def recent_detections(limit: int = 50):
    """Return the last `limit` detection results (max 500 stored in memory)."""
    results = list(state.recent_results)[-limit:]
    return {"count": len(results), "results": results}


# ──────────────────────────────────────────────────────────────────────────────
# Dev entry-point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api:app", host="0.0.0.0", port=8000, reload=True)
