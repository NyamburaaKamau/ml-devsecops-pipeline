import os
import time
import logging
import joblib
import numpy as np
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator
from prometheus_client import (
    Counter, Histogram, Gauge,
    generate_latest, CONTENT_TYPE_LATEST
)
from starlette.responses import Response

logging.basicConfig(
    level=logging.INFO,
    format='{"time": "%(asctime)s", "level": "%(levelname)s", "message": "%(message)s"}'
)
logger = logging.getLogger(__name__)

# ── Prometheus Metrics ─────────────────────────────────────────────────────
PREDICTIONS_TOTAL = Counter(
    "ml_predictions_total",
    "Total predictions made",
    ["model_version", "result"]
)
PREDICTION_LATENCY = Histogram(
    "ml_prediction_latency_seconds",
    "Prediction latency",
    ["model_version"],
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0]
)
MODEL_LOADED = Gauge("ml_model_loaded", "Whether model is loaded", ["version"])
PREDICTION_CONFIDENCE = Histogram(
    "ml_prediction_confidence",
    "Model prediction confidence scores",
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
)

# ── Global State ───────────────────────────────────────────────────────────
model = None
MODEL_VERSION = os.getenv("MODEL_VERSION", "1.0.0")
MODEL_PATH = os.getenv("MODEL_PATH", "ml/model/model.joblib")


# ── Lifespan ───────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup, cleanup on shutdown."""
    global model
    logger.info(f"Loading model from {MODEL_PATH}...")
    try:
        model = joblib.load(MODEL_PATH)
        MODEL_LOADED.labels(version=MODEL_VERSION).set(1)
        logger.info(f"Model v{MODEL_VERSION} loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        MODEL_LOADED.labels(version=MODEL_VERSION).set(0)
        raise

    yield  # App runs here

    logger.info("Shutting down gracefully...")
    MODEL_LOADED.labels(version=MODEL_VERSION).set(0)


# ── App ────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="ML Churn Prediction API",
    description="Production ML model serving with full observability",
    version=MODEL_VERSION,
    lifespan=lifespan,
    docs_url="/docs" if os.getenv("ENVIRONMENT") != "production" else None,
)


# ── Schemas ────────────────────────────────────────────────────────────────
class PredictionRequest(BaseModel):
    tenure_months:      int   = Field(..., ge=0, le=120, description="Months as customer")
    monthly_charges:    float = Field(..., ge=0, le=500,  description="Monthly bill amount")
    total_charges:      float = Field(..., ge=0,          description="Total charges to date")
    num_products:       int   = Field(..., ge=1, le=10,   description="Number of products")
    support_calls:      int   = Field(..., ge=0, le=50,   description="Support calls made")
    payment_delay_days: int   = Field(..., ge=0, le=90,   description="Days payment delayed")
    contract_length:    int   = Field(..., description="Contract length in months")
    has_online_backup:  int   = Field(..., ge=0, le=1,    description="Has online backup (0/1)")
    has_tech_support:   int   = Field(..., ge=0, le=1,    description="Has tech support (0/1)")

    @field_validator("contract_length")
    @classmethod
    def validate_contract(cls, v):
        if v not in [1, 12, 24]:
            raise ValueError("contract_length must be 1, 12, or 24 months")
        return v

    model_config = {"json_schema_extra": {
        "example": {
            "tenure_months": 6,
            "monthly_charges": 75.5,
            "total_charges": 450.0,
            "num_products": 2,
            "support_calls": 4,
            "payment_delay_days": 5,
            "contract_length": 1,
            "has_online_backup": 0,
            "has_tech_support": 0
        }
    }}


class PredictionResponse(BaseModel):
    churn_prediction:   bool
    churn_probability:  float
    confidence:         str
    model_version:      str
    latency_ms:         float


# ── Routes ─────────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    """Kubernetes liveness probe."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "healthy", "model_version": MODEL_VERSION}


@app.get("/ready")
async def ready():
    """Kubernetes readiness probe — only ready when model is loaded."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not ready")
    return {"status": "ready", "model_version": MODEL_VERSION}


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Make a churn prediction."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    start = time.perf_counter()

    features = np.array([[
        request.tenure_months,
        request.monthly_charges,
        request.total_charges,
        request.num_products,
        request.support_calls,
        request.payment_delay_days,
        request.contract_length,
        request.has_online_backup,
        request.has_tech_support,
    ]])

    prediction = bool(model.predict(features)[0])
    probability = float(model.predict_proba(features)[0][1])
    latency_ms = (time.perf_counter() - start) * 1000

    # Confidence band
    if probability > 0.8 or probability < 0.2:
        confidence = "high"
    elif probability > 0.65 or probability < 0.35:
        confidence = "medium"
    else:
        confidence = "low"

    # Metrics
    PREDICTIONS_TOTAL.labels(
        model_version=MODEL_VERSION,
        result="churn" if prediction else "retain"
    ).inc()
    PREDICTION_LATENCY.labels(model_version=MODEL_VERSION).observe(latency_ms / 1000)
    PREDICTION_CONFIDENCE.observe(max(probability, 1 - probability))

    logger.info(f"Prediction: churn={prediction} prob={probability:.4f} "
                f"confidence={confidence} latency={latency_ms:.2f}ms")

    return PredictionResponse(
        churn_prediction=prediction,
        churn_probability=round(probability, 4),
        confidence=confidence,
        model_version=MODEL_VERSION,
        latency_ms=round(latency_ms, 2),
    )


@app.get("/")
async def root():
    return {
        "service": "ML Churn Prediction API",
        "version": MODEL_VERSION,
        "docs": "/docs",
        "health": "/health",
        "metrics": "/metrics",
    }