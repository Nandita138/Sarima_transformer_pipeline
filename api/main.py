import sys
import time
import json
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Any
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.data_loader import GridGuardDataLoader
from src.models.transformer_model import TransformerAnomalyDetector
from src.explainability.shap_explainer import GridGuardExplainer


class PredictionRequest(BaseModel):
    data: List[Dict[str, float]] = Field(..., description="Time series window of telemetry features")
    threshold: Optional[float] = Field(default=None)


class PredictionResponse(BaseModel):
    anomaly_score: float
    is_anomaly: bool
    threshold_used: float
    inference_time_ms: float


class ExplanationResponse(BaseModel):
    anomaly_score: float
    is_anomaly: bool
    fault_category: str
    explanation: str
    top_feature_attributions: Dict[str, float]
    inference_time_ms: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    explainer_loaded: bool
    version: str
    uptime_seconds: float


app = FastAPI(
    title="GridGuard REST API",
    description="Real-Time Power Grid Anomaly Detection & SHAP Explainable AI Fault Attribution API",
    version="1.0.0"
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

START_TIME = time.time()
STATE: Dict[str, Any] = {
    "loaded": False,
    "explainer_loaded": False,
    "detector": None,
    "loader": None,
    "explainer": None,
    "window_size": 96,
    "predictions": 0
}


def load_saved_models() -> None:
    model_dir = Path("models_saved")
    scaler_path = model_dir / "scaler.pkl"
    transformer_path = model_dir / "transformer_model.pt"

    if not scaler_path.exists() or not transformer_path.exists():
        missing = [str(p) for p in (scaler_path, transformer_path) if not p.exists()]
        raise FileNotFoundError(f"Missing saved model files: {', '.join(missing)}. Run run_pipeline.py first.")

    loader = GridGuardDataLoader("config/dataset_config.yaml")
    loader.load_scaler(str(scaler_path))

    detector = TransformerAnomalyDetector("config/model_config.yaml")
    detector.load(str(transformer_path), input_dim=len(loader.feature_names))

    STATE["loader"] = loader
    STATE["detector"] = detector
    STATE["window_size"] = loader.config.window_size
    STATE["loaded"] = True

    try:
        explainer = GridGuardExplainer(detector, loader.feature_names, "config/model_config.yaml")
        # Setup background data with synthetic normal baseline
        dummy_bg = np.zeros((20, loader.config.window_size, len(loader.feature_names)))
        explainer.setup(dummy_bg)
        STATE["explainer"] = explainer
        STATE["explainer_loaded"] = True
    except Exception as exc:
        print(f"[GridGuard API] Explainer warning: {exc}")


@app.on_event("startup")
async def startup_event():
    try:
        load_saved_models()
        print("[GridGuard API] Saved models loaded successfully.")
    except Exception as exc:
        print(f"[GridGuard API] Failed to load saved models on startup: {exc}")


@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="healthy",
        model_loaded=STATE["loaded"],
        explainer_loaded=STATE["explainer_loaded"],
        version="1.0.0",
        uptime_seconds=round(time.time() - START_TIME, 2)
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    if not STATE["loaded"]:
        raise HTTPException(503, "Model not loaded. Run run_pipeline.py first.")
    start = time.perf_counter()
    df = pd.DataFrame(request.data)
    if df.empty:
        raise HTTPException(400, "No input data provided.")
    try:
        values = STATE["loader"].transform_features(df)
    except Exception as exc:
        raise HTTPException(400, str(exc))

    ws = STATE["window_size"]
    if len(values) < ws:
        pad = np.zeros((ws - len(values), values.shape[1]), dtype=np.float32)
        values = np.vstack([pad, values])
    elif len(values) > ws:
        values = values[-ws:]

    X = values.reshape(1, ws, -1).astype(np.float32)
    scores = STATE["detector"].predict(X)
    score = float(scores[0])

    thresh = request.threshold if request.threshold is not None else STATE["detector"].optimal_threshold
    elapsed = (time.perf_counter() - start) * 1000
    STATE["predictions"] += 1

    return PredictionResponse(
        anomaly_score=round(score, 6),
        is_anomaly=bool(score >= thresh),
        threshold_used=round(thresh, 4),
        inference_time_ms=round(elapsed, 3)
    )


@app.post("/explain", response_model=ExplanationResponse)
async def explain(request: PredictionRequest):
    if not STATE["loaded"]:
        raise HTTPException(503, "Model not loaded. Run run_pipeline.py first.")

    start = time.perf_counter()
    df = pd.DataFrame(request.data)
    if df.empty:
        raise HTTPException(400, "No input data provided.")

    try:
        values = STATE["loader"].transform_features(df)
    except Exception as exc:
        raise HTTPException(400, str(exc))

    ws = STATE["window_size"]
    if len(values) < ws:
        pad = np.zeros((ws - len(values), values.shape[1]), dtype=np.float32)
        values = np.vstack([pad, values])
    elif len(values) > ws:
        values = values[-ws:]

    X = values.reshape(1, ws, -1).astype(np.float32)
    scores = STATE["detector"].predict(X)
    score = float(scores[0])
    thresh = request.threshold if request.threshold is not None else STATE["detector"].optimal_threshold
    is_anom = bool(score >= thresh)

    explanation_text = "Normal grid telemetry."
    fault_cat = "Normal Operation"
    attributions: Dict[str, float] = {}

    if STATE["explainer"] is not None:
        try:
            res = STATE["explainer"].explain(X, top_k=5)
            explanation_text = res["text_explanations"][0]
            fault_cat = res["fault_categories"][0]
            for idx, fname in enumerate(STATE["loader"].feature_names[:5]):
                attributions[fname] = float(res["feature_importance"][0, idx])
        except Exception as exc:
            explanation_text = f"SHAP attribution notice: {exc}"

    elapsed = (time.perf_counter() - start) * 1000
    return ExplanationResponse(
        anomaly_score=round(score, 6),
        is_anomaly=is_anom,
        fault_category=fault_cat,
        explanation=explanation_text,
        top_feature_attributions=attributions,
        inference_time_ms=round(elapsed, 3)
    )


@app.get("/metrics")
async def metrics():
    res_path = Path("models_saved/evaluation_results.json")
    benchmark_data = {}
    if res_path.exists():
        with open(res_path, "r") as f:
            benchmark_data = json.load(f)

    return {
        "model": "GridGuard Hybrid (SARIMA + Transformer)",
        "total_predictions": STATE["predictions"],
        "benchmarks": benchmark_data
    }


if __name__ == "__main__":
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
