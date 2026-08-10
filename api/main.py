import sys  
import time  
import numpy as np  
import pandas as pd  
from typing import List, Dict, Optional  
from pathlib import Path  
from fastapi import FastAPI, HTTPException  
from fastapi.middleware.cors import CORSMiddleware  
from pydantic import BaseModel, Field  
import uvicorn

sys.path.insert(0, str(Path(__file__).parent.parent))


class PredictionRequest(BaseModel):  
    data: List[Dict[str, float]] = Field(...)  
    threshold: Optional[float] = Field(default=0.5)


class PredictionResponse(BaseModel):  
    anomaly_score: float  
    is_anomaly: bool  
    threshold_used: float  
    inference_time_ms: float


class HealthResponse(BaseModel):  
    status: str  
    model_loaded: bool  
    version: str  
    uptime_seconds: float


app = FastAPI(title="GridGuard API", version="1.0.0")  
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

START_TIME = time.time()  
STATE = {"loaded": False, "detector": None, "window_size": 96, "predictions": 0}


@app.get("/health", response_model=HealthResponse)  
async def health():  
    return HealthResponse(status="healthy", model_loaded=STATE["loaded"],  
                          version="1.0.0", uptime_seconds=round(time.time() - START_TIME, 2))


@app.post("/predict", response_model=PredictionResponse)  
async def predict(request: PredictionRequest):  
    if not STATE["loaded"]:  
        raise HTTPException(503, "Model not loaded. Run run_pipeline.py first.")  
    start = time.perf_counter()  
    df = pd.DataFrame(request.data)  
    values = df.values  
    ws = STATE["window_size"]  
    if len(values) < ws:  
        pad = np.zeros((ws - len(values), values.shape[1]))  
        values = np.vstack([pad, values])  
    elif len(values) > ws:  
        values = values[-ws:]  
    X = values.reshape(1, ws, -1).astype(np.float32)  
    scores = STATE["detector"].predict(X)  
    score = float(scores[0])  
    elapsed = (time.perf_counter() - start) * 1000  
    STATE["predictions"] += 1  
    return PredictionResponse(anomaly_score=round(score, 6), is_anomaly=score >= request.threshold,  
                              threshold_used=request.threshold, inference_time_ms=round(elapsed, 3))


@app.get("/metrics")  
async def metrics():  
    return {"model": "GridGuard Hybrid", "total_predictions": STATE["predictions"]}


if __name__ == "__main__":  
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)  
