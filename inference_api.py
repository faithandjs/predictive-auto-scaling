#!/usr/bin/env python3
"""
Production Inference API - Predictive Auto-Scaling
COM747 - Predictive Auto-Scaling Project
FastAPI server with ~3ms inference latency
Run with: python inference_api.py
"""
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pandas as pd
import xgboost as xgb
import pickle
import time

app = FastAPI(
    title="Predictive Auto-Scaling API",
    description="Forecasts VM CPU 15 minutes ahead and decides whether to scale up",
    version="1.0.0"
)

# Load trained XGBoost models
reg_model = xgb.Booster()
reg_model.load_model("models/xgb_regressor.json")

clf_model = xgb.Booster()
clf_model.load_model("models/xgb_classifier.json")

# The models were trained on standardised inputs, so the same scaler that
# features.py fitted on the training data must be applied here.
with open("data/features/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)
SCALED_COLS = list(scaler.feature_names_in_)  # every feature except time_of_day

# EXACT 19 feature names in EXACT order matching training
FEATURE_NAMES = [
    'max_cpu', 'avg_cpu',
    'avg_cpu_lag_1', 'max_cpu_lag_1',
    'avg_cpu_lag_2', 'max_cpu_lag_2',
    'avg_cpu_lag_3', 'max_cpu_lag_3',
    'avg_cpu_lag_5', 'max_cpu_lag_5',
    'avg_cpu_roll_mean_3', 'avg_cpu_roll_std_3', 'max_cpu_roll_mean_3',
    'avg_cpu_roll_mean_5', 'avg_cpu_roll_std_5', 'max_cpu_roll_mean_5',
    'avg_cpu_delta_1', 'avg_cpu_delta_3', 'time_of_day'
]

SCALE_THRESHOLD = 70.0  # 75% - 5% safety buffer
MIN_HISTORY = 6         # current reading + 5 previous (needed for lag_5 / roll_5)


class MetricsInput(BaseModel):
    vm_id: str
    timestamp: int
    avg_cpu: float
    max_cpu: float
    min_cpu: float


# In-memory per-VM history store
vm_history: dict = {}


def build_features(history: list):
    """Build the 19-feature vector exactly as features.py does, then scale it.

    Mirrors training:
      - lag_k     = value k readings before the current one
      - roll_*_w  = stats over the w readings BEFORE the current one
                    (features.py uses .shift(1).rolling(w)), std with ddof=1
      - delta_k   = current avg_cpu - avg_cpu lag_k
      - time_of_day is left unscaled (it was excluded from the scaler)
    Returns None if there is not enough history (triggers reactive fallback).
    """
    if len(history) < MIN_HISTORY:
        return None

    avg = [h['avg_cpu'] for h in history]
    mx  = [h['max_cpu'] for h in history]

    f = {'max_cpu': mx[-1], 'avg_cpu': avg[-1]}
    for lag in [1, 2, 3, 5]:
        f[f'avg_cpu_lag_{lag}'] = avg[-1 - lag]
        f[f'max_cpu_lag_{lag}'] = mx[-1 - lag]
    for w in [3, 5]:
        prev_avg = avg[-1 - w:-1]
        prev_max = mx[-1 - w:-1]
        f[f'avg_cpu_roll_mean_{w}'] = float(np.mean(prev_avg))
        f[f'avg_cpu_roll_std_{w}']  = float(np.std(prev_avg, ddof=1))
        f[f'max_cpu_roll_mean_{w}'] = float(np.mean(prev_max))
    f['avg_cpu_delta_1'] = avg[-1] - f['avg_cpu_lag_1']
    f['avg_cpu_delta_3'] = avg[-1] - f['avg_cpu_lag_3']
    f['time_of_day'] = history[-1]['timestamp'] % 86400

    row = pd.DataFrame([f])
    row[SCALED_COLS] = scaler.transform(row[SCALED_COLS])
    return row[FEATURE_NAMES].values.astype(float)[0]


@app.post("/predict")
async def predict(metrics: MetricsInput):
    """
    Main endpoint: receive current VM metrics, return scale-up decision.
    Uses 15-minute-ahead CPU forecast; falls back to reactive if history < 6.
    """
    start = time.time()

    # Update per-VM history (keep last 50 readings)
    if metrics.vm_id not in vm_history:
        vm_history[metrics.vm_id] = []
    vm_history[metrics.vm_id].append(metrics.model_dump())
    vm_history[metrics.vm_id] = vm_history[metrics.vm_id][-50:]

    features = build_features(vm_history[metrics.vm_id])

    # Graceful degradation: not enough history -> use reactive threshold
    if features is None:
        return {
            "vm_id": metrics.vm_id,
            "decision": "reactive_fallback",
            "scale_up": metrics.avg_cpu > 75,
            "reason": "insufficient_history",
            "history_count": len(vm_history[metrics.vm_id]),
            "latency_ms": round((time.time() - start) * 1000, 2)
        }

    # Predictive inference
    dmatrix = xgb.DMatrix(np.array([features]), feature_names=FEATURE_NAMES)
    predicted_cpu = float(reg_model.predict(dmatrix)[0])
    scale_prob = float(clf_model.predict(dmatrix)[0])
    scale_up = predicted_cpu > SCALE_THRESHOLD

    latency_ms = round((time.time() - start) * 1000, 2)

    return {
        "vm_id": metrics.vm_id,
        "decision": "predictive",
        "predicted_cpu_15min": round(predicted_cpu, 2),
        "scale_probability": round(scale_prob, 4),
        "scale_up": scale_up,
        "threshold_used": SCALE_THRESHOLD,
        "history_used": len(vm_history[metrics.vm_id]),
        "latency_ms": latency_ms
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "ok",
        "model_loaded": True,
        "vms_tracked": len(vm_history)
    }


@app.get("/stats")
async def stats():
    """Show how many VMs are being tracked."""
    return {
        "vms_tracked": len(vm_history),
        "feature_count": len(FEATURE_NAMES),
        "scale_threshold": SCALE_THRESHOLD
    }


if __name__ == "__main__":
    import uvicorn
    print("Starting Predictive Auto-Scaling API on http://localhost:8000")
    print("Docs available at: http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)
