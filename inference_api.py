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
import xgboost as xgb
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


class MetricsInput(BaseModel):
    vm_id: str
    timestamp: int
    avg_cpu: float
    max_cpu: float
    min_cpu: float


# In-memory per-VM history store
vm_history: dict = {}


def build_features(history: list):
    """Build the 19-feature vector from last 5 readings.
    Returns None if insufficient history (triggers reactive fallback).
    """
    if len(history) < 5:
        return None

    recent = history[-5:]
    avg_cpus = [h['avg_cpu'] for h in recent]
    max_cpus = [h['max_cpu'] for h in recent]

    features = [
        max_cpus[-1],                                                # 1. max_cpu
        avg_cpus[-1],                                                # 2. avg_cpu
        avg_cpus[-2],                                                # 3. avg_cpu_lag_1
        max_cpus[-2],                                                # 4. max_cpu_lag_1
        avg_cpus[-3],                                                # 5. avg_cpu_lag_2
        max_cpus[-3],                                                # 6. max_cpu_lag_2
        avg_cpus[-4] if len(avg_cpus) > 3 else avg_cpus[-3],         # 7. avg_cpu_lag_3
        max_cpus[-4] if len(max_cpus) > 3 else max_cpus[-3],         # 8. max_cpu_lag_3
        avg_cpus[0],                                                 # 9. avg_cpu_lag_5
        max_cpus[0],                                                 # 10. max_cpu_lag_5
        np.mean(avg_cpus[-3:]),                                      # 11. avg_cpu_roll_mean_3
        float(np.std(avg_cpus[-3:])) if len(avg_cpus) >= 3 else 0,   # 12. avg_cpu_roll_std_3
        np.mean(max_cpus[-3:]),                                      # 13. max_cpu_roll_mean_3
        np.mean(avg_cpus),                                           # 14. avg_cpu_roll_mean_5
        float(np.std(avg_cpus)),                                     # 15. avg_cpu_roll_std_5
        np.mean(max_cpus),                                           # 16. max_cpu_roll_mean_5
        avg_cpus[-1] - avg_cpus[-2],                                 # 17. avg_cpu_delta_1
        avg_cpus[-1] - avg_cpus[-4] if len(avg_cpus) > 3 else 0,     # 18. avg_cpu_delta_3
        recent[-1]['timestamp'] % 86400                              # 19. time_of_day
    ]
    return features


@app.post("/predict")
async def predict(metrics: MetricsInput):
    """
    Main endpoint: receive current VM metrics, return scale-up decision.
    Uses 15-minute-ahead CPU forecast; falls back to reactive if history < 5.
    """
    start = time.time()

    # Update per-VM history (keep last 50 readings)
    if metrics.vm_id not in vm_history:
        vm_history[metrics.vm_id] = []
    vm_history[metrics.vm_id].append(metrics.dict())
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
