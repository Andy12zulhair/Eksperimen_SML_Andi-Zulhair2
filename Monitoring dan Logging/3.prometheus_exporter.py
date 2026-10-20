"""
Prometheus Exporter and Model Serving for Diabetes Risk Prediction Model
Author: Andi Zulhair
DICODING SUBMISSION - KRITERIA 4 (ADVANCE - 4 PTS)
"""

import os
import time
import psutil
import cloudpickle
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Union
from fastapi import FastAPI, Request, Response, HTTPException, status
from fastapi.responses import JSONResponse
import uvicorn
from prometheus_client import (
    Counter,
    Gauge,
    Histogram,
    generate_latest,
    CONTENT_TYPE_LATEST,
    REGISTRY
)

# Initialize FastAPI App
app = FastAPI(
    title="Diabetes Risk Model Serving & Prometheus Exporter",
    description="Serving endpoint and metrics exporter for ML monitoring",
    version="1.0.0"
)

# -------------------------------------------------------------
# 12 ADVANCED METRICS FOR PROMETHEUS & GRAFANA MONITORING
# -------------------------------------------------------------
# Metric 1: Total HTTP Requests
ML_REQUESTS_TOTAL = Counter(
    'ml_requests_total',
    'Total HTTP requests processed by endpoint and status code',
    ['endpoint', 'status_code']
)

# Metric 2: Requests Currently in Progress
ML_REQUESTS_IN_PROGRESS = Gauge(
    'ml_requests_in_progress',
    'Current number of HTTP requests being processed'
)

# Metric 3: Total Predictions Partitioned by Class
ML_PREDICTIONS_TOTAL = Counter(
    'ml_predictions_total',
    'Total model predictions categorized by predicted diabetes risk class',
    ['predicted_class']
)

# Metric 4: Prediction Latency in Seconds
ML_PREDICTION_LATENCY = Histogram(
    'ml_prediction_latency_seconds',
    'Inference processing time in seconds',
    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5]
)

# Metric 5: Prediction Confidence / Probabilities
ML_PREDICTION_CONFIDENCE = Histogram(
    'ml_prediction_confidence',
    'Confidence probability score of model predictions',
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
)

# Metric 6: Positive Class Ratio (Positive predictions / Total predictions)
ML_POSITIVE_CLASS_RATIO = Gauge(
    'ml_positive_class_ratio',
    'Moving ratio of positive diabetes risk predictions over total predictions'
)

# Metric 7: Feature Drift Indicator for Fasting Blood Sugar
ML_DRIFT_FASTING_BLOOD_SUGAR = Gauge(
    'ml_feature_drift_fasting_blood_sugar',
    'Moving average of incoming fasting_blood_sugar feature values'
)

# Metric 8: Feature Drift Indicator for BMI
ML_DRIFT_BMI = Gauge(
    'ml_feature_drift_bmi',
    'Moving average of incoming bmi feature values'
)

# Metric 9: Feature Drift Indicator for Age
ML_DRIFT_AGE = Gauge(
    'ml_feature_drift_age',
    'Moving average of incoming age feature values'
)

# Metric 10: Model Process Resident Memory Usage (RSS bytes)
ML_MODEL_MEMORY_BYTES = Gauge(
    'ml_model_memory_usage_bytes',
    'Resident memory usage of the model serving process in bytes'
)

# Metric 11: Model Process CPU Utilization Percent
ML_MODEL_CPU_PERCENT = Gauge(
    'ml_model_cpu_utilization_percent',
    'CPU utilization percentage of the model serving process'
)

# Metric 12: Total Inference / Validation Errors
ML_ERRORS_TOTAL = Counter(
    'ml_errors_total',
    'Total number of inference exceptions and validation errors',
    ['error_type']
)

# State tracking for moving statistics
prediction_counts = {"0": 0, "1": 0}
fasting_sugar_history = []
bmi_history = []
age_history = []

# Load model
MODEL = None
FEATURE_COLUMNS = [
    'age', 'gender', 'city', 'bmi', 'family_history_diabetes',
    'physical_activity_level', 'diet_type', 'smoking_status',
    'alcohol_consumption', 'hours_sleep_per_night', 'stress_level',
    'fasting_blood_sugar', 'hba1c_level', 'blood_pressure_systolic',
    'blood_pressure_diastolic', 'waist_circumference_cm', 'income_bracket'
]

def load_trained_model():
    global MODEL
    possible_paths = [
        os.path.join(os.path.dirname(__file__), "..", "Workflow-CI", "MLProject", "saved_model", "model.pkl"),
        os.path.join("Workflow-CI", "MLProject", "saved_model", "model.pkl"),
        os.path.join(os.path.dirname(__file__), "model.pkl"),
    ]
    for p in possible_paths:
        if os.path.exists(p):
            with open(p, "rb") as f:
                MODEL = cloudpickle.load(f)
            print(f"[Serving] Successfully loaded model from: {p}")
            return
    print("[Serving] Warning: Saved model not found, fallback initialization")

load_trained_model()

def update_system_metrics():
    """Update process CPU and memory metrics."""
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    ML_MODEL_MEMORY_BYTES.set(mem_info.rss)
    cpu_pct = process.cpu_percent(interval=None)
    ML_MODEL_CPU_PERCENT.set(cpu_pct)

@app.get("/health")
def health_check():
    """Healthcheck endpoint for Kubernetes, Docker, or local serving check."""
    ML_REQUESTS_TOTAL.labels(endpoint="/health", status_code="200").inc()
    update_system_metrics()
    return {
        "status": "healthy",
        "service": "diabetes-risk-monitoring-exporter",
        "author": "Andi Zulhair",
        "model_loaded": MODEL is not None
    }

@app.get("/metrics")
def get_metrics():
    """Prometheus metrics scrape endpoint."""
    update_system_metrics()
    return Response(content=generate_latest(REGISTRY), media_type=CONTENT_TYPE_LATEST)

@app.post("/predict")
async def predict(payload: Dict[str, Any]):
    """
    Model Inference endpoint.
    Accepts JSON containing feature inputs:
    - {"dataframe_records": [{...}, ...]} or
    - {"features": [{...}] or dict} or
    - {"instances": [[...]]}
    """
    start_time = time.time()
    ML_REQUESTS_IN_PROGRESS.inc()
    
    try:
        if "dataframe_records" in payload:
            df = pd.DataFrame(payload["dataframe_records"])
        elif "features" in payload:
            feats = payload["features"]
            df = pd.DataFrame(feats if isinstance(feats, list) else [feats])
        elif "instances" in payload:
            df = pd.DataFrame(payload["instances"], columns=FEATURE_COLUMNS)
        else:
            # Assume direct dictionary of features or list
            df = pd.DataFrame([payload])
            
        # Ensure required columns
        for col in FEATURE_COLUMNS:
            if col not in df.columns:
                df[col] = 0.0
                
        df = df[FEATURE_COLUMNS]

        # Check for simulated latency if provided (for testing alerts)
        if "simulate_delay" in payload:
            time.sleep(float(payload["simulate_delay"]))

        # Check for simulated error if requested (for testing alerts)
        if payload.get("simulate_error", False):
            raise ValueError("Simulated inference failure for alert validation")

        # Run Model Inference
        predictions = MODEL.predict(df)
        probabilities = MODEL.predict_proba(df)

        results = []
        for i, pred in enumerate(predictions):
            pred_class = str(int(pred))
            prob = float(probabilities[i][int(pred)])
            results.append({
                "prediction": int(pred),
                "risk_label": "High Risk" if pred == 1 else "Low Risk",
                "confidence": round(prob, 4)
            })

            # Update Metrics
            ML_PREDICTIONS_TOTAL.labels(predicted_class=pred_class).inc()
            ML_PREDICTION_CONFIDENCE.observe(prob)
            prediction_counts[pred_class] = prediction_counts.get(pred_class, 0) + 1

        # Update Positive Class Ratio
        total_p = sum(prediction_counts.values())
        if total_p > 0:
            ML_POSITIVE_CLASS_RATIO.set(prediction_counts.get("1", 0) / total_p)

        # Update Feature Drift Metrics (moving average of incoming inputs)
        if 'fasting_blood_sugar' in df.columns:
            val = float(df['fasting_blood_sugar'].mean())
            fasting_sugar_history.append(val)
            ML_DRIFT_FASTING_BLOOD_SUGAR.set(np.mean(fasting_sugar_history[-50:]))

        if 'bmi' in df.columns:
            val = float(df['bmi'].mean())
            bmi_history.append(val)
            ML_DRIFT_BMI.set(np.mean(bmi_history[-50:]))

        if 'age' in df.columns:
            val = float(df['age'].mean())
            age_history.append(val)
            ML_DRIFT_AGE.set(np.mean(age_history[-50:]))

        duration = time.time() - start_time
        ML_PREDICTION_LATENCY.observe(duration)
        ML_REQUESTS_TOTAL.labels(endpoint="/predict", status_code="200").inc()
        update_system_metrics()

        return JSONResponse(content={
            "predictions": results,
            "total_items": len(results),
            "inference_time_seconds": round(duration, 5)
        })

    except Exception as e:
        duration = time.time() - start_time
        ML_PREDICTION_LATENCY.observe(duration)
        ML_ERRORS_TOTAL.labels(error_type=type(e).__name__).inc()
        ML_REQUESTS_TOTAL.labels(endpoint="/predict", status_code="500").inc()
        update_system_metrics()
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        ML_REQUESTS_IN_PROGRESS.dec()

if __name__ == "__main__":
    print("[Serving] Starting Model Serving & Prometheus Exporter on http://127.0.0.1:8000 ...")
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
