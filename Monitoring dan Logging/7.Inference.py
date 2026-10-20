"""
Inference script to test model serving and generate traffic for monitoring & alerting.
Author: Andi Zulhair
DICODING SUBMISSION - KRITERIA 4
"""

import time
import json
import random
import requests
import pandas as pd
import os

SERVING_URL = "http://127.0.0.1:8000/predict"
HEALTH_URL = "http://127.0.0.1:8000/health"

# Feature names required by the model
FEATURE_NAMES = [
    'age', 'gender', 'city', 'bmi', 'family_history_diabetes',
    'physical_activity_level', 'diet_type', 'smoking_status',
    'alcohol_consumption', 'hours_sleep_per_night', 'stress_level',
    'fasting_blood_sugar', 'hba1c_level', 'blood_pressure_systolic',
    'blood_pressure_diastolic', 'waist_circumference_cm', 'income_bracket'
]

def get_sample_data():
    """Retrieve realistic sample data from clean dataset or synthetic fallback."""
    possible_csv = [
        os.path.join(os.path.dirname(__file__), "..", "Workflow-CI", "MLProject", "diabetes_risk_preprocessing", "diabetes_risk_clean.csv"),
        os.path.join(os.path.dirname(__file__), "..", "Membangun_model", "diabetes_risk_preprocessing", "diabetes_risk_clean.csv"),
        os.path.join("Workflow-CI", "MLProject", "diabetes_risk_preprocessing", "diabetes_risk_clean.csv")
    ]
    for p in possible_csv:
        if os.path.exists(p):
            df = pd.read_csv(p)
            feature_df = df[[c for c in FEATURE_NAMES if c in df.columns]]
            return feature_df.to_dict(orient="records")
    
    # Fallback synthetic realistic samples
    samples = []
    for _ in range(50):
        samples.append({
            'age': random.uniform(-1.5, 1.5),
            'gender': random.choice([-1.0, 1.0]),
            'city': random.uniform(-1.0, 1.0),
            'bmi': random.uniform(-1.5, 2.5),
            'family_history_diabetes': random.choice([-0.7, 1.3]),
            'physical_activity_level': random.uniform(-1.0, 1.5),
            'diet_type': random.uniform(-1.0, 1.0),
            'smoking_status': random.uniform(-1.5, 1.0),
            'alcohol_consumption': random.uniform(-1.0, 2.0),
            'hours_sleep_per_night': random.uniform(-1.0, 1.5),
            'stress_level': random.uniform(-1.0, 1.5),
            'fasting_blood_sugar': random.uniform(-1.0, 2.5),
            'hba1c_level': random.uniform(-1.0, 2.5),
            'blood_pressure_systolic': random.uniform(-1.0, 2.0),
            'blood_pressure_diastolic': random.uniform(-1.0, 2.0),
            'waist_circumference_cm': random.uniform(-1.0, 2.0),
            'income_bracket': random.uniform(-1.5, 1.5)
        })
    return samples

def test_health():
    """Verify health endpoint."""
    try:
        res = requests.get(HEALTH_URL, timeout=5)
        print("=== Health Check Status ===")
        print(f"Status Code: {res.status_code}")
        print(f"Response: {res.json()}")
        return res.status_code == 200
    except Exception as e:
        print(f"Health check failed: {e}")
        return False

def run_single_inference(sample):
    """Run a single sample inference to demonstrate serving."""
    payload = {"features": sample}
    response = requests.post(SERVING_URL, json=payload, timeout=10)
    print("\n=== Single Inference Result ===")
    print(f"HTTP Status: {response.status_code}")
    print(f"Response Payload:\n{json.dumps(response.json(), indent=2)}")
    return response.json()

def generate_traffic(num_requests=60, simulate_spikes=True):
    """
    Generate diverse traffic to populate Prometheus & Grafana metrics
    and trigger alert rules (High Latency, High Error Rate, Positive Class surge).
    """
    print(f"\n[Traffic Generator] Generating {num_requests} inference requests...")
    dataset_records = get_sample_data()
    
    for i in range(1, num_requests + 1):
        sample = random.choice(dataset_records).copy()
        
        # Introduce drift / variations
        if i % 5 == 0:
            sample['fasting_blood_sugar'] = random.uniform(1.5, 3.5) # Elevated blood sugar
            sample['bmi'] = random.uniform(1.2, 3.0)
            
        payload = {"features": sample}
        
        # Trigger latency spike every 15 requests (to trigger High Latency alert)
        if simulate_spikes and i % 12 == 0:
            payload["simulate_delay"] = 0.35
            print(f"[Req #{i}] Simulating latency spike (0.35s)...")
            
        # Trigger error every 20 requests (to trigger High Error Rate alert)
        if simulate_spikes and i % 20 == 0:
            payload["simulate_error"] = True
            print(f"[Req #{i}] Simulating validation error for alert...")

        try:
            res = requests.post(SERVING_URL, json=payload, timeout=10)
            status_desc = "OK" if res.status_code == 200 else f"ERR {res.status_code}"
            print(f"Request #{i:02d} -> {status_desc} | Status: {res.status_code}")
        except Exception as err:
            print(f"Request #{i:02d} -> Exception: {err}")
            
        time.sleep(0.1) # Smooth pacing

    print("[Traffic Generator] Traffic generation completed successfully.")

if __name__ == "__main__":
    print("==================================================")
    print(" ML Model Inference & Traffic Generator")
    print(" Author: Andi Zulhair")
    print("==================================================")
    if test_health():
        samples = get_sample_data()
        run_single_inference(samples[0])
        generate_traffic(num_requests=40, simulate_spikes=True)
    else:
        print("[Error] Serving endpoint is not reachable at http://127.0.0.1:8000")
