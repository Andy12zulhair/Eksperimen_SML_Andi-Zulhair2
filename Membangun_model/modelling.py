import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Set local MLflow tracking server URI (localhost / 127.0.0.1:5000)
try:
    mlflow.set_tracking_uri("http://127.0.0.1:5000/")
except Exception:
    pass

# Enable MLflow autologging for Basic level
mlflow.autolog()

def main():
    # Set experiment name
    mlflow.set_experiment("Diabetes_Risk_Basic_Experiment")
    
    # Load dataset
    data_path = os.path.join("diabetes_risk_preprocessing", "diabetes_risk_clean.csv")
    df = pd.read_csv(data_path)
    
    target_col = 'diabetes_risk'
    if target_col not in df.columns:
        target_col = df.columns[-1]
        
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Start MLflow run
    with mlflow.start_run():
        # Train model
        model = RandomForestClassifier(n_estimators=50, random_state=42)
        model.fit(X_train, y_train)
        
        # Predict
        preds = model.predict(X_test)
        
        # Calculate metric
        accuracy = accuracy_score(y_test, preds)
        print(f"Model trained with Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    main()
