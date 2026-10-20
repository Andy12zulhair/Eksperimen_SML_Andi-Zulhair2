import os
import sys
import shutil
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
from dotenv import load_dotenv

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

def init_tracking():
    # Load DagsHub Credentials from .env if available
    load_dotenv()
    tracking_uri = os.getenv('MLFLOW_TRACKING_URI')
    
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    else:
        try:
            import dagshub
            dagshub.init(repo_owner='Andy12zulhair', repo_name='Submission_andizulhair', mlflow=True)
        except Exception:
            # Fallback to local MLflow tracking server
            mlflow.set_tracking_uri('http://127.0.0.1:5000/')

def main():
    init_tracking()
    
    # Set experiment name
    mlflow.set_experiment('Diabetes_Risk_Tuning_Experiment')
    
    # Load dataset
    data_path = os.path.join('diabetes_risk_preprocessing', 'diabetes_risk_clean.csv')
    df = pd.read_csv(data_path)
    
    target_col = 'diabetes_risk'
    if target_col not in df.columns:
        target_col = df.columns[-1]
        
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Disable autologging as per criteria (Manual Logging)
    mlflow.autolog(disable=True)
    
    with mlflow.start_run():
        print('Starting hyperparameter tuning...')
        # Define model and parameters for tuning
        base_model = RandomForestClassifier(random_state=42)
        param_grid = {
            'n_estimators': [50, 100],
            'max_depth': [None, 10, 20]
        }
        
        grid_search = GridSearchCV(base_model, param_grid, cv=3, scoring='accuracy')
        grid_search.fit(X_train, y_train)
        
        best_model = grid_search.best_estimator_
        
        # Make predictions
        preds = best_model.predict(X_test)
        
        # Calculate metrics (Accuracy, Precision, Recall, F1-Score)
        accuracy = accuracy_score(y_test, preds)
        precision = precision_score(y_test, preds, average='weighted', zero_division=0)
        recall = recall_score(y_test, preds, average='weighted', zero_division=0)
        f1 = f1_score(y_test, preds, average='weighted', zero_division=0)
        
        print(f'Best Params: {grid_search.best_params_}')
        print(f'Accuracy : {accuracy:.4f}')
        print(f'Precision: {precision:.4f}')
        print(f'Recall   : {recall:.4f}')
        print(f'F1-Score : {f1:.4f}')
        
        # -----------------------------
        # MANUAL LOGGING (MEMENUHI BASIC & SKILLED)
        # -----------------------------
        # 1. Log Hyperparameters
        mlflow.log_params(grid_search.best_params_)
        
        # 2. Log Metrics (Minimal accuracy, precision, recall, f1)
        mlflow.log_metric('accuracy', accuracy)
        mlflow.log_metric('precision', precision)
        mlflow.log_metric('recall', recall)
        mlflow.log_metric('f1_score', f1)
        
        # 3. Model Signature & Input Example
        signature = infer_signature(X_train, preds)
        input_example = X_train.head(3)
        
        # 4. Log Model via MLflow Model Logging
        mlflow.sklearn.log_model(
            sk_model=best_model,
            artifact_path='model',
            signature=signature,
            input_example=input_example,
            skops_trusted_types=[
                'sklearn.tree._tree.Tree',
                'sklearn.tree._classes.DecisionTreeClassifier',
                'sklearn.ensemble._forest.RandomForestClassifier',
            ]
        )
        
        # 4b. Explicitly save & log model folder directly into run artifacts (so 'model/' appears in run artifacts tab)
        temp_model_dir = 'temp_model_artifacts'
        if os.path.exists(temp_model_dir):
            shutil.rmtree(temp_model_dir)
        mlflow.sklearn.save_model(
            sk_model=best_model,
            path=temp_model_dir,
            signature=signature,
            input_example=input_example,
            skops_trusted_types=[
                'sklearn.tree._tree.Tree',
                'sklearn.tree._classes.DecisionTreeClassifier',
                'sklearn.ensemble._forest.RandomForestClassifier',
            ]
        )
        with open(os.path.join(temp_model_dir, 'model.pkl'), 'wb') as f_pkl:
            pickle.dump(best_model, f_pkl)
        mlflow.log_artifacts(temp_model_dir, artifact_path='model')
        shutil.rmtree(temp_model_dir)
        print('Logged model artifacts directly to run artifacts: model/')
        
        # 5. Artifact Tambahan 1: Feature Importances
        importances = best_model.feature_importances_
        fi_df = pd.DataFrame({'feature': X.columns, 'importance': importances})
        fi_df = fi_df.sort_values(by='importance', ascending=False)
        fi_csv_path = 'feature_importance.csv'
        fi_df.to_csv(fi_csv_path, index=False)
        mlflow.log_artifact(fi_csv_path)
        print('Logged artifact: feature_importance.csv')
        
        # 6. Artifact Tambahan 2: Confusion Matrix Image
        cm = confusion_matrix(y_test, preds)
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        cm_image_path = 'confusion_matrix.png'
        plt.savefig(cm_image_path)
        plt.close()
        mlflow.log_artifact(cm_image_path)
        print('Logged artifact: confusion_matrix.png')
        
        # Cleanup temporary local image/csv
        if os.path.exists(fi_csv_path):
            os.remove(fi_csv_path)
        if os.path.exists(cm_image_path):
            os.remove(cm_image_path)
        
        print('[SUCCESS] Model, parameters, all metrics, and artifacts logged successfully to MLflow / DagsHub!')

if __name__ == '__main__':
    main()
