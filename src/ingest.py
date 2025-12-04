import pandas as pd
from sklearn.datasets import load_breast_cancer
import mlflow
import os

# --- CRITICAL FIX: Set Tracking URI programmatically ---
mlflow.set_tracking_uri("http://localhost:5001")

# Define the relative path to save the data
data_dir = "data"
file_name = "breast_cancer.csv"
file_path = os.path.join(data_dir, file_name)

# --- MLflow Run Context ---
with mlflow.start_run(run_name="ingest_data") as run:
    # 1. Load Data
    data = load_breast_cancer(as_frame=True)
    df = data.frame
    
    # 2. Save Data Locally
    os.makedirs(data_dir, exist_ok=True)
    df.to_csv(file_path, index=False)

    print(f"Data ingested and saved to: {file_path}")

    # 3. Log Artifact to MLflow
    mlflow.log_artifact(file_path, artifact_path="raw_data")
    print(f"Logged artifact to run ID: {run.info.run_id}")