import argparse
import pandas as pd
import numpy as np
import mlflow
from mlflow.tracking import MlflowClient
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score
import joblib
import os

# --- CRITICAL FIX: Set Tracking URI programmatically ---
# This ensures the script, even in an isolated environment, knows where the server is.
# This bypasses the persistent 403 Forbidden error.
# mlflow.set_tracking_uri("http://localhost:5001")

def train_model(n_estimators, max_depth):
    """
    Trains a Random Forest classifier, logs parameters, metrics, and the model
    to the active MLflow run, and saves the trained model as an artifact.
    """
    
    # --- 1. Fetch Data Artifact from Ingest Run ---
    
    # We assume the 'ingest' run logged the data as 'raw_data/breast_cancer.csv'
    # We will fetch the latest run that logged this artifact.
    
    client = MlflowClient()
    
    # Look for the last run that logged the 'raw_data' artifact
    # Note: If you have multiple experiments, you may need a more specific filter
    runs = client.search_runs(
        experiment_ids=["0"],  # Assuming the default experiment ID 0
        filter_string="tags.mlflow.project.entryPoint = 'ingest'",
        order_by=["start_time DESC"],
        max_results=1
    )
    
    if not runs:
        print("Error: Could not find a previous 'ingest' run to retrieve data from.")
        return
        
    ingest_run_id = runs[0].info.run_id
    
    # Create a local directory for artifacts
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)

    # Download the artifact (the data file)
    local_path = client.download_artifacts(
        run_id=ingest_run_id, 
        path="raw_data/breast_cancer.csv", 
        dst_path=data_dir
    )
    
    print(f"Successfully downloaded data from run {ingest_run_id} to {local_path}")
    
    # Load the data
    df = pd.read_csv(local_path)
    X = df.drop(columns=['target'])
    y = df['target']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # --- 2. Start MLflow Run for Training ---
    with mlflow.start_run(run_name="train_random_forest") as run:
        
        # Log parameters to MLflow
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)

        # Train the model
        model = RandomForestClassifier(
            n_estimators=n_estimators, 
            max_depth=max_depth, 
            random_state=42
        )
        model.fit(X_train, y_train)

        # Make predictions
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        # Calculate metrics
        roc_auc = roc_auc_score(y_test, y_proba)
        accuracy = accuracy_score(y_test, y_pred)

        # Log metrics to MLflow
        mlflow.log_metric("roc_auc", roc_auc)
        mlflow.log_metric("accuracy", accuracy)
        
        print(f"Model Trained. AUC: {roc_auc:.4f}, Accuracy: {accuracy:.4f}")

        # --- 3. Log Model and Save Artifacts ---
        
        # Log the scikit-learn model
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name="BreastCancerClassifier"
        )
        
        # Save the model locally as a joblib file (optional, but good practice)
        model_path = "model.joblib"
        joblib.dump(model, model_path)
        mlflow.log_artifact(model_path, artifact_path="model_output")
        os.remove(model_path) # Clean up local copy
        
        print(f"Logged model and artifacts to run ID: {run.info.run_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=100,
        help="The number of trees in the forest.",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=3,
        help="The maximum depth of the tree.",
    )
    args = parser.parse_args()
    
    train_model(args.n_estimators, args.max_depth)