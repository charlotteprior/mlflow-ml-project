import os
import pandas as pd
from fastapi import FastAPI
import mlflow

# --- STEP 1: DEFINE THE RUN ID ---
# IMPORTANT: Replace the placeholder below with the actual Run ID 
# from your *successful training run* in the MLflow UI.
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://host.docker.internal:5001")
RUN_ID = os.getenv("RUN_ID", "fef4dbf213f645d594d85200b3124ab6")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# --- STEP 2: LOAD THE MODEL ---
try:
    # Construct the MLflow model URI
    model_uri = f"runs:/{RUN_ID}/model"
    
    # Load the model from the MLflow tracking server
    model = mlflow.sklearn.load_model(model_uri)
    print(f"Successfully loaded model from run: {RUN_ID}")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None # Set to None if loading fails

app = FastAPI()

@app.get("/health")
def health_check():
    """Simple health check endpoint."""
    if model:
        return {"status": "ok", "model_loaded": True}
    else:
        return {"status": "error", "model_loaded": False, "message": "Model failed to load."}

@app.post("/predict")
def predict(features: dict):
    """
    Makes a prediction using the loaded scikit-learn model.
    The input features should be passed as a JSON object containing 30 keys (features).
    """
    if not model:
        return {"error": "Model not available for prediction."}, 500

    # Convert the input dictionary to a pandas DataFrame (necessary for scikit-learn models)
    try:
        # FastAPI passes the JSON body as a dict. We wrap it in a list to represent a single row.
        input_data = pd.DataFrame([features])
        
        # Make prediction
        prediction = model.predict(input_data).tolist()
        
        # Make probability prediction (optional, but useful for classification)
        probabilities = model.predict_proba(input_data).tolist()
        
        return {
            "prediction": prediction, 
            "probabilities": probabilities
        }
    except Exception as e:
        return {"error": f"Prediction failed: {e}"}, 400
    
    