import json
import os
import time
import logging
import joblib
import pandas as pd

from pydantic import BaseModel, ValidationError, field_validator
from typing import List

# -----------------------------
# Logging setup
# -----------------------------
logger = logging.getLogger("inference")
logger.setLevel(logging.INFO)

model = None
MODEL_FEATURES = None
metadata = None

# -----------------------------
# Constants
# -----------------------------
REQUIRED = ["gender","SeniorCitizen","Partner","Dependents","tenure","PhoneService","MultipleLines","InternetService","OnlineSecurity","OnlineBackup","DeviceProtection","TechSupport","StreamingTV","StreamingMovies","Contract","PaperlessBilling","PaymentMethod","MonthlyCharges","TotalCharges"]

#-----------------------------
# Schema validation
# -----------------------------
class TelcoInput(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float

# -----------------------------
# Load model
# -----------------------------
def init():
    global model, MODEL_FEATURES, metadata
    logger.info({"event": "init_started"})

    model_dir = os.getenv("AZUREML_MODEL_DIR")
    model_path = os.path.join(model_dir, "model_pipeline.pkl")
  
    logger.info({"event": "loading_model", "model_path": model_path})

    bundle = joblib.load(model_path)

    model = bundle["model"]
    MODEL_FEATURES = bundle["features"]
    metadata = bundle.get("metadata", {})

    logger.info({
        "event": "model_loaded",
        "metadata": metadata
    })

# -----------------------------
# Payload parsing
# -----------------------------
def parse_payload(raw_data):
    data = json.loads(raw_data)

    # Shape 1: {"input_data": [...]}
    if isinstance(data, dict) and "input_data" in data:
        data = data["input_data"]

    # Shape 2: {"columns": [...], "data": [[...], ...]}
    if isinstance(data, dict) and "columns" in data and "data" in data:
        return [
            dict(zip(data["columns"], row))
            for row in data["data"]
        ]

    # Shape 3: single row dict
    if isinstance(data, dict):
        return [data]

    # Shape 4: list of row dicts
    if isinstance(data, list):
        return data

    raise ValueError("Invalid input format")
    
# -----------------------------
# Dataframe validation
# -----------------------------
def validate_dataframe(df: pd.DataFrame):
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        return False, f"Missing columns: {missing}. Received columns: {df.columns.tolist()}"

    for c in ["tenure", "MonthlyCharges", "TotalCharges"]:
        if not pd.to_numeric(df[c], errors="coerce").notna().all():
            return False, f"Non-numeric values found in {c}"

    return True, None

# -----------------------------
# Scoring
# -----------------------------
def run(raw_data):
    start = time.time()

    try:
        logger.info({"event": "request_received"})

        records = parse_payload(raw_data)

        # Pydantic schema validation
        validated_records = [
            TelcoInput(**record).model_dump()
            for record in records
        ]

        df = pd.DataFrame(validated_records)

        logger.info({
            "event": "dataframe_created",
            "columns": df.columns.tolist(),
            "rows": len(df)
        })

        ok, err = validate_dataframe(df)
        if not ok:
            logger.warning({"event": "validation_failed", "error": err})
            return {"error": err}
        
        df = df[MODEL_FEATURES]
       
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        df["MonthlyCharges"] = pd.to_numeric(df["MonthlyCharges"], errors="coerce")
        df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

        logger.info({"event": "prediction_started"})

        preds = model.predict(df)
        probs = model.predict_proba(df)[:, 1]
        
        latency_ms = int((time.time() - start) * 1000)

        logger.info({
            "event": "prediction_success",
            "latency_ms": latency_ms
        })

        return {
            "predictions": preds.tolist(),
            "probabilities": probs.tolist()
        }

    except Exception as e:
        logger.warning({"event": "schema_validation_failed", "error": str(e)})
        return {"error": str(e)}

    except Exception as e:
        logger.exception("inference_failed")
        return {"error": str(e)}