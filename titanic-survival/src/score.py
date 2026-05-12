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
MODEL_VERSION = None

# -----------------------------
# Constants
# -----------------------------
REQUIRED = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
ALLOWED_SEX = {"male", "female"}
ALLOWED_EMBARKED = {"S", "C", "Q"}

#-----------------------------
# Schema validation
#-----------------------------
class TitanicInput(BaseModel):
    Pclass: int
    Sex: str
    Age: float
    SibSp: int
    Parch: int
    Fare: float
    Embarked: str

    @field_validator("Sex")
    @classmethod
    def validate_sex(cls, value):
        if value not in ALLOWED_SEX:
            raise ValueError("Sex must be 'male' or 'female'")
        return value

    @field_validator("Embarked")
    @classmethod
    def validate_embarked(cls, value):
        if value not in ALLOWED_EMBARKED:
            raise ValueError("Embarked must be one of S, C, Q")
        return value

# -----------------------------
# Load model
# -----------------------------
def init():
    global model, MODEL_FEATURES, metadata, MODEL_VERSION

    logger.info({"event": "init_started"})

    model_dir = os.getenv("AZUREML_MODEL_DIR")
    model_path = os.path.join(model_dir, "model_pipeline.pkl")
    MODEL_VERSION = os.path.basename(model_dir)

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
        records = [
            dict(zip(data["columns"], row))
            for row in data["data"]
        ]
        return records

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

    for c in ["Age", "Fare", "SibSp", "Parch", "Pclass"]:
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
            TitanicInput(**record).model_dump()
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

     
        # Feature engineering
        df["Family_Size"] = df["SibSp"] + df["Parch"]
        df["Alone"] = df["Family_Size"].apply(lambda x: 1 if x == 0 else 0)

        df = df.drop(["SibSp", "Parch"], axis=1)

        logger.info({"event": "prediction_started"})

        preds = model.predict(df)
        probs = model.predict_proba(df)[:, 1]

        latency_ms = int((time.time() - start) * 1000)

        logger.info({
            "event": "prediction_success",
            "deployment": os.getenv("AZUREML_DEPLOYMENT_NAME"),
            "model_version": MODEL_VERSION,
            "latency_ms": latency_ms
        })

        return {
            "model_version": MODEL_VERSION,
            "deployment": os.getenv("AZUREML_DEPLOYMENT_NAME"),
            "predictions": preds.tolist(),
            "probabilities": probs.tolist(),
            "latency_ms": latency_ms
        }

    except ValidationError as e:
        logger.warning({"event": "schema_validation_failed", "error": str(e)})
        return {"error": str(e)}

    except Exception as e:
        logger.exception("inference_failed")
        return {"error": str(e)}