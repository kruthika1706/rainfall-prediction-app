from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import numpy as np
import json, os, joblib
import tensorflow as tf

# --- CONFIG ---
ARTIFACT_DIR = os.path.join(os.path.dirname(__file__), "artifacts")
SCALER_PATH = os.path.join(ARTIFACT_DIR, "scaler.pkl")
ANN_PATH = os.path.join(ARTIFACT_DIR, "ann.h5")
LSTM_PATH = os.path.join(ARTIFACT_DIR, "lstm.h5")
FEAT_PATH = os.path.join(ARTIFACT_DIR, "features.json")

# Check artifacts exist (placeholders allowed for now)
if not os.path.exists(ARTIFACT_DIR):
    os.makedirs(ARTIFACT_DIR)

# Try loading safely; if missing, server will still start but predict will error with clear message
scaler = None
ann_model = None
lstm_model = None

try:
    if os.path.exists(SCALER_PATH):
        scaler = joblib.load(SCALER_PATH)
    if os.path.exists(ANN_PATH):
        ann_model = tf.keras.models.load_model(ANN_PATH)
    if os.path.exists(LSTM_PATH):
        lstm_model = tf.keras.models.load_model(LSTM_PATH)
except Exception as e:
    print("Warning: failed to load some artifacts:", e)

if os.path.exists(FEAT_PATH):
    with open(FEAT_PATH, "r") as f:
        FEATURE_ORDER = json.load(f)
else:
    FEATURE_ORDER = ["Temperature","Humidity","Wind Speed","Precipitation","Cloud Cover","Pressure"]

N_FEATURES = len(FEATURE_ORDER)

app = FastAPI(title="Rainfall Prediction API (starter)")

class PredictIn(BaseModel):
    features: list[float] = Field(..., description=f"Feature vector in order: {FEATURE_ORDER}")

class PredictBatchIn(BaseModel):
    rows: list[list[float]] = Field(..., description=f"2D list where each row follows order: {FEATURE_ORDER}")

@app.get("/health")
def health():
    return {"status":"ok", "features": FEATURE_ORDER, "n_features": N_FEATURES}

def _require_artifacts():
    if scaler is None or ann_model is None or lstm_model is None:
        raise RuntimeError("Model artifacts missing. Place scaler.pkl, ann.h5, lstm.h5 in backend/artifacts/")

def _hybrid_predict_rows(rows_2d: np.ndarray) -> np.ndarray:
    _require_artifacts()
    if rows_2d.ndim != 2 or rows_2d.shape[1] != N_FEATURES:
        raise ValueError(f"Expected (n, {N_FEATURES}) got {rows_2d.shape}")
    X_scaled = scaler.transform(rows_2d)
    ann_pred = ann_model.predict(X_scaled, verbose=0).reshape(-1)
    X_lstm = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))
    lstm_pred = lstm_model.predict(X_lstm, verbose=0).reshape(-1)
    hybrid = (ann_pred + lstm_pred) / 2.0
    return hybrid

@app.post("/predict")
def predict(inp: PredictIn):
    try:
        row = np.array(inp.features, dtype=float).reshape(1, -1)
        preds = _hybrid_predict_rows(row)
        return {"prediction": float(preds[0])}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/predict-batch")
def predict_batch(inp: PredictBatchIn):
    try:
        rows = np.array(inp.rows, dtype=float)
        preds = _hybrid_predict_rows(rows)
        return {"predictions": preds.astype(float).tolist()}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
