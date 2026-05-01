# src/models/predict.py

import pandas as pd
import joblib
from pathlib import Path
from src.utils import config, get_logger

logger = get_logger(__name__)

MODELS_DIR = Path(config.paths.models_dir)


def load_artifacts():
    """Load model, scaler, and feature columns from models/ directory."""
    model = joblib.load(MODELS_DIR / "best_model.pkl")
    scaler = joblib.load(MODELS_DIR / "scaler.pkl")
    feature_columns = joblib.load(MODELS_DIR / "feature_columns.pkl")

    logger.info("Model, scaler and feature columns loaded successfully.")
    return model, scaler, feature_columns


def preprocess_input(data: dict, scaler, feature_columns: list) -> pd.DataFrame:
    """Transform raw input dict into model-ready DataFrame."""
    df = pd.DataFrame([data])

    # lowercase column names
    df.columns = df.columns.str.lower().str.replace(" ", "_")

    # encode binary columns
    binary_cols = ["gender", "partner", "dependents", "phoneservice", "paperlessbilling", "multiplelines"]
    for col in binary_cols:
        if col in df.columns:
            if col == "gender":
                df[col] = df[col].map({"Male": 1, "Female": 0})
            else:
                df[col] = df[col].map({"Yes": 1, "No": 0})

    # fix totalcharges
    df["totalcharges"] = pd.to_numeric(df["totalcharges"], errors="coerce").fillna(0)

    # one-hot encode categorical columns
    categorical_cols = [
        "internetservice", "onlinesecurity", "onlinebackup",
        "deviceprotection", "techsupport", "streamingtv",
        "streamingmovies", "contract", "paymentmethod",
    ]
    existing = [col for col in categorical_cols if col in df.columns]
    df = pd.get_dummies(df, columns=existing, drop_first=True)

    # scale numerical columns
    numerical_cols = ["tenure", "monthlycharges", "totalcharges"]
    existing_num = [col for col in numerical_cols if col in df.columns]
    df[existing_num] = scaler.transform(df[existing_num])

    # align columns with training features
    df = df.reindex(columns=feature_columns, fill_value=0)

    logger.info("Input preprocessed successfully.")
    return df


def predict_pipeline(data: dict) -> dict:
    """Run full prediction pipeline on a single customer input."""
    model, scaler, feature_columns = load_artifacts()

    df = preprocess_input(data, scaler, feature_columns)

    churn_prob = model.predict_proba(df)[:, 1][0]
    churn_label = int(churn_prob >= 0.5)

    result = {
        "churn_probability": round(float(churn_prob), 4),
        "churn_prediction": churn_label,
        "churn_label": "Churn" if churn_label == 1 else "No Churn"
    }

    logger.info(f"Prediction: {result}")
    return result