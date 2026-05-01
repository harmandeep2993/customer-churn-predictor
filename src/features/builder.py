# src/features/builder.py

import pandas as pd
import joblib
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from src.utils import config, get_logger

logger = get_logger(__name__)


def encode_target(df: pd.DataFrame) -> pd.DataFrame:
    """Encode target column 'churn' to binary 0/1."""
    df["churn"] = df["churn"].map({"Yes": 1, "No": 0})
    logger.info("Target column 'churn' encoded.")
    return df


def encode_categorical(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode multi-class categorical columns."""
    columns_to_encode = [
        "multiplelines",
        "internetservice",
        "onlinesecurity",
        "onlinebackup",
        "deviceprotection",
        "techsupport",
        "streamingtv",
        "streamingmovies",
        "contract",
        "paymentmethod",
    ]

    existing = [col for col in columns_to_encode if col in df.columns]
    df = pd.get_dummies(df, columns=existing, drop_first=True)

    logger.info(f"One-hot encoded {len(existing)} categorical columns.")
    return df


def scale_numerical(df: pd.DataFrame) -> tuple[pd.DataFrame, StandardScaler]:
    """Standard scale numerical columns, return df and fitted scaler."""
    numerical_cols = ["tenure", "monthlycharges", "totalcharges"]
    existing = [col for col in numerical_cols if col in df.columns]

    scaler = StandardScaler()
    df[existing] = scaler.fit_transform(df[existing])

    logger.info(f"Scaled {len(existing)} numerical columns.")
    return df, scaler


def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, StandardScaler, list]:
    """Full feature engineering pipeline. Returns df, scaler, feature columns."""
    df = encode_target(df)
    df = encode_categorical(df)
    df, scaler = scale_numerical(df)

    feature_columns = [col for col in df.columns if col != "churn"]
    logger.info(f"Feature engineering complete. Final shape: {df.shape}")

    return df, scaler, feature_columns


def save_artifacts(scaler: StandardScaler, feature_columns: list) -> None:
    """Save scaler and feature columns to models/ directory."""
    models_dir = Path(config.paths.models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(scaler, models_dir / "scaler.pkl")
    joblib.dump(feature_columns, models_dir / "feature_columns.pkl")

    logger.info(f"Scaler and feature columns saved to {models_dir}")