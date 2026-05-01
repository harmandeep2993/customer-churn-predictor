# src/models/train.py

import pandas as pd
import mlflow
import joblib
import mlflow.sklearn
from pathlib import Path
from mlflow.tracking import MlflowClient
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score

from src.utils import config, get_logger
from src.models.evaluate import evaluate_pipeline

logger = get_logger(__name__)


def setup_mlflow() -> None:
    """Set up MLflow tracking and experiment, restoring if deleted."""
    mlflow.set_tracking_uri(config.mlflow.tracking_uri)
    
    client = MlflowClient()
    experiment = client.get_experiment_by_name(config.mlflow.experiment_name)

    if experiment is not None and experiment.lifecycle_stage == "deleted":
        client.restore_experiment(experiment.experiment_id)
        logger.info(f"Restored deleted experiment '{config.mlflow.experiment_name}'.")

    mlflow.set_experiment(config.mlflow.experiment_name)
    logger.info(f"MLflow experiment set to '{config.mlflow.experiment_name}'.")


def get_models() -> dict:
    """Return model instances with hyperparams from config."""
    raw = config.raw["models"]
    return {
        "logistic_regression": LogisticRegression(**raw["logistic_regression"], class_weight= "balanced"),
        "random_forest": RandomForestClassifier(**raw["random_forest"], class_weight="balanced"),
        "xgboost": XGBClassifier(**raw["xgboost"], eval_metric="logloss", scale_pos_weight=3),
    }


def split_data(df: pd.DataFrame):
    """Split features and target into train/test sets."""
    X = df.drop(columns=["churn"])
    y = df["churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config.training.test_size,
        random_state=config.training.random_state,
        stratify=y
    )

    logger.info(f"Train size: {X_train.shape}, Test size: {X_test.shape}")
    return X_train, X_test, y_train, y_test


def compute_metrics(model, X_test, y_test) -> dict:
    """Compute evaluation metrics for a trained model."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    return {
        "roc_auc": roc_auc_score(y_test, y_prob),
        "f1": f1_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
    }


def save_best_model(model, model_name: str) -> None:
    """Save best model to models/ directory."""
    models_dir = Path(config.paths.models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    path = models_dir / "best_model.pkl"
    joblib.dump(model, path)
    logger.info(f"Best model '{model_name}' saved to {path}")


def train_pipeline(df: pd.DataFrame) -> None:
    """Train all models, log to MLflow, register best model."""
    setup_mlflow()

    X_train, X_test, y_train, y_test = split_data(df)
    models = get_models()

    best_run_id = None
    best_auc = 0.0
    best_model = None
    best_model_name = None

    for model_name, model in models.items():
        with mlflow.start_run(run_name=model_name):
            logger.info(f"Training {model_name}...")

            mlflow.log_params(config.raw["models"][model_name])
            model.fit(X_train, y_train)

            metrics = compute_metrics(model, X_test, y_test)
            mlflow.log_metrics(metrics)

            evaluate_pipeline(model, X_test, y_test, model_name)
            mlflow.sklearn.log_model(model, artifact_path="model")

            run_id = mlflow.active_run().info.run_id
            logger.info(f"{model_name} — ROC-AUC: {metrics['roc_auc']:.4f}")

            if metrics["roc_auc"] > best_auc:
                best_auc = metrics["roc_auc"]
                best_run_id = run_id
                best_model = model
                best_model_name = model_name

    # save best model locally
    save_best_model(best_model, best_model_name)

    # register best model to MLflow
    model_uri = f"runs:/{best_run_id}/model"
    mlflow.register_model(model_uri=model_uri, name=config.mlflow.model_name)
    logger.info(f"Best model '{best_model_name}' registered — ROC-AUC: {best_auc:.4f}")

    # register best model
    model_uri = f"runs:/{best_run_id}/model"
    mlflow.register_model(model_uri=model_uri, name=config.mlflow.model_name)
    logger.info(f"Best model registered — ROC-AUC: {best_auc:.4f}, run_id: {best_run_id}")