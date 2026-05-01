# src/models/train.py

import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score

from src.utils import config, get_logger
from src.models import evaluate_and_log

logger = get_logger(__name__)


def get_models() -> dict:
    """Return model instances with hyperparams from config."""
    raw = config.raw["models"]
    return {
        "logistic_regression": LogisticRegression(**raw["logistic_regression"]),
        "random_forest": RandomForestClassifier(**raw["random_forest"]),
        "xgboost": XGBClassifier(**raw["xgboost"], eval_metric="logloss"),
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


def train_and_log(df: pd.DataFrame) -> None:
    """Train all models, log to MLflow, register best model."""
    mlflow.set_tracking_uri(config.mlflow.tracking_uri)
    mlflow.set_experiment(config.mlflow.experiment_name)

    X_train, X_test, y_train, y_test = split_data(df)
    models = get_models()

    best_run_id = None
    best_auc = 0.0

    for model_name, model in models.items():
        with mlflow.start_run(run_name=model_name):
            logger.info(f"Training {model_name}...")

            # log hyperparams
            mlflow.log_params(config.raw["models"][model_name])

            # train
            model.fit(X_train, y_train)

            # evaluate
            metrics = compute_metrics(model, X_test, y_test)
            mlflow.log_metrics(metrics)

            # Evaluate Model
            evaluate_and_log(model, X_test, y_test, model_name)

            # log model
            mlflow.sklearn.log_model(model, artifact_path="model")

            run_id = mlflow.active_run().info.run_id
            logger.info(f"{model_name} — ROC-AUC: {metrics['roc_auc']:.4f}")

            if metrics["roc_auc"] > best_auc:
                best_auc = metrics["roc_auc"]
                best_run_id = run_id

    # register best model
    model_uri = f"runs:/{best_run_id}/model"
    mlflow.register_model(model_uri=model_uri, name=config.mlflow.model_name)
    logger.info(f"Best model registered — ROC-AUC: {best_auc:.4f}, run_id: {best_run_id}")