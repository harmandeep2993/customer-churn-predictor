# src/models/evaluate.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    roc_auc_score,
    ConfusionMatrixDisplay
)
import mlflow

from src.utils import config, get_logger

logger = get_logger(__name__)

REPORTS_DIR = config.paths.reports_dir


def plot_confusion_matrix(model, X_test, y_test, model_name: str) -> Path:
    """Plot and save confusion matrix."""
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(cm, display_labels=["No Churn", "Churn"]).plot(ax=ax, colorbar=False)
    ax.set_title(f"Confusion Matrix — {model_name}")

    path = Path(REPORTS_DIR) / f"confusion_matrix_{model_name}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)

    logger.info(f"Confusion matrix saved to {path}")
    return path


def plot_roc_curve(model, X_test, y_test, model_name: str) -> Path:
    """Plot and save ROC curve."""
    y_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
    ax.plot([0, 1], [0, 1], "k--", label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve — {model_name}")
    ax.legend()

    path = Path(REPORTS_DIR) / f"roc_curve_{model_name}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)

    logger.info(f"ROC curve saved to {path}")
    return path


def print_classification_report(model, X_test, y_test, model_name: str) -> None:
    """Log classification report to logger."""
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, target_names=["No Churn", "Churn"])
    logger.info(f"\nClassification Report — {model_name}:\n{report}")


def evaluate_pipeline(model, X_test, y_test, model_name: str) -> None:
    """Run full evaluation and log artifacts to active MLflow run."""
    print_classification_report(model, X_test, y_test, model_name)

    cm_path = plot_confusion_matrix(model, X_test, y_test, model_name)
    roc_path = plot_roc_curve(model, X_test, y_test, model_name)

    mlflow.log_artifact(str(cm_path))
    mlflow.log_artifact(str(roc_path))

    logger.info(f"Evaluation complete for {model_name}.")