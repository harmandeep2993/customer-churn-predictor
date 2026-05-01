# main.py

from src.data.loader import loader_pipeline
from src.data import preprocess_pipeline
from src.features import build_features
from src.models.train import train_pipeline
from src.utils import get_logger

logger = get_logger(__name__)


def main():
    logger.info("Starting Telco Churn Prediction Pipeline...")

    # Step 1: Load
    logger.info("Step 1: Loading raw data...")
    df = loader_pipeline()

    # Step 2: Preprocess
    logger.info("Step 2: Preprocessing data...")
    df = preprocess_pipeline(df)

    # Step 3: Feature engineering
    logger.info("Step 3: Building features...")
    df = build_features(df)

    # Step 4: Train + log to MLflow
    logger.info("Step 4: Training models and logging to MLflow...")
    train_pipeline(df)

    logger.info("Pipeline completed successfully.")


if __name__ == "__main__":
    main()