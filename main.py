from src.data.loader import loader_pipeline
from src.data.preprocess import preprocess_pipeline
from src.features.builder import build_features, save_artifacts
from src.models.train import train_pipeline
from src.utils import get_logger

logger = get_logger(__name__)


def main():
    logger.info("Starting Telco Churn Prediction Pipeline...")

    logger.info("Step 1: Loading raw data...")
    df = loader_pipeline()

    logger.info("Step 2: Preprocessing data...")
    df = preprocess_pipeline(df)

    logger.info("Step 3: Building features...")
    df, scaler, feature_columns = build_features(df)
    save_artifacts(scaler, feature_columns)

    logger.info("Step 4: Training models and logging to MLflow...")
    train_pipeline(df)

    logger.info("Pipeline completed successfully.")


if __name__ == "__main__":
    main()