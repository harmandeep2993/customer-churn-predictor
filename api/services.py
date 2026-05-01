# api/services.py

from src.models.predict import predict_pipeline
from api.schemas import CustomerInput, PredictionOutput
from src.utils import get_logger

logger = get_logger(__name__)


def run_prediction(customer: CustomerInput) -> PredictionOutput:
    """Convert Pydantic input to dict, run prediction, return output."""
    data = customer.model_dump()
    result = predict_pipeline(data)
    logger.info(f"Prediction served: {result}")
    return PredictionOutput(**result)