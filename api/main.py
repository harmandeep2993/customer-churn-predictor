# api/main.py

from fastapi import FastAPI, HTTPException
from api.schemas import CustomerInput, PredictionOutput
from api.services import run_prediction
from src.utils import get_logger

logger = get_logger(__name__)

app = FastAPI(
    title="Telco Churn Prediction API",
    description="Predict customer churn probability using ML models.",
    version="1.0.0"
)


@app.get("/health")
def health_check():
    """Check if API is running."""
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionOutput)
def predict(customer: CustomerInput):
    """Predict churn probability for a single customer."""
    try:
        result = run_prediction(customer)
        return result
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))