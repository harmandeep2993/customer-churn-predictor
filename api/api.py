from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from extras.components.model_predictor import ModelPredictor

app = FastAPI(title="Customer Churn Prediction API", version="1.0")

# load model once at startup
predictor = ModelPredictor()
model = predictor.load_model()

class CustomerInput(BaseModel):
    gender: str
    SeniorCitizen: str
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float

@app.get("/")
def root():
    return {"status": "API running", "message": "Use POST /predict for predictions."}

@app.post("/predict")
def predict(data: CustomerInput):
    predictor = ModelPredictor()
    model = predictor.load_model()

    # Convert input to DataFrame
    input_df = pd.DataFrame([data.dict()])
    pred, prob = predictor.predict_churn(input_df, model)

    result = {
        "prediction": "Churn" if pred[0] == 1 else "No Churn",
        "probability": round(float(prob[0]), 4)
    }
    return result