# api/schemas.py

from pydantic import BaseModel, Field


class CustomerInput(BaseModel):
    gender: str = Field(..., example="Male")
    SeniorCitizen: int = Field(..., example=0)
    Partner: str = Field(..., example="Yes")
    Dependents: str = Field(..., example="No")
    tenure: int = Field(..., example=12)
    PhoneService: str = Field(..., example="Yes")
    MultipleLines: str = Field(..., example="No")
    InternetService: str = Field(..., example="Fiber optic")
    OnlineSecurity: str = Field(..., example="No")
    OnlineBackup: str = Field(..., example="No")
    DeviceProtection: str = Field(..., example="No")
    TechSupport: str = Field(..., example="No")
    StreamingTV: str = Field(..., example="Yes")
    StreamingMovies: str = Field(..., example="Yes")
    Contract: str = Field(..., example="Month-to-month")
    PaperlessBilling: str = Field(..., example="Yes")
    PaymentMethod: str = Field(..., example="Electronic check")
    MonthlyCharges: float = Field(..., example=70.5)
    TotalCharges: float = Field(..., example=846.0)


class PredictionOutput(BaseModel):
    churn_probability: float
    churn_prediction: int
    churn_label: str