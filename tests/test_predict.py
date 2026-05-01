# tests/test_predict.py

from src.models.predict import predict_pipeline


sample = {
    "gender": "Male", "SeniorCitizen": 0, "Partner": "Yes",
    "Dependents": "No", "tenure": 12, "PhoneService": "Yes",
    "MultipleLines": "No", "InternetService": "Fiber optic",
    "OnlineSecurity": "No", "OnlineBackup": "No",
    "DeviceProtection": "No", "TechSupport": "No",
    "StreamingTV": "Yes", "StreamingMovies": "Yes",
    "Contract": "Month-to-month", "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check", "MonthlyCharges": 70.5,
    "TotalCharges": 846.0
}


def test_predict_returns_dict():
    result = predict_pipeline(sample)
    assert isinstance(result, dict)


def test_predict_keys():
    result = predict_pipeline(sample)
    assert "churn_probability" in result
    assert "churn_prediction" in result
    assert "churn_label" in result


def test_predict_probability_range():
    result = predict_pipeline(sample)
    assert 0.0 <= result["churn_probability"] <= 1.0


def test_predict_label_valid():
    result = predict_pipeline(sample)
    assert result["churn_label"] in ["Churn", "No Churn"]