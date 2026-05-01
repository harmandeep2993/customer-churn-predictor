import sys
from pathlib import Path
import pandas as pd
import streamlit as st

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from extras.components.model_predictor import ModelPredictor

# Initialize predictor and load model
predictor = ModelPredictor()
model = predictor.load_model()

# Title
st.title("Customer Churn Prediction App")
st.write("Predict whether a telecom customer is likely to churn based on their profile and service details.")

# ==========================================================
# Input form
# ==========================================================
with st.form("churn_form"):
    st.subheader("Customer Information")

    # --- Row 1: Demographics ---
    col1, col2, col3 = st.columns(3)
    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
    with col2:
        senior = st.selectbox("Senior Citizen", ["Yes", "No"])
    with col3:
        dependents = st.selectbox("Dependents", ["Yes", "No"])

    # --- Row 2: Account Info ---
    col4, col5, col6 = st.columns(3)
    with col4:
        partner = st.selectbox("Partner", ["Yes", "No"])
    with col5:
        tenure = st.slider("Tenure (Months)", 0, 72, 12)
    with col6:
        contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])

    # --- Row 3: Services ---
    st.subheader("Services Subscribed")
    col7, col8, col9 = st.columns(3)
    with col7:
        phone_service = st.selectbox("Phone Service", ["Yes", "No"])
        multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
        internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    with col8:
        online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
        online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
        device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
    with col9:
        tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
        streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
        streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])

    # --- Row 4: Billing Info ---
    st.subheader("Billing and Charges")
    col10, col11, col12 = st.columns(3)
    with col10:
        paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
    with col11:
        payment_method = st.selectbox(
            "Payment Method",
            ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
        )
    with col12:
        monthly_charges = st.number_input("Monthly Charges ($)", 0.0, 200.0, 70.0)
        total_charges = st.number_input("Total Charges ($)", 0.0, 10000.0, 2500.0)

    submitted = st.form_submit_button("Predict Churn")

# ==========================================================
# Prediction section
# ==========================================================
if submitted:
    input_data = {
        "gender": gender,
        "SeniorCitizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
    }

    st.subheader("Entered Customer Details")
    st.dataframe(pd.DataFrame([input_data]))

    # Run prediction
    input_df = pd.DataFrame([input_data])
    pred, prob = predictor.predict_churn(input_df, model)

    st.subheader("Prediction Result")
    if pred[0] == 1:
        st.error("Customer is likely to churn.")
    else:
        st.success("Customer is not likely to churn.")

    st.metric("Churn Probability", f"{prob[0]*100:.2f}%")