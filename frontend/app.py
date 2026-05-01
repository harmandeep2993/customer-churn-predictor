# frontend/app.py

import httpx
from nicegui import ui

API_URL = "http://localhost:8000/predict"

RISK_FACTORS = {
    "Contract": {"Month-to-month": "High risk", "One year": "Medium risk", "Two year": "Low risk"},
    "InternetService": {"Fiber optic": "High risk", "DSL": "Medium risk", "No": "Low risk"},
    "TechSupport": {"No": "High risk", "Yes": "Low risk", "No internet service": "Low risk"},
    "OnlineSecurity": {"No": "High risk", "Yes": "Low risk", "No internet service": "Low risk"},
}

BADGE_COLORS = {
    "High risk": "background:#FCEBEB;color:#A32D2D;",
    "Medium risk": "background:#FAEEDA;color:#633806;",
    "Low risk": "background:#EAF3DE;color:#3B6D11;",
}


def create_app():

    with ui.column().style("width:100%;max-width:1300px;margin:0.5rem auto 0.75rem;padding:0 1rem;"):
        ui.label("Customer Churn Predictor").style("font-size:28px;font-weight:600;color:#185FA5;")
        ui.label("Enter customer details below to predict churn probability.").style("font-size:13px;color:gray;margin-top:-8px;")

    with ui.row().style("width:100%;max-width:1300px;margin:0rem auto;gap:16px;padding:0 1rem;align-items:start;"):

        # ── Left: form ───────────────────────────────────────────
        with ui.card().style("flex:1;padding:1.25rem;"):

            # High impact fields
            ui.label("HIGH IMPACT FIELDS").style("font-size:11px;font-weight:600;color:#185FA5;letter-spacing:0.07em;margin-bottom:6px;")
            with ui.grid(columns=4).style("width:100%;gap:8px;margin-bottom:0.75rem;"):
                contract = ui.select(label="Contract", options=["Month-to-month", "One year", "Two year"], value="Month-to-month").style("width:100%;")
                tenure = ui.number(label="Tenure (mo.)", value=12, min=0, max=72).style("width:100%;")
                monthly = ui.number(label="Monthly ($)", value=70.5, min=0).style("width:100%;")
                total = ui.number(label="Total ($)", value=846.0, min=0).style("width:100%;")
                internet = ui.select(label="Internet service", options=["Fiber optic", "DSL", "No"], value="Fiber optic").style("width:100%;")
                online_sec = ui.select(label="Online security", options=["Yes", "No", "No internet service"], value="No").style("width:100%;")
                tech_support = ui.select(label="Tech support", options=["Yes", "No", "No internet service"], value="No").style("width:100%;")
                paperless = ui.select(label="Paperless billing", options=["Yes", "No"], value="Yes").style("width:100%;")
                payment = ui.select(label="Payment method", options=["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"], value="Electronic check").style("width:100%;")
                online_bkp = ui.select(label="Online backup", options=["Yes", "No", "No internet service"], value="No").style("width:100%;")
                device_prot = ui.select(label="Device protection", options=["Yes", "No", "No internet service"], value="No").style("width:100%;")

            ui.separator()

            # Low impact fields
            ui.label("LOW IMPACT FIELDS").style("font-size:11px;font-weight:600;color:gray;letter-spacing:0.07em;margin-top:0.5rem;margin-bottom:6px;")
            with ui.grid(columns=4).style("width:100%;gap:8px;margin-bottom:0.75rem;"):
                gender = ui.select(label="Gender", options=["Male", "Female"], value="Male").style("width:100%;")
                senior = ui.select(label="Senior citizen", options={0: "No", 1: "Yes"}, value=0).style("width:100%;")
                partner = ui.select(label="Partner", options=["Yes", "No"], value="Yes").style("width:100%;")
                dependents = ui.select(label="Dependents", options=["Yes", "No"], value="No").style("width:100%;")
                phone = ui.select(label="Phone service", options=["Yes", "No"], value="Yes").style("width:100%;")
                multilines = ui.select(label="Multiple lines", options=["Yes", "No", "No phone service"], value="No").style("width:100%;")
                streaming_tv = ui.select(label="Streaming TV", options=["Yes", "No", "No internet service"], value="Yes").style("width:100%;")
                streaming_mov = ui.select(label="Streaming movies", options=["Yes", "No", "No internet service"], value="Yes").style("width:100%;")

            ui.button("Predict churn", on_click=lambda: predict()).style(
                "width:100%;margin-top:0.75rem;background:#185FA5;color:white;"
            )

        # ── Right: results panel ─────────────────────────────────
        with ui.column().style("width:300px;gap:12px;"):

            with ui.card().style("width:100%;padding:1rem;"):
                ui.label("PREDICTION RESULT").style("font-size:10px;font-weight:500;color:gray;letter-spacing:0.07em;margin-bottom:10px;")
                prob_number = ui.label("—").style("font-size:28px;font-weight:500;color:gray;text-align:center;width:100%;")
                ui.label("Churn probability").style("font-size:11px;color:gray;text-align:center;width:100%;margin-bottom:10px;")
                result_banner = ui.label("").style("display:none;")

            with ui.card().style("width:100%;padding:1rem;"):
                ui.label("RISK FACTORS").style("font-size:10px;font-weight:500;color:gray;letter-spacing:0.07em;margin-bottom:8px;")
                risk_rows = {}
                for factor in RISK_FACTORS:
                    with ui.row().style("width:100%;justify-content:space-between;align-items:center;padding:5px 0;border-bottom:0.5px solid #eee;"):
                        ui.label(factor).style("font-size:11px;color:gray;")
                        risk_rows[factor] = ui.label("—").style("font-size:10px;padding:2px 8px;border-radius:4px;background:#f0f0f0;color:gray;")

    async def predict():
        payload = {
            "gender": gender.value,
            "SeniorCitizen": senior.value,
            "Partner": partner.value,
            "Dependents": dependents.value,
            "tenure": int(tenure.value),
            "PhoneService": phone.value,
            "MultipleLines": multilines.value,
            "InternetService": internet.value,
            "OnlineSecurity": online_sec.value,
            "OnlineBackup": online_bkp.value,
            "DeviceProtection": device_prot.value,
            "TechSupport": tech_support.value,
            "StreamingTV": streaming_tv.value,
            "StreamingMovies": streaming_mov.value,
            "Contract": contract.value,
            "PaperlessBilling": paperless.value,
            "PaymentMethod": payment.value,
            "MonthlyCharges": float(monthly.value),
            "TotalCharges": float(total.value),
        }

        async with httpx.AsyncClient() as client:
            response = await client.post(API_URL, json=payload)
            data = response.json()

        prob = data["churn_probability"]
        label = data["churn_label"]

        prob_number.set_text(f"{prob:.1%}")

        if label == "Churn":
            prob_number.style("font-size:28px;font-weight:500;color:#E24B4A;text-align:center;width:100%;")
            result_banner.set_text("Churn predicted — customer is likely to churn")
            result_banner.style("display:block;padding:10px;border-radius:8px;background:#FCEBEB;color:#A32D2D;font-size:12px;font-weight:500;margin-top:8px;")
        else:
            prob_number.style("font-size:28px;font-weight:500;color:#639922;text-align:center;width:100%;")
            result_banner.set_text("No churn predicted — customer is likely to stay")
            result_banner.style("display:block;padding:10px;border-radius:8px;background:#EAF3DE;color:#3B6D11;font-size:12px;font-weight:500;margin-top:8px;")

        field_map = {
            "Contract": contract.value,
            "InternetService": internet.value,
            "TechSupport": tech_support.value,
            "OnlineSecurity": online_sec.value,
        }
        for factor, badges in RISK_FACTORS.items():
            risk = badges.get(field_map[factor], "Low risk")
            color = BADGE_COLORS[risk]
            risk_rows[factor].set_text(risk)
            risk_rows[factor].style(f"font-size:10px;padding:2px 8px;border-radius:4px;{color}")


create_app()
ui.run(port=8501, title="Telco Churn Prediction")