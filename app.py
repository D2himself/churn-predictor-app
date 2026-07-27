import streamlit as st
import pandas as pd
import joblib

pipeline = joblib.load('churn_pipeline.joblib')

st.title("Customer Churn Predictor")

tenure = st.slider("Tenure (months)", 0, 72, 12)
monthly_charges = st.slider("Monthly charges ($)", 18.0, 120.0, 65,0)
contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
internet_service = st.selectbox("Internet service", ["DSL", "Fiber optic", "No"])

# build on row matching the training columns. Any column not asked
# above gets a reasonable default value, since the pipeline needs
# every column it was trained on to be present.
row = pd.DataFrame([{
    "gender": "Female", "SeniorCitizen": 0, "Partner": "No", "Dependents": "No",
    "tenure": tenure, "PhoneService": "Yes", "MultipleLines": "No", 
    "OnlineBackup": "No", "DeviceProtection": "No", "TechSupport": "No",
    "StreamingTV": "No", "StreamingMovies": "No", "Contract": contract,
    "PaperlessBilling": "Yes", "PaymentMethod": "Electronic check",
    "MonthlyCharges": monthly_charges, "TotalCharges": monthly_charges * tenure,

}])

if st.button("Predict"):
    proba = pipeline.predict_proba(row)[0, 1]
    st.write(f"Predicted churn probability: {proba:.1%}")
    if proba >= 0.3:
        st.write("Flagged as at risk of churning.")
    else:
        st.write("Not flagged as at risk.")
