import streamlit as st
import joblib

st.title("Credit Wise - Loan Prediction")

from pathlib import Path
import joblib

model_path = Path(__file__).parent / "credit_model.pkl"
model = joblib.load(model_path)

st.write("Enter the required applicant details.")

# Replace these with your actual model features
feature1 = st.number_input("Feature 1", value=0.0)
feature2 = st.number_input("Feature 2", value=0.0)

if st.button("Predict"):
    prediction = model.predict([[feature1, feature2]])
    st.success(f"Prediction: {prediction[0]}")
