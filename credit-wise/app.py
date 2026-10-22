
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

BASE_DIR = Path(__file__).parent

model = joblib.load(BASE_DIR / "credit_model.pkl")
scaler = joblib.load(BASE_DIR / "scaler.pkl")
ohe = joblib.load(BASE_DIR / "encoder.pkl")

st.set_page_config(page_title="Credit Wise", page_icon="💳")
st.title("Credit Wise - Loan Approval Prediction")
st.write("Enter the applicant's details below.")

with st.form("loan_form"):
    st.subheader("Applicant Information")

    income = st.number_input("Applicant Income", min_value=0.0, value=15000.0)
    co_income = st.number_input("Coapplicant Income", min_value=0.0, value=0.0)
    employment = st.selectbox(
        "Employment Status",
        ["Contract", "Salaried", "Self-employed", "Unemployed"]
    )
    age = st.number_input("Age", min_value=18, max_value=100, value=30)
    marital = st.selectbox("Marital Status", ["Married", "Single"])
    dependents = st.number_input("Dependents", min_value=0, value=0, step=1)
    credit_score = st.number_input(
        "Credit Score", min_value=0, max_value=900, value=700
    )
    existing_loans = st.number_input(
        "Existing Loans", min_value=0, value=0, step=1
    )
    dti = st.number_input(
        "DTI Ratio", min_value=0.0, max_value=1.0, value=0.30
    )
    savings = st.number_input("Savings", min_value=0.0, value=10000.0)
    collateral = st.number_input(
        "Collateral Value", min_value=0.0, value=0.0
    )
    loan_amount = st.number_input(
        "Loan Amount", min_value=0.0, value=15000.0
    )
    loan_term = st.number_input(
        "Loan Term", min_value=1, value=360, step=1
    )
    purpose = st.selectbox(
        "Loan Purpose", ["Business", "Car", "Education", "Home", "Personal"]
    )
    area = st.selectbox(
        "Property Area", ["Rural", "Semiurban", "Urban"]
    )
    education = st.selectbox(
        "Education Level", ["Graduate", "Not Graduate"]
    )
    gender = st.selectbox("Gender", ["Female", "Male"])
    employer = st.selectbox(
        "Employer Category",
        ["Business", "Government", "MNC", "Private", "Unemployed"]
    )

    submitted = st.form_submit_button("Predict Loan Approval")

if submitted:
    # Match the numerical feature order used during training.
    numeric = pd.DataFrame([{
        "Applicant_Income": income,
        "Coapplicant_Income": co_income,
        "Age": age,
        "Dependents": dependents,
        "Credit_Score": credit_score,
        "Existing_Loans": existing_loans,
        "DTI_Ratio": dti,
        "Savings": savings,
        "Collateral_Value": collateral,
        "Loan_Amount": loan_amount,
        "Loan_Term": loan_term,
        "Education_Level": 0 if education == "Graduate" else 1
    }])

    categorical = pd.DataFrame([{
        "Employment_Status": employment,
        "Marital_Status": marital,
        "Loan_Purpose": purpose,
        "Property_Area": area,
        "Gender": gender,
        "Employer_Category": employer
    }])

    try:
        encoded_array = ohe.transform(categorical)
        encoded = pd.DataFrame(
            encoded_array,
            columns=ohe.get_feature_names_out(),
            index=numeric.index
        )

        features = pd.concat([numeric, encoded], axis=1)
        scaled_features = scaler.transform(features)
        prediction = model.predict(scaled_features)[0]

        if prediction == 1:
            st.success("Prediction: Loan Approved")
        else:
            st.error("Prediction: Loan Not Approved")

    except Exception as e:
        st.error(f"Prediction error: {e}")
