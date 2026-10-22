
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

BASE_DIR = Path(__file__).parent

# Load trained model and preprocessing objects
model = joblib.load(BASE_DIR / "credit_model.pkl")
scaler = joblib.load(BASE_DIR / "scaler.pkl")
ohe = joblib.load(BASE_DIR / "encoder.pkl")

st.set_page_config(
    page_title="Credit Wise",
    page_icon="💳",
    layout="wide"
)

st.title("💳 Credit Wise - Loan Approval Prediction")
st.write("Enter the applicant's details to predict loan approval.")

with st.form("loan_form"):
    st.subheader("Financial Information")

    col1, col2 = st.columns(2)

    with col1:
        income = st.number_input(
            "Applicant Income", min_value=0.0, value=15000.0
        )
        co_income = st.number_input(
            "Coapplicant Income", min_value=0.0, value=0.0
        )
        age = st.number_input(
            "Age", min_value=18, max_value=100, value=30
        )
        dependents = st.number_input(
            "Dependents", min_value=0, value=0, step=1
        )
        credit_score = st.number_input(
            "Credit Score", min_value=0, max_value=900, value=700
        )
        existing_loans = st.number_input(
            "Existing Loans", min_value=0, value=0, step=1
        )

    with col2:
        dti = st.number_input(
            "DTI Ratio", min_value=0.0, max_value=1.0, value=0.30
        )
        savings = st.number_input(
            "Savings", min_value=0.0, value=10000.0
        )
        collateral = st.number_input(
            "Collateral Value", min_value=0.0, value=0.0
        )
        loan_amount = st.number_input(
            "Loan Amount", min_value=0.0, value=15000.0
        )
        loan_term = st.number_input(
            "Loan Term", min_value=1, value=360, step=1
        )

    st.subheader("Applicant Details")

    col3, col4 = st.columns(2)

    with col3:
        employment = st.selectbox(
            "Employment Status",
            ["Contract", "Salaried", "Self-employed", "Unemployed"]
        )
        marital = st.selectbox(
            "Marital Status", ["Married", "Single"]
        )
        purpose = st.selectbox(
            "Loan Purpose",
            ["Business", "Car", "Education", "Home", "Personal"]
        )

    with col4:
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
    # Education_Level was label-encoded in the notebook.
    # LabelEncoder orders these classes alphabetically.
    education_encoded = 0 if education == "Graduate" else 1

    # Applicant_ID is intentionally excluded.
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
        "Education_Level": education_encoded
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
        # Apply the fitted OneHotEncoder
        encoded_array = ohe.transform(categorical)

        encoded = pd.DataFrame(
            encoded_array,
            columns=ohe.get_feature_names_out(),
            index=numeric.index
        )

        # Match the feature layout used in the notebook
        features = pd.concat([numeric, encoded], axis=1)

        # Enforce exact training feature names and order
        expected_columns = list(scaler.feature_names_in_)
        features = features.reindex(columns=expected_columns)

        # Scale and predict
        scaled_features = scaler.transform(features)
        prediction = model.predict(scaled_features)[0]

        if prediction == 1:
            st.success("Prediction: Loan Approved")
        else:
            st.error("Prediction: Loan Not Approved")

    except Exception as e:
        st.error(f"Prediction error: {e}")
