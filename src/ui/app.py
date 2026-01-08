import os
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.title("Income Prediction (ML Demo)")
st.header("Input features")

WORKCLASS = [
    "Private",
    "Self-emp-not-inc",
    "Self-emp-inc",
    "Federal-gov",
    "Local-gov",
    "State-gov",
    "Without-pay",
    "Never-worked",
]

EDUCATION = [
    "Preschool",
    "1st-4th",
    "5th-6th",
    "7th-8th",
    "9th",
    "10th",
    "11th",
    "12th",
    "HS-grad",
    "Some-college",
    "Assoc-voc",
    "Assoc-acdm",
    "Bachelors",
    "Masters",
    "Prof-school",
    "Doctorate",
]

MARITAL_STATUS = [
    "Never-married",
    "Married-civ-spouse",
    "Divorced",
    "Separated",
    "Widowed",
    "Married-spouse-absent",
    "Married-AF-spouse",
]

OCCUPATION = [
    "Tech-support",
    "Craft-repair",
    "Other-service",
    "Sales",
    "Exec-managerial",
    "Prof-specialty",
    "Handlers-cleaners",
    "Machine-op-inspct",
    "Adm-clerical",
    "Farming-fishing",
    "Transport-moving",
    "Priv-house-serv",
    "Protective-serv",
    "Armed-Forces",
]

RELATIONSHIP = [
    "Wife",
    "Own-child",
    "Husband",
    "Not-in-family",
    "Other-relative",
    "Unmarried",
]

RACE = ["White", "Black", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other"]

NATIVE_COUNTRY = [
    "United-States",
    "Mexico",
    "Philippines",
    "Germany",
    "Canada",
    "India",
    "England",
    "China",
    "Japan",
    "South",
    "Cuba",
    "Italy",
    "Poland",
    "Columbia",
]


age = st.number_input("Age", min_value=17, max_value=100, value=39)
workclass = st.selectbox("Workclass", WORKCLASS, index=WORKCLASS.index("State-gov"))
fnlwgt = st.number_input("Final weight (fnlwgt)", min_value=0, value=77516)
education = st.selectbox("Education", EDUCATION, index=EDUCATION.index("Bachelors"))
marital_status = st.selectbox("Marital status", MARITAL_STATUS)
occupation = st.selectbox(
    "Occupation", OCCUPATION, index=OCCUPATION.index("Adm-clerical")
)
relationship = st.selectbox("Relationship", RELATIONSHIP)
race = st.selectbox("Race", RACE)
sex = st.selectbox("Sex", ["Male", "Female"])
capital_gain = st.number_input("Capital gain", min_value=0, value=2174)
capital_loss = st.number_input("Capital loss", min_value=0, value=0)
hours_per_week = st.number_input("Hours per week", min_value=1, max_value=99, value=40)
native_country = st.selectbox(
    "Native country",
    NATIVE_COUNTRY,
    index=NATIVE_COUNTRY.index("United-States"),
)


if st.button("Predict"):
    payload = {
        "age": age,
        "workclass": workclass,
        "fnlwgt": fnlwgt,
        "education": education,
        "marital-status": marital_status,
        "occupation": occupation,
        "relationship": relationship,
        "race": race,
        "sex": sex,
        "capital-gain": capital_gain,
        "capital-loss": capital_loss,
        "hours-per-week": hours_per_week,
        "native-country": native_country,
    }

    try:
        r = requests.post(
            f"{API_URL}/predict",
            json=payload,
            timeout=10,
        )
        st.write("Status:", r.status_code)
        st.json(r.json())
    except Exception as e:
        st.error(f"Request failed: {e}")
