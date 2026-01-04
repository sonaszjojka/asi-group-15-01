import os
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.title("Income Prediction (ML Demo)")

st.header("Input features")

age = st.number_input("age", min_value=0, value=39)
workclass = st.text_input("workclass", value="State-gov")
fnlwgt = st.number_input("fnlwgt", min_value=0, value=77516)
education = st.text_input("education", value="Bachelors")
education_num = st.number_input("education-num", min_value=0, value=13)
marital_status = st.text_input("marital-status", value="Never-married")
occupation = st.text_input("occupation", value="Adm-clerical")
relationship = st.text_input("relationship", value="Not-in-family")
race = st.text_input("race", value="White")
sex = st.selectbox("sex", ["Male", "Female"])
capital_gain = st.number_input("capital-gain", min_value=0, value=2174)
capital_loss = st.number_input("capital-loss", min_value=0, value=0)
hours_per_week = st.number_input("hours-per-week", min_value=1, value=40)
native_country = st.text_input("native-country", value="United-States")

if st.button("Predict"):
    payload = {
        "age": int(age),
        "workclass": workclass,
        "fnlwgt": int(fnlwgt),
        "education": education,
        "education-num": int(education_num),
        "marital-status": marital_status,
        "occupation": occupation,
        "relationship": relationship,
        "race": race,
        "sex": sex,
        "capital-gain": int(capital_gain),
        "capital-loss": int(capital_loss),
        "hours-per-week": int(hours_per_week),
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
