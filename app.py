import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Load model and scaler
@st.cache_data
def load_model():
    df = pd.read_csv("diabetes.csv")
    X = df.drop("Outcome", axis=1)
    y = df["Outcome"]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = LogisticRegression()
    model.fit(X_scaled, y)
    return model, scaler, X.columns

model, scaler, feature_names = load_model()

# Streamlit UI
st.title("🩺 Diabetes Prediction App")
st.markdown("Enter patient health data below:")

# Input form
user_input = []
for col in feature_names:
    val = st.number_input(f"{col}", min_value=0.0)
    user_input.append(val)

if st.button("Predict"):
    input_scaled = scaler.transform([user_input])
    pred = model.predict(input_scaled)[0]
    prob = model.predict_proba(input_scaled)[0][1]
    if pred == 1:
        st.error(f"🔴 Likely Diabetic (Risk: {prob:.2f})")
    else:
        st.success(f"🟢 Not Likely Diabetic (Risk:{prob:.2f})")
