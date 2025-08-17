import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load trained model
model = joblib.load("failure_prediction_model.joblib")

st.title("Machine Failure Prediction App")
st.write("Enter the machine sensor readings to predict if it will fail.")

# Create input fields for all features
air_temp = st.number_input("Air temperature [K]", value=300.0)
process_temp = st.number_input("Process temperature [K]", value=310.0)
rot_speed = st.number_input("Rotational speed [rpm]", value=1500.0)
torque = st.number_input("Torque [Nm]", value=40.0)
tool_wear = st.number_input("Tool wear [min]", value=10.0)

# Machine Type — from your dummy variables
machine_type = st.selectbox("Machine Type", ["L", "M", "H"])

# Convert Machine Type to dummy variables like in training
type_L = 1 if machine_type == "L" else 0
type_M = 1 if machine_type == "M" else 0
type_H = 1 if machine_type == "H" else 0

if st.button("Predict"):
    # Order of features must match training
    features = np.array([[air_temp, process_temp, rot_speed, torque, tool_wear,
                          type_H, type_L, type_M]])
    prediction = model.predict(features)[0]
    failure_prob = model.predict_proba(features)[0]

    if prediction == 1:
        st.error("⚠ Machine is likely to fail (failure probability: {:.2f}%)".format(failure_prob[1] * 100))
    else:
        st.success("✅ Machine is not likely to fail (failure probability: {:.2f}%)".format(failure_prob[1] * 100))
