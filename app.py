import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load trained model
model = joblib.load("failure_prediction_model.joblib")

st.title("Machine Failure Prediction")
st.write(
    "This demo predicts the likelihood of machine failure based on sensor readings. "
    "The model was trained on the [UCI Machine Predictive Maintenance dataset]"
    "(https://archive.ics.uci.edu/dataset/601/predictive+maintenance)."
)


# Create two columns
col1, col2 = st.columns(2)

with col1:
    machine_type = st.selectbox("Machine Type", ["L", "M", "H"])
    air_temp = st.slider("Air temperature [K]", min_value=290, max_value=320, value=300, step=1)
    process_temp = st.slider("Process temperature [K]", min_value=300, max_value=350, value=310, step=1)
    

with col2:
    rot_speed = st.slider("Rotational speed [rpm]", min_value=1000, max_value=3000, value=1500, step=10)
    torque = st.slider("Torque [Nm]", min_value=10, max_value=100, value=40, step=1)
    tool_wear = st.slider("Tool wear [min]", min_value=1, max_value=50, value=10, step=1)
    
    
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
