import streamlit as st
import pandas as pd
import pickle

# Load dataset and models
df = pd.read_csv("hospital_appts_full_corrected.csv")
with open("scripts_model.pkl", "rb") as f:
    scripts_model = pickle.load(f)
with open("crowd_model.pkl", "rb") as f:
    crowd_model = pickle.load(f)

# Streamlit app
st.title("Hospital Demand Predictor")
st.write("Predict pharmacy scripts and crowd level for hospital clinics")

# Input widgets
specialty = st.selectbox("Specialty", ["Oncology", "Cardiology", "Neurology", "Orthopedics", "Endocrinology"])
location = st.selectbox("Location", ["Main", "City2", "City3"])
appt_count = st.slider("Appointment Count", 5, 60, 20)
consult_ratio = st.slider("Consult Ratio", 0.3, 0.7, 0.69, step=0.01)
duration = st.slider("Average Duration (min)", 25, 60, 52)

# Predict
if st.button("Predict"):
    scripts_pred = scripts_model.predict([[appt_count, consult_ratio]])
    crowd_pred = crowd_model.predict([[appt_count, consult_ratio, duration]])
    st.write(f"**Predicted Scripts**: {int(scripts_pred[0])}")
    st.write(f"**Crowd Level**: {'Crowded' if crowd_pred[0] == 1 else 'Normal'}")
    if crowd_pred[0] == 1:
        st.write("Send greeters to manage waiting room!")
    else:
        st.write("Waiting room expected to be manageable.")
