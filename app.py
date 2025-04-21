import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt

# Load dataset and models
df = pd.read_csv("hospital_appts_full_corrected.csv")
with open("scripts_model.pkl", "rb") as f:
    scripts_model = pickle.load(f)
with open("crowd_model.pkl", "rb") as f:
    crowd_model = pickle.load(f)

# Streamlit app
st.title("Hospital Demand Predictor")
st.write("Predict pharmacy scripts and waiting room status for hospital clinics")

# Input widgets
specialty = st.selectbox("Clinic Specialty", ["Oncology", "Cardiology", "Neurology", "Orthopedics", "Endocrinology"])
location = st.selectbox("Clinic Location", ["Main", "City2", "City3"])
appt_count = st.slider("Number of Appointments", 5, 60, 20)
consult_ratio = st.slider("Consultation Ratio", 0.3, 0.7, 0.69, step=0.01)
duration = st.slider("Average Appointment Duration (min)", 25, 60, 52)
prediction_time = st.time_input("Prediction Time", value=pd.to_datetime("09:00").time(), step=3600)  # Hourly steps

# Predict
if st.button("Predict"):
    scripts_pred = scripts_model.predict([[appt_count, consult_ratio]])
    crowd_pred = crowd_model.predict([[appt_count, consult_ratio, duration]])
    crowd_prob = crowd_model.predict_proba([[appt_count, consult_ratio, duration]])
    # Hardcode Appt Count >= 35 rule
    if appt_count >= 35:
        crowd_pred[0] = 1
        crowd_prob[0] = [0.2, 0.8]
    # Display prediction with time and improved labels
    st.write(f"**At {prediction_time.strftime('%I:%M %p')} in {specialty} ({location}):**")
    st.write(f"**Expected Pharmacy Scripts**: {int(scripts_pred[0])}")
    st.write(f"**Waiting Room Status**: {'Crowded' if crowd_pred[0] == 1 else 'Normal'}")
    st.write(f"**Probability of Crowding**: Crowded: {crowd_prob[0][1]:.2f}, Normal: {crowd_prob[0][0]:.2f}")
    if crowd_pred[0] == 1:
        st.write("**Action**: Send greeters to manage the waiting room!")
    else:
        st.write("**Action**: Waiting room should be manageable; no extra staff needed.")

    # Visualization: Scripts vs. Appointment Count
    st.write("**Trend Analysis: Expected Scripts vs. Number of Appointments**")
    appt_range = range(5, 61, 5)
    scripts_preds = [scripts_model.predict([[appt, consult_ratio]])[0] for appt in appt_range]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(appt_range, scripts_preds, marker='o', color='b', label=f'Consult Ratio: {consult_ratio}')
    ax.set_xlabel("Number of Appointments")
    ax.set_ylabel("Expected Pharmacy Scripts")
    ax.set_title("Expected Scripts vs. Number of Appointments")
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)
