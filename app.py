import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import numpy as np

# Load dataset and models
df = pd.read_csv("hospital_appts_full_corrected.csv")
with open("scripts_model.pkl", "rb") as f:
    scripts_model = pickle.load(f)
with open("crowd_model.pkl", "rb") as f:
    crowd_model = pickle.load(f)

# Streamlit app
st.title("Hospital Demand Predictor")
st.write("Predict pharmacy scripts, waiting room status, and resource needs for hospital clinics")

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

    # Nurse Needs
    extra_nurses = 0
    if crowd_pred[0] == 1:
        extra_nurses = max(1, (appt_count - 35) // 15 + 1)  # 1 nurse per 15 appts above threshold
        st.write(f"**Nurse Staffing**: Add {extra_nurses} extra nurse{'s' if extra_nurses > 1 else ''} to manage the crowd.")
    # Supportive nurses for vitals and injections
    patients_vitals = int(appt_count * 0.5)  # 50% need vitals
    patients_injections = int(appt_count * 0.2)  # 20% need injections
    total_support_tasks = patients_vitals + patients_injections
    support_nurses = max(1, total_support_tasks // 20)  # 1 nurse per 20 tasks
    st.write(f"**Supportive Nurses**: Assign {support_nurses} nurse{'s' if support_nurses > 1 else ''} for ~{patients_vitals} patients needing vitals and ~{patients_injections} needing injections.")

    # Scans and Lab Tests
    scan_ratio = {"Oncology": 0.4, "Cardiology": 0.3, "Neurology": 0.3, "Orthopedics": 0.5, "Endocrinology": 0.2}
    lab_ratio = {"Oncology": 0.6, "Cardiology": 0.5, "Neurology": 0.4, "Orthopedics": 0.3, "Endocrinology": 0.7}
    expected_scans = int(appt_count * scan_ratio[specialty])
    expected_labs = int(appt_count * lab_ratio[specialty])
    st.write(f"**Expected Diagnostics**: ~{expected_scans} scans and ~{expected_labs} lab tests.")

    # Actionable Guidance
    if crowd_pred[0] == 1:
        st.write("**Action**: Send greeters to manage the waiting room!")
    else:
        st.write("**Action**: Waiting room should be manageable; no extra staff needed.")

    # Visualization 1: Bar chart for the specific prediction
    st.write("**Your Prediction at a Glance**")
    fig1, ax1 = plt.subplots(figsize=(4, 3))
    ax1.bar([appt_count], [scripts_pred[0]], color='skyblue', width=2)
    ax1.set_xlabel("Number of Appointments")
    ax1.set_ylabel("Expected Pharmacy Scripts")
    ax1.set_title(f"Scripts for {appt_count} Appointments")
    ax1.set_xticks([appt_count])
    ax1.set_ylim(0, max(scripts_pred[0] + 5, 10))  # Adjust y-axis for visibility
    st.pyplot(fig1)

    # Visualization 2: Line chart showing how scripts vary with Consultation Ratio
    st.write("**How Consultation Ratio Affects Scripts (for your appointment count)**")
    consult_range = np.arange(0.3, 0.71, 0.05)  # From 0.3 to 0.7, step 0.05
    scripts_vary = [scripts_model.predict([[appt_count, cr]])[0] for cr in consult_range]
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    ax2.plot(consult_range, scripts_vary, marker='o', color='green', label=f'{appt_count} Appointments')
    ax2.plot(consult_ratio, scripts_pred[0], marker='*', color='red', markersize=15, label=f'Current: {consult_ratio:.2f}, {int(scripts_pred[0])} Scripts')
    ax2.set_xlabel("Consultation Ratio")
    ax2.set_ylabel("Expected Pharmacy Scripts")
    ax2.set_title(f"Scripts vs. Consultation Ratio ({appt_count} Appointments)")
    ax2.grid(True)
    ax2.legend()
    st.pyplot(fig2)
