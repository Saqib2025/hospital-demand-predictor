import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import numpy as np

# Initialize session state to store inputs
if 'appt_count' not in st.session_state:
    st.session_state.appt_count = 20
if 'consult_ratio' not in st.session_state:
    st.session_state.consult_ratio = 0.69
if 'duration' not in st.session_state:
    st.session_state.duration = 52
if 'specialty' not in st.session_state:
    st.session_state.specialty = "Oncology"
if 'location' not in st.session_state:
    st.session_state.location = "Main"
if 'prediction_time' not in st.session_state:
    st.session_state.prediction_time = pd.to_datetime("09:00").time()

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
specialty = st.selectbox("Clinic Specialty", ["Oncology", "Cardiology", "Neurology", "Orthopedics", "Endocrinology"], key="specialty")
location = st.selectbox("Clinic Location", ["Main", "City2", "City3"], key="location")
appt_count = st.number_input("Number of Appointments", min_value=5, max_value=60, value=20, step=1, key="appt_count")
consult_ratio = st.number_input("Consultation Ratio", min_value=0.3, max_value=0.7, value=0.69, step=0.01, key="consult_ratio", format="%.2f")
duration = st.number_input("Average Appointment Duration (min)", min_value=25, max_value=60, value=52, step=1, key="duration")
prediction_time = st.time_input("Prediction Time", value=pd.to_datetime("09:00").time(), step=3600, key="prediction_time")

# Predict
if st.button("Predict"):
    # Log inputs for debugging
    st.write(f"**Debug Inputs**: Specialty: {specialty}, Location: {location}, Appt Count: {appt_count}, Consult Ratio: {consult_ratio}, Duration: {duration}, Time: {prediction_time.strftime('%I:%M %p')}")

    # Update session state
    st.session_state.appt_count = appt_count
    st.session_state.consult_ratio = consult_ratio
    st.session_state.duration = duration
    st.session_state.specialty = specialty
    st.session_state.location = location
    st.session_state.prediction_time = prediction_time

    # Pharmacy Scripts Prediction with Coding Error Buffer
    scripts_pred = scripts_model.predict([[appt_count, consult_ratio]])
    scripts_base = int(scripts_pred[0])
    coding_error_rate = 0.05  # Assume 5% of appointments might have coding errors
    error_buffer = int(scripts_base * 0.1)  # ±10% buffer for coding errors
    scripts_lower = max(0, scripts_base - error_buffer)
    scripts_upper = scripts_base + error_buffer
    st.write(f"**At {prediction_time.strftime('%I:%M %p')} in {specialty} ({location}):**")
    st.write(f"**Expected Pharmacy Scripts**: {scripts_base} (Range: {scripts_lower}–{scripts_upper} due to potential coding errors)")

    # Common Medicine Supplies
    med_supplies = {
        "Oncology": ["Chemotherapy drugs", "Antiemetics"],
        "Cardiology": ["Statins", "Beta-blockers"],
        "Neurology": ["Antiepileptics", "Migraine meds"],
        "Orthopedics": ["NSAIDs", "Pain relievers"],
        "Endocrinology": ["Insulin", "Thyroid meds"]
    }
    st.write(f"**Common Medicine Supplies**: {', '.join(med_supplies[specialty])}")

    # Waiting Room Status
    crowd_pred = crowd_model.predict([[appt_count, consult_ratio, duration]])
    crowd_prob = crowd_model.predict_proba([[appt_count, consult_ratio, duration]])
    if appt_count >= 35:
        crowd_pred[0] = 1
        crowd_prob[0] = [0.2, 0.8]
    st.write(f"**Waiting Room Status**: {'Crowded' if crowd_pred[0] == 1 else 'Normal'}")
    st.write(f"**Probability of Crowding**: Crowded: {crowd_prob[0][1]:.2f}, Normal: {crowd_prob[0][0]:.2f}")

    # Nurse Needs
    extra_nurses = 0
    if crowd_pred[0] == 1:
        extra_nurses = max(1, (appt_count - 35) // 15 + 1)
        st.write(f"**Nurse Staffing**: Add {extra_nurses} extra nurse{'s' if extra_nurses > 1 else ''} to manage the crowd.")
    patients_vitals = int(appt_count * 0.5)
    patients_injections = int(appt_count * 0.2)
    total_support_tasks = patients_vitals + patients_injections
    support_nurses = max(1, total_support_tasks // 20)
    st.write(f"**Supportive Nurses**: Assign {support_nurses} nurse{'s' if support_nurses > 1 else ''} for ~{patients_vitals} patients needing vitals and ~{patients_injections} needing injections.")

    # Pharmacy Process
    st.write("**Pharmacy Process**:")
    st.write(f"1. **Preparation**: Stock {', '.join(med_supplies[specialty])}; prepare for {scripts_lower}–{scripts_upper} scripts.")
    st.write(f"2. **Verification**: Cross-check billing codes for prescriptions against {specialty} specialty.")
    if crowd_pred[0] == 1:
        st.write("3. **Dispensing**: Waiting room is Crowded—assign extra staff to the pharmacy counter; inform patients of potential delays.")
    else:
        st.write("3. **Dispensing**: Waiting room is Normal—proceed with standard dispensing; inform patients of expected wait times.")

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
    ax1.set_ylim(0, max(scripts_pred[0] + 5, 10))
    st.pyplot(fig1)

    # Visualization 2: Line chart showing how scripts vary with Consultation Ratio
    st.write("**How Consultation Ratio Affects Scripts (for your appointment count)**")
    consult_range = np.arange(0.3, 0.71, 0.05)
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
