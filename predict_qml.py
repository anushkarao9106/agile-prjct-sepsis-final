import streamlit as st

# Title and Header
st.set_page_config(page_title="SepsisX - QML", layout="centered")
st.title("🧬 SepsisX")
st.markdown("## Sepsis Risk Prediction (Quantum Machine Learning)")

# Input form
with st.form("sepsis_qml_form"):
    st.markdown("### Patient Information")

    col1, col2 = st.columns(2)

    with col1:
        PRG = st.number_input("Plasma Glucose (PRG - mg/dL):", step=0.01, format="%.2f")
        PL = st.number_input("Blood Work Result-1 (PL - mu U/ml):", step=0.01, format="%.2f")
        PR = st.number_input("Blood Pressure (PR - mm Hg):", step=0.01, format="%.2f")
        SK = st.number_input("Blood Work Result-2 (SK - mm):", step=0.01, format="%.2f")
        Insurance = st.number_input("Insurance (1 for Yes, 0 for No):", min_value=0, max_value=1)

    with col2:
        TS = st.number_input("Blood Work Result-3 (TS - mu U/ml):", step=0.01, format="%.2f")
        M11 = st.number_input("Body Mass Index (M11 - kg/m²):", step=0.01, format="%.2f")
        BD2 = st.number_input("Blood Work Result-4 (BD2 - mu U/ml):", step=0.01, format="%.2f")
        Age = st.number_input("Age (years):", step=1.0, format="%.0f")

    submitted = st.form_submit_button("Predict")

# Handling prediction (mockup, replace with your actual model call)
if submitted:
    # Replace this section with your actual QML prediction logic
    st.success("✅ Prediction: High Risk of Sepsis")
    st.info("Suggested Treatment: Immediate ICU admission and monitoring.")

# Back to Home (as a navigation link or note)
st.markdown("---")
st.markdown("[⬅️ Back to Home](/home)")
