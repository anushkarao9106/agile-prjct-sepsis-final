import streamlit as st

st.set_page_config(page_title="SepsisX - Predict Sepsis Risk (QNN)", layout="centered")

st.markdown("""
    <style>
    .main {
        background-color: #f9f9f9;
        padding: 20px;
        border-radius: 10px;
    }
    .title {
        text-align: center;
        font-weight: bold;
        font-size: 24px;
        margin-bottom: 20px;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main">', unsafe_allow_html=True)
st.markdown('<div class="title">Sepsis Risk Prediction (Quantum Neural Network)</div>', unsafe_allow_html=True)

with st.form("sepsis_qnn_form"):
    col1, col2 = st.columns(2)

    with col1:
        PRG = st.number_input("Plasma Glucose (PRG - mg/dL):", step=0.01, format="%.2f")
        PL = st.number_input("Blood Work Result‑1 (PL - μU/ml):", step=0.01, format="%.2f")
        PR = st.number_input("Blood Pressure (PR - mm Hg):", step=0.01, format="%.2f")
        SK = st.number_input("Blood Work Result‑2 (SK - mm):", step=0.01, format="%.2f")
        Insurance = st.number_input("Insurance (1 for Yes, 0 for No):", step=1, format="%d")

    with col2:
        TS = st.number_input("Blood Work Result‑3 (TS - μU/ml):", step=0.01, format="%.2f")
        M11 = st.number_input("Body Mass Index (M11 - kg/m²):", step=0.01, format="%.2f")
        BD2 = st.number_input("Blood Work Result‑4 (BD2 - μU/ml):", step=0.01, format="%.2f")
        Age = st.number_input("Age (years):", step=1, format="%d")

    submit = st.form_submit_button("Predict")

if submit:
    # Placeholder for model prediction logic
    prediction = "High Risk"  # or "Low Risk", etc.
    treatment_suggestion = "Immediate clinical evaluation is advised."

    st.success(f"Prediction: **{prediction}**")
    st.info(f"Treatment Suggestion: {treatment_suggestion}")

st.markdown('[Back to Home](#)', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)
