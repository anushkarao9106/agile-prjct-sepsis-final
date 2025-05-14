import streamlit as st

# Page Title
st.title("SepsisX - Predict Sepsis Risk (ML Model)")
st.markdown("### Sepsis Risk Prediction (ML Model)")

# Input Form
st.subheader("Enter the following details to predict sepsis risk:")

# Left column inputs
col1, col2 = st.columns(2)

# Left Column
with col1:
    PRG = st.number_input("Plasma Glucose (PRG - mg/dL):", step=0.01)
    PL = st.number_input("Blood Work Result‑1 (PL - mu U/ml):", step=0.01)
    PR = st.number_input("Blood Pressure (PR - mm Hg):", step=0.01)
    SK = st.number_input("Blood Work Result‑2 (SK - mm):", step=0.01)
    Insurance = st.number_input("Insurance (1 for Yes, 0 for No):", min_value=0, max_value=1)

# Right Column
with col2:
    TS = st.number_input("Blood Work Result‑3 (TS - mu U/ml):", step=0.01)
    M11 = st.number_input("Body Mass Index (M11 - kg/m²):", step=0.01)
    BD2 = st.number_input("Blood Work Result‑4 (BD2 - mu U/ml):", step=0.01)
    Age = st.number_input("Age (years):", min_value=0)

# Predict Button
if st.button("Predict"):
    # Placeholder for the prediction logic
    # For now, replace this with your actual ML model prediction code
    prediction = "Low Risk of Sepsis"  # Dummy prediction
    treatment_suggestion = "Continue monitoring patient vitals regularly."  # Dummy suggestion

    # Display results
    st.success(prediction)
    st.info(treatment_suggestion)

# Back to Home Button (For demo purposes, it doesn't go anywhere)
if st.button("Back to Home"):
    st.write("You can return to the home section.")
