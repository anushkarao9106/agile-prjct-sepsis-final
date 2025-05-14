import streamlit as st

# Title and subtitle
st.title("SepsisX - Predict Sepsis Risk (Deep Learning)")
st.markdown("### Sepsis Risk Prediction (Deep Learning)")

# Create columns for layout
col1, col2 = st.columns(2)

# Left-side inputs
with col1:
    PRG = st.number_input("Plasma Glucose (PRG - mg/dL):", step=0.01, format="%.2f")
    PL = st.number_input("Blood Work Result-1 (PL - mu U/ml):", step=0.01, format="%.2f")
    PR = st.number_input("Blood Pressure (PR - mm Hg):", step=0.01, format="%.2f")
    SK = st.number_input("Blood Work Result-2 (SK - mm):", step=0.01, format="%.2f")
    Insurance = st.number_input("Insurance (1 for Yes, 0 for No):", min_value=0, max_value=1)

# Right-side inputs
with col2:
    TS = st.number_input("Blood Work Result-3 (TS - mu U/ml):", step=0.01, format="%.2f")
    M11 = st.number_input("Body Mass Index (M11 - kg/m²):", step=0.01, format="%.2f")
    BD2 = st.number_input("Blood Work Result-4 (BD2 - mu U/ml):", step=0.01, format="%.2f")
    Age = st.number_input("Age (years):", min_value=0, step=1)

# Predict button
if st.button("Predict"):
    # Replace this with your actual model prediction code
    # Example:
    # prediction = model.predict([[PRG, PL, PR, SK, Insurance, TS, M11, BD2, Age]])
    
    prediction = "High Risk of Sepsis"  # Dummy output
    treatment_suggestion = "Immediate medical attention recommended."

    st.success(prediction)
    st.info(treatment_suggestion)

# Back button (just a placeholder since there's no route navigation in Streamlit)
if st.button("Back"):
    st.markdown("Go back to the [Home](#) section.")
