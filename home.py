# home.py

import streamlit as st

def run():
    st.title("SepsisX")
    st.write("Detect Sepsis Early, Save Lives Instantly.")
    
    # Hero section
    st.markdown(
        "Real-time, AI-powered sepsis prediction and clinical support — "
        "helping hospitals take action before it’s too late.\n\n"
    )

    # Navigation buttons
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if st.button("🔍 Predict using ML"):
            st.session_state.page = "predict_ml"
            st.experimental_rerun()
    with col2:
        if st.button("🤖 Predict using DL"):
            st.session_state.page = "predict_dl"
            st.experimental_rerun()
    with col3:
        if st.button("🧬 Predict using QML"):
            st.session_state.page = "predict_qml"
            st.experimental_rerun()
    with col4:
        if st.button("⚕️ Predict using QNN"):
            st.session_state.page = "predict_qnn"
            st.experimental_rerun()

    st.write("---")
    st.header("Key Features of SepsisX")
    st.markdown("""
    - **Real‑time Risk Prediction**: Instant analysis of vitals, lab data, and symptoms.  
    - **AI‑Powered Engine**: ANN & XGBoost for accuracy and explainability.  
    - **Smart Alerts**: Threshold notifications via SMS, email, dashboard.  
    - **Automated Reports**: PDF logs, history, treatment suggestions.  
    - **Multilingual Support**: English, Spanish, Japanese.
    """)

    st.write("---")
    st.header("Why Early Detection Matters")
    st.markdown("> Every hour of delay increases mortality risk by **7%**.")
    stats = {
        "49 M": "New cases/yr globally",
        "11 M": "Annual deaths globally",
        "3×": "Improved survival with early detection",
        "20 %": "ICU stays involve undetected sepsis",
        "$24 B": "Annual US treatment cost"
    }
    cols = st.columns(len(stats))
    for (val, label), col in zip(stats.items(), cols):
        col.metric(label, val)

    st.write("---")
    st.header("Lab Tests & Treatment Protocols")
    ls1, ls2 = st.columns(2)
    with ls1:
        st.subheader("Critical Lab Tests")
        st.markdown(
            "- **Blood Cultures** – Identify infections\n"
            "- **Lactate Level** – Tissue oxygenation\n"
            "- **Procalcitonin (PCT)** – Bacterial marker\n"
            "- **CRP** – Inflammation\n"
            "- **CBC** – WBC count\n"
            "- **Metabolic Panel** – Kidney/Liver\n"
            "- **Coagulation** – DIC check\n"
            "- **Urinalysis** – UTI source\n"
            "- **Imaging** – X‑ray/CT for source"
        )
    with ls2:
        st.subheader("Evidence‑Based Treatments")
        st.markdown(
            "- **IV Antibiotics** within 1 hr\n"
            "- **Fluids** 30 mL/kg within 3 hrs\n"
            "- **Vasopressors**: MAP ≥ 65 mm Hg\n"
            "- **Oxygen/Ventilation** if needed\n"
            "- **Source Control**: drain/​surgery\n"
            "- **Steroids** for refractory shock\n"
            "- **Renal Replacement** if kidney fails\n"
            "- **Continuous Monitoring**: vitals, labs"
        )

    st.write("---")
    st.markdown("© 2025 SepsisX — Early detection saves lives")
