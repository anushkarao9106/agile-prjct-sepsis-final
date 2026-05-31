# SepsisX: AI-Powered Sepsis Risk Prediction

SepsisX is a clinical decision support system designed to predict the risk of sepsis in ICU patients using Machine Learning (ML), Deep Learning (DL), and exploratory Quantum Machine Learning (QML) techniques. The project implements and compares Random Forest, Artificial Neural Networks (ANN), and Quantum Neural Networks (QNN) to identify the most effective approach for early sepsis detection and clinical decision support. It includes data preprocessing, model training, evaluation, deployment-ready backend, and real-time prediction functionality.

---

## Dependencies

Install the required libraries:

```bash
pip install -r requirements.txt
```

---

## Model Summary

### Comparative Analysis

* Compared **Random Forest (ML)**, **ANN (DL)**, and **QNN (QML)** models using Accuracy, Precision, Recall, and F1 Score.
* **Random Forest achieved the highest predictive performance** among all evaluated models.
* Best Results:

  * Accuracy: **~84.17%**
  * F1 Score: **~75.73%**

---

## Technologies Used

* Python
* Scikit-learn
* TensorFlow
* Qiskit
* Flask
* SQLite

---

## Input Features

| Feature   | Description                        |
| --------- | ---------------------------------- |
| PRG       | Plasma Glucose (mg/dL)             |
| PL        | Plasma Lipid Profile Score         |
| PR        | Pulse Rate (bpm)                   |
| SK        | Skin Thickness (mm)                |
| TS        | Temperature (°C)                   |
| M11       | Inflammatory Marker (e.g., CRP)    |
| BD2       | Blood Pressure Differential (mmHg) |
| Age       | Age in years                       |
| Insurance | Binary (0 = no insurance, 1 = yes) |

---

## Notes

* Comparative evaluation of **Random Forest (ML)**, **ANN (DL)**, and **QNN (QML)** identified **Random Forest as the best-performing model**, achieving **~84.17% accuracy** and **~61.73% F1 Score**.
* ANN was implemented as the primary Deep Learning model, while QNN was explored to assess the potential of quantum-enhanced learning for healthcare prediction tasks.
* The system supports integration with a Flask-based web application for secure, real-time sepsis risk assessment.
* Ensure that `Patients_Files_Train.csv` is present in the project directory before training.

---

## Project Outcome

* Developed and evaluated ML, DL, and QML approaches for ICU sepsis risk prediction.
* Demonstrated that **Random Forest outperformed ANN and QNN** on the available dataset.
* Built a foundation for real-time clinical decision support through a deployable prediction pipeline.
