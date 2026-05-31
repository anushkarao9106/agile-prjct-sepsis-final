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

* Compared Random Forest (ML), ANN (DL), and QNN (QML) models using Accuracy, Precision, Recall, and F1 Score.
* ANN achieved the highest predictive performance among all evaluated models.
* Best Results:

  * Accuracy: ~74.17%
  * F1 Score: ~61.73%
* ANN was selected as the final deployment model for real-time sepsis risk prediction.

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

* Comparative evaluation of Random Forest (ML), ANN (DL), and QNN (QML) identified ANN as the best-performing model, achieving ~74.17% accuracy and ~61.73% F1 Score.
* Random Forest served as the baseline ML model, while QNN was explored to assess the potential of quantum-enhanced learning.
* The system supports integration with a Flask-based web application for secure, real-time sepsis risk assessment.
* Ensure that `Patients_Files_Train.csv` is present in the project directory before training.
