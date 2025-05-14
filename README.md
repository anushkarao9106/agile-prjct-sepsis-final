#  SepsisX: AI-Powered Sepsis Risk Prediction

SepsisX is a clinical decision support system that uses an **Artificial Neural Network (ANN)** model to predict the risk of sepsis in ICU patients based on clinical input parameters. This project includes model training, evaluation, deployment-ready backend, and prediction functionality.

---

## Project Structure

```
sepsisx_model/
│
├── train_model.py             # Trains the ANN model using patient data
├── test_model.py              # Evaluates the trained model on test data
├── predict_single_input.py    # Predicts sepsis risk for a custom input
├── model_utils.py             # Utility to load model and scaler
├── sepsis_model.h5            # Saved trained ANN model (generated after training)
├── scaler.pkl                 # Saved StandardScaler (generated after training)
├── requirements.txt           # Python package dependencies
├── README.md                  # You're here!
└── Patients_Files_Train.csv   # Training dataset (you must add this file)
```

---

##  Dependencies

Make sure to install the required libraries:

```bash
pip install -r requirements.txt
```

---

##  Steps to Run

###  Train the Model

This script loads the CSV data, preprocesses it, builds an ANN, trains the model, and saves it to disk.

```bash
python train_model.py
```

 Output:
- `sepsis_model.h5`: Trained model  
- `scaler.pkl`: StandardScaler for future predictions

---

###  Evaluate the Model

Run evaluation metrics (Accuracy, Precision, Recall, F1 Score, Confusion Matrix) on test split from the same dataset.

```bash
python test_model.py
```

Output:
- Console output of classification performance  
- Confusion matrix printout

---

###  Predict on Custom Input

Use this to manually input values for a single patient and receive the prediction.

```bash
python predict_single_input.py
```

You’ll be prompted to enter:

```
PRG, PL, PR, SK, TS, M11, BD2, Age, Insurance
```

 Output:
- Probability of Sepsis
- Whether it's a **High Risk** or **Low Risk**

---

###  Re-training with Updated Dataset

Just replace the `Patients_Files_Train.csv` file with an updated version and rerun:

```bash
python train_model.py
```

---

## Model Summary

- **Architecture**: ANN (Dense: 64 → 32 → 1)
- **Activations**: ReLU (hidden), Sigmoid (output)
- **Loss Function**: Binary Crossentropy
- **Optimizer**: Adam
- **Accuracy Achieved**: ~74.17%
- **F1 Score**: ~61.73%

---

## Input Features

| Feature    | Description                          |
|------------|--------------------------------------|
| PRG        | Plasma Glucose (mg/dL)               |
| PL         | Plasma Lipid Profile Score           |
| PR         | Pulse Rate (bpm)                     |
| SK         | Skin Thickness (mm)                  |
| TS         | Temperature (°C)                     |
| M11        | Inflammatory Marker (e.g. CRP)       |
| BD2        | Blood Pressure Differential (mmHg)   |
| Age        | Age in years                         |
| Insurance  | Binary (0 = no insurance, 1 = yes)   |

---

##  Notes

- Make sure the dataset file `Patients_Files_Train.csv` is present in the same folder.
- You can integrate this model with a Flask web app (`app.py`) for GUI-based sepsis prediction.