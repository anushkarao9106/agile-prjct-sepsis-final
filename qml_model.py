# qml_model.py

import pennylane as qml
from pennylane import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score
import joblib

# ─── 1. Load & Preprocess ───────────────────────────────────────────────────────
print("⏳ Loading dataset and preprocessing...")
data = pd.read_csv('Paitients_Files_Train.csv')
data['Sepssis'] = data['Sepssis'].map({'Positive': 1, 'Negative': 0})

X_all = data.drop(columns=['ID', 'Sepssis']).values
y_all = data['Sepssis'].values

# Select top 4 features out of 9
selector = SelectKBest(score_func=f_classif, k=4)
X_selected = selector.fit_transform(X_all, y_all)
joblib.dump(selector, 'selector_qml.pkl')
print("✔ Saved feature selector to models/selector_qml.pkl")

# Split into train / test
X_train, X_test, y_train, y_test = train_test_split(
    X_selected, y_all, test_size=0.2, random_state=42
)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)
joblib.dump(scaler, 'scaler_qml.pkl')
print("✔ Saved scaler to models/scaler_qml.pkl")

# ─── 2. Quantum Circuit Setup ───────────────────────────────────────────────────
n_qubits = X_train.shape[1]  # should be 4
dev = qml.device("default.qubit", wires=n_qubits)

def qaoa_circuit(params, x):
    # data encoding
    for i in range(n_qubits):
        qml.RY(x[i], wires=i)
    # parameterized layer
    p     = params[:n_qubits]
    gamma = params[n_qubits:]
    for i in range(n_qubits):
        qml.RZ(gamma[i], wires=i)
        qml.RX(p[i],      wires=i)
    return qml.expval(qml.PauliZ(0))

@qml.qnode(dev)
def qnode(x, params):
    return qaoa_circuit(params, x)

# ─── 3. QAOA Model Class ─────────────────────────────────────────────────────────
class QAOA:
    def __init__(self, steps=50, lr=0.01):
        self.steps = steps
        self.lr    = lr
        # initialize 2*n_qubits parameters
        self.params = np.random.randn(2 * n_qubits)

    def fit(self, X, y):
        opt   = qml.AdamOptimizer(stepsize=self.lr)
        y_arr = np.array(y)

        for i in range(self.steps):
            # cost over entire training set
            def cost_fn(p):
                loss = 0
                for j in range(len(X)):
                    ev = qnode(X[j], p)
                    # map label y in {0,1} to target exp in {1,-1}: t = 1 - 2*y
                    target = 1 - 2 * y_arr[j]
                    loss += (ev - target) ** 2
                return loss / len(X)

            self.params = opt.step(cost_fn, self.params)
            if (i + 1) % 10 == 0 or i == 0:
                print(f"  • Step {i+1}/{self.steps}")

    def predict(self, X):
        preds = []
        for x in X:
            ev = qnode(x, self.params)
            # class 1 if expectation < 0
            preds.append(1 if ev < 0 else 0)
        return np.array(preds)

# ─── 4. Train & Save ─────────────────────────────────────────────────────────────
print("⏳ Training QAOA model...")
model = QAOA(steps=50, lr=0.01)
model.fit(X_train, y_train)

# Evaluate on test set
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"✅ QAOA Test Accuracy: {acc * 100:.2f}%")

# Save trained QAOA parameters
joblib.dump(model.params, 'sepsis_qml.pkl')
print("✔ Saved QAOA parameters to models/sepsis_qml.pkl")
