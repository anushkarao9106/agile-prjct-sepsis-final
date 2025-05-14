import pennylane as qml
from pennylane import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import joblib
import os

# Load and preprocess dataset
data = pd.read_csv('Paitients_Files_Train.csv')
data['Sepssis'] = data['Sepssis'].map({'Positive': 1, 'Negative': 0})
X = data.drop(columns=['ID', 'Sepssis'])
y = data['Sepssis']

# Train–test split and feature scaling
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Save the scaler in current folder
joblib.dump(scaler, 'scaler_qnn.pkl')

# Keep only first 4 features
X_train = X_train[:, :4]
X_test = X_test[:, :4]

# Quantum device
n_qubits = 4
dev = qml.device("default.qubit", wires=n_qubits)

# Quantum layer + entanglement
def quantum_layer(params, x):
    for i in range(n_qubits):
        qml.RY(x[i], wires=i)
    qml.templates.BasicEntanglerLayers(params, wires=range(n_qubits))

@qml.qnode(dev)
def qnode(x, params):
    quantum_layer(params, x)
    return qml.expval(qml.PauliZ(0))

# QNN wrapper
class QNN:
    def __init__(self, steps=50, lr=0.2):
        self.steps = steps
        self.lr = lr
        self.params = np.random.randn(4, n_qubits)

    def predict_single(self, x):
        raw = qnode(x, self.params)  # after training this is a numpy scalar
        return 1 if raw < 0 else 0

    def predict(self, X):
        return [self.predict_single(x) for x in X]

    def fit(self, X, y):
        opt = qml.GradientDescentOptimizer(stepsize=self.lr)
        y = np.array(y)

        def cost_fn(params):
            loss = 0
            for xi, yi in zip(X, y):
                pred = qnode(xi, params)      # symbolic ArrayBox during grad
                target = 1 - 2 * yi            # map {0,1} -> {1,-1}
                loss += (pred - target) ** 2
            return loss / len(X)

        for _ in range(self.steps):
            self.params = opt.step(cost_fn, self.params)

# Train & evaluate
model = QNN(steps=50, lr=0.2)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("QNN Accuracy:", accuracy_score(y_test, y_pred))

# Save parameters in current folder
joblib.dump(model.params, 'sepsis_qnn_model_params.pkl')
