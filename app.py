import streamlit as st
import sqlite3
import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model
import joblib
import pennylane as qml
from pennylane import numpy as pnp

# Import your page modules
from pages.login import login
from pages.signup import signup
from pages.home import home
from pages.predict_dl import predict_dl
from pages.predict_ml import predict_ml
from pages.predict_qml import predict_qml
from pages.predict_qml import predict_qnn

# ─── Load Models & Scalers ─────────────────────────────────────────────────────
# Deep Learning model
dl_model = load_model('models/sepsis_dl_model.h5')

# Standard scaler
scaler = joblib.load('models/scaler.pkl')

# Selector for QML
selector = joblib.load('models/selector_qml.pkl')
# Machine Learning model
ml_model = joblib.load('models/sepsis_ml_model.pkl')

# Quantum‑ML parameters (update filenames as per your actual files)
qml_params = joblib.load('models/sepsis_qml.pkl')
qnn_params = joblib.load('models/sepsis_qnn_model_params.pkl')

# ─── Quantum Device & Circuits ────────────────────────────────────────────────
n_qubits = 4
dev = qml.device('default.qubit', wires=n_qubits)

@qml.qnode(dev)
def qml_qnode(x, params):
    for i in range(n_qubits):
        qml.RY(x[i], wires=i)
    qml.templates.StronglyEntanglingLayers(params, wires=range(n_qubits))
    return qml.expval(qml.PauliZ(0))

@qml.qnode(dev)
def qnn_qnode(x, params):
    for i in range(n_qubits):
        qml.RY(x[i], wires=i)
    qml.templates.BasicEntanglerLayers(params, wires=range(n_qubits))
    return qml.expval(qml.PauliZ(0))

# ─── Database Initialization ──────────────────────────────────────────────────
def init_db():
    with sqlite3.connect('users.db') as conn:
        conn.execute(''' 
            CREATE TABLE IF NOT EXISTS users ( 
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password TEXT NOT NULL
            ) 
        ''')

init_db()

# ─── Authentication ─────────────────────────────────────────────────────────
def login():
    st.title('Login')
    username = st.text_input('Username')
    password = st.text_input('Password', type='password')

    if st.button('Login'):
        with sqlite3.connect('users.db') as conn:
            cur = conn.cursor()
            cur.execute("SELECT * FROM users WHERE username=? AND password=?", (username, password))
            if cur.fetchone():
                st.session_state.username = username
                st.success("Logged in successfully!")
                return True
            else:
                st.error("Invalid username or password.")
    return False

def signup():
    st.title('Signup')
    username = st.text_input('Username')
    password = st.text_input('Password', type='password')

    if st.button('Sign up'):
        with sqlite3.connect('users.db') as conn:
            try:
                conn.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, password))
                st.success('User registered successfully!')
            except sqlite3.IntegrityError:
                st.error('Username already exists!')

# ─── Prediction Helper ─────────────────────────────────────────────────────────
def parse_and_scale(form):
    vals = [
        float(form[k]) for k in
        ['PRG', 'PL', 'PR', 'SK', 'TS', 'M11', 'BD2', 'Age', 'Insurance']
    ]
    arr = np.array([vals])
    return scaler.transform(arr)

def format_result(score):
    pct = round(score * 100, 2)
    if score >= 0.5:
        return f'High Risk of Sepsis ({pct}%)', "Immediate medical intervention is necessary. Please consult a healthcare professional."
    else:
        return f'Low Risk of Sepsis ({pct}%)', "Follow-up monitoring is recommended. Maintain a healthy lifestyle and schedule regular check-ups."

# ─── Prediction Routes ─────────────────────────────────────────────────────────
def predict_dl():
    if 'username' not in st.session_state:
        st.warning('Please log in first.')
        return

    st.title('Deep Learning Prediction')
    form = st.form(key='dl_form')
    PRG = form.number_input('PRG')
    PL = form.number_input('PL')
    PR = form.number_input('PR')
    SK = form.number_input('SK')
    TS = form.number_input('TS')
    M11 = form.number_input('M11')
    BD2 = form.number_input('BD2')
    Age = form.number_input('Age')
    Insurance = form.number_input('Insurance')

    submit = form.form_submit_button('Submit')
    if submit:
        try:
            X = parse_and_scale({
                'PRG': PRG, 'PL': PL, 'PR': PR, 'SK': SK,
                'TS': TS, 'M11': M11, 'BD2': BD2, 'Age': Age, 'Insurance': Insurance
            })
            score = dl_model.predict(X)[0][0]
            result, suggestion = format_result(score)
            st.write(result)
            st.write(suggestion)
        except Exception as e:
            st.error(f"Error: {e}")

def predict_ml():
    if 'username' not in st.session_state:
        st.warning('Please log in first.')
        return

    st.title('Machine Learning Prediction')
    form = st.form(key='ml_form')
    PRG = form.number_input('PRG')
    PL = form.number_input('PL')
    PR = form.number_input('PR')
    SK = form.number_input('SK')
    TS = form.number_input('TS')
    M11 = form.number_input('M11')
    BD2 = form.number_input('BD2')
    Age = form.number_input('Age')
    Insurance = form.number_input('Insurance')

    submit = form.form_submit_button('Submit')
    if submit:
        try:
            X = parse_and_scale({
                'PRG': PRG, 'PL': PL, 'PR': PR, 'SK': SK,
                'TS': TS, 'M11': M11, 'BD2': BD2, 'Age': Age, 'Insurance': Insurance
            })
            proba = ml_model.predict_proba(X)[0, 1]
            result, suggestion = format_result(proba)
            st.write(result)
            st.write(suggestion)
        except Exception as e:
            st.error(f"Error: {e}")

def predict_qml():
    if 'username' not in st.session_state:
        st.warning('Please log in first.')
        return

    st.title('Quantum ML Prediction')
    form = st.form(key='qml_form')
    PRG = form.number_input('PRG')
    PL = form.number_input('PL')
    PR = form.number_input('PR')
    SK = form.number_input('SK')
    TS = form.number_input('TS')
    M11 = form.number_input('M11')
    BD2 = form.number_input('BD2')
    Age = form.number_input('Age')
    Insurance = form.number_input('Insurance')

    submit = form.form_submit_button('Submit')
    if submit:
        try:
            vals = [
                PRG, PL, PR, SK, TS, M11, BD2, Age, Insurance
            ]
            arr = np.array([vals])

            selector = joblib.load('models/selector_qml.pkl')  
            selected = selector.transform(arr)

            scaler_qml = joblib.load('models/scaler_qml.pkl')
            scaled = scaler_qml.transform(selected)

            params = joblib.load('models/sepsis_qml.pkl')
            qml_pred = qml_qnode(scaled[0], params)

            score = (1 - qml_pred) / 2
            result, suggestion = format_result(score)

            st.write(result)
            st.write(suggestion)
        except Exception as e:
            st.error(f"Error: {e}")

def predict_qnn():
    if 'username' not in st.session_state:
        st.warning('Please log in first.')
        return

    st.title('Quantum Neural Network Prediction')
    form = st.form(key='qnn_form')
    PRG = form.number_input('PRG')
    PL = form.number_input('PL')
    PR = form.number_input('PR')
    SK = form.number_input('SK')
    TS = form.number_input('TS')
    M11 = form.number_input('M11')
    BD2 = form.number_input('BD2')
    Age = form.number_input('Age')
    Insurance = form.number_input('Insurance')

    submit = form.form_submit_button('Submit')
    if submit:
        try:
            X = parse_and_scale({
                'PRG': PRG, 'PL': PL, 'PR': PR, 'SK': SK,
                'TS': TS, 'M11': M11, 'BD2': BD2, 'Age': Age, 'Insurance': Insurance
            })[0]
            x_q = X[:4]
            exp = qnn_qnode(x_q, qnn_params)
            score = (exp + 1) / 2
            result, suggestion = format_result(score)
            st.write(result)
            st.write(suggestion)
        except Exception as e:
            st.error(f"Error: {e}")

# ─── Main Flow ───────────────────────────────────────────────────────────────
def main():
    st.set_page_config(page_title="Sepsis Prediction", page_icon="🩺")

    # Check if user is logged in
    if 'username' not in st.session_state:
        option = st.sidebar.selectbox('Choose an option', ['Login', 'Signup'])
        if option == 'Login':
            if login():
                st.sidebar.success('Login successful')
        else:
            signup()
    else:
        # After login, show the home page with prediction options
        st.sidebar.title('Prediction Options')
        option = st.sidebar.selectbox('Choose Prediction Model', ['DL', 'ML', 'QML', 'QNN'], index=0)
        if option == 'DL':
            predict_dl()
        elif option == 'ML':
            predict_ml()
        elif option == 'QML':
            predict_qml()
        elif option == 'QNN':
            predict_qnn()

if __name__ == '__main__':
    main()
