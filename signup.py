import streamlit as st

st.set_page_config(page_title="Signup", layout="centered")

# Page title
st.markdown("<h2 style='text-align: center;'>Signup</h2>", unsafe_allow_html=True)

# Placeholder for signup message
msg = ""

# Signup form
with st.form("signup_form"):
    username = st.text_input("Choose Username")
    password = st.text_input("Choose Password", type="password")
    submit = st.form_submit_button("Sign Up")

# Handle form submission
if submit:
    if username and password:
        # Replace with your actual signup logic (e.g., database store)
        msg = f"Signup successful for user: {username}"
    else:
        msg = "Please fill in all fields."

# Display message
if msg:
    st.info(msg)

# Link to login
st.markdown("Already have an account? [Login](/login)")
