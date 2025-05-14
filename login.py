import streamlit as st

# Set the title of the app
st.title("Login")

# Display login form
username = st.text_input("Username")
password = st.text_input("Password", type="password")

# Handle form submission
if st.button("Login"):
    if username == "your_username" and password == "your_password":  # Replace with actual authentication logic
        st.success("Login successful!")
    else:
        st.error("Invalid username or password.")

# Provide a signup link
st.write("Don't have an account? [Sign up](#)")  # Replace '#' with the actual signup link
