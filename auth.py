import streamlit as st
from db import add_user, verify_user

# ---------------- LOGIN FUNCTION ----------------
def login():

    if "username" not in st.session_state:
        st.session_state["username"] = None

    st.title("🔐 Login / Register")

    menu = st.selectbox("Select Option", ["Login", "Register"])

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if menu == "Register":
        if st.button("Register"):
            if add_user(username, password):
                st.success("Account created successfully! Please login.")
            else:
                st.error("Username already exists.")

    if menu == "Login":
        if st.button("Login"):
            if verify_user(username, password):
                st.session_state["username"] = username
                st.success("Login successful!")
                return True
            else:
                st.error("Invalid username or password.")

    if st.session_state["username"]:
        return True

    return False
