import streamlit as st
import requests
from config import API_URL

def render():
    st.title("📂 Session Management")
    if st.button("Create Session"):
        # Call backend
        pass
