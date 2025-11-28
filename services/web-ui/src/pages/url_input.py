import streamlit as st
import requests
from config import API_URL

def render():
    st.title("🔗 URL Input")
    urls = st.text_area("Enter URLs")
    if st.button("Submit"):
        # Call backend
        pass
