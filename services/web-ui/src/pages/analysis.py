import streamlit as st
import requests
from config import API_URL, get_api_headers

def render():
    st.title("🧠 Content Analysis")
    if "session_id" not in st.session_state: return
    session_id = st.session_state["session_id"]

    if st.button("Start Analysis"):
        requests.post(f"{API_URL}/api/analyzer-service/analyze", json={"session_id": session_id}, headers=get_api_headers())
        st.success("Started")
