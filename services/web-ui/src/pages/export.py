import streamlit as st
import requests
from config import API_URL, get_api_headers

def render():
    st.title("📤 Export")
    if "session_id" not in st.session_state: return

    fmt = st.selectbox("Format", ["notion", "markdown"])
    if st.button("Export"):
        requests.post(f"{API_URL}/api/export-service/export", json={"session_id": st.session_state["session_id"], "format": fmt})
        st.success("Exported")
