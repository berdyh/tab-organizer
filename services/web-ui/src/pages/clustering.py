import streamlit as st
import requests
import plotly.express as px
from config import API_URL, get_api_headers

def render():
    st.title("📊 Clustering")
    if "session_id" not in st.session_state: return
    session_id = st.session_state["session_id"]

    if st.button("Cluster"):
        requests.post(f"{API_URL}/api/clustering-service/cluster", json={"session_id": session_id}, headers=get_api_headers())
        st.success("Started")

    # Viz logic
    try:
        r = requests.get(f"{API_URL}/api/clustering-service/clusters/{session_id}")
        if r.status_code == 200:
            st.json(r.json())
    except: pass
