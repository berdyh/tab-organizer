import streamlit as st
import requests
import os
import pandas as pd
from config import API_URL

st.set_page_config(
    page_title="Web Scraping & Clustering Tool",
    page_icon="🕸️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Dashboard", "Settings", "Sessions", "URL Input", "Scraping Status", "Analysis", "Clustering", "Chatbot", "Export", "Services Health"])

st.sidebar.markdown("---")
st.sidebar.info(f"Connected to: {API_URL}")

if "session_id" in st.session_state:
    st.sidebar.success(f"Session: {st.session_state['session_id']}")
else:
    st.sidebar.warning("No Session Selected")

def get_health_status():
    try:
        response = requests.get(f"{API_URL}/health")
        if response.status_code == 200:
            return response.json()
        return None
    except:
        return None

if page == "Dashboard":
    st.title("🕸️ Web Scraping & Clustering Tool")
    status = get_health_status()
    if status:
        st.success("System is connected and healthy.")
        st.json(status)
    else:
        st.error(f"Cannot connect to API Gateway at {API_URL}")

elif page == "Settings":
    from pages import settings
    settings.render()

elif page == "Services Health":
    st.title("❤️ Services Health")
    if st.button("Refresh Status"):
        st.rerun()
    try:
        response = requests.get(f"{API_URL}/services")
        if response.status_code == 200:
            st.json(response.json())
        else:
            st.error("Failed.")
    except Exception as e:
        st.error(f"Error: {e}")

elif page == "Sessions":
    from pages import sessions
    sessions.render()

elif page == "URL Input":
    from pages import url_input
    url_input.render()

elif page == "Scraping Status":
    from pages import scraping
    scraping.render()

elif page == "Analysis":
    from pages import analysis
    analysis.render()

elif page == "Clustering":
    from pages import clustering
    clustering.render()

elif page == "Chatbot":
    from pages import chatbot
    chatbot.render()

elif page == "Export":
    from pages import export
    export.render()
