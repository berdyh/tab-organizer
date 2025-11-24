import streamlit as st
import requests
import pandas as pd
from config import API_URL

def render():
    st.title("📂 Session Management")

    st.subheader("Create New Session")
    new_session_name = st.text_input("Session Name")
    if st.button("Create Session"):
        if new_session_name:
            try:
                response = requests.post(f"{API_URL}/api/session-service/sessions", json={"name": new_session_name})
                if response.status_code in [200, 201]:
                    st.success(f"Session '{new_session_name}' created.")
                    # Changed key to 'id' based on review
                    st.session_state["session_id"] = response.json().get("id")
                    st.rerun()
                else:
                    st.error(f"Failed to create session: {response.text}")
            except Exception as e:
                st.error(f"Error: {e}")
        else:
            st.warning("Please enter a session name.")

    st.subheader("Select Session")
    try:
        response = requests.get(f"{API_URL}/api/session-service/sessions")
        if response.status_code == 200:
            sessions = response.json()
            if sessions:
                # Changed key to 'id'
                session_options = {s["name"]: s["id"] for s in sessions}
                selected_name = st.selectbox("Choose a session", list(session_options.keys()))
                if st.button("Select"):
                    st.session_state["session_id"] = session_options[selected_name]
                    st.success(f"Selected session: {selected_name}")
            else:
                st.info("No sessions found.")
        else:
            st.error("Failed to fetch sessions.")
    except Exception as e:
        st.error(f"Error fetching sessions: {e}")

    if "session_id" in st.session_state:
        st.info(f"Current Session ID: {st.session_state['session_id']}")
