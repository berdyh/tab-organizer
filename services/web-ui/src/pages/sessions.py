import streamlit as st
import requests
from config import API_URL, get_api_headers

def render():
    st.title("📂 Session Management")
    new_session_name = st.text_input("Session Name")
    if st.button("Create Session"):
        try:
            response = requests.post(f"{API_URL}/api/session-service/sessions", json={"name": new_session_name})
            if response.status_code in [200, 201]:
                st.success(f"Session '{new_session_name}' created.")
                st.session_state["session_id"] = response.json().get("id")
                st.rerun()
            else:
                st.error(f"Failed: {response.text}")
        except Exception as e:
            st.error(f"Error: {e}")

    st.subheader("Select Session")
    try:
        response = requests.get(f"{API_URL}/api/session-service/sessions")
        if response.status_code == 200:
            sessions = response.json()
            if sessions:
                opts = {s["name"]: s["id"] for s in sessions}
                sel = st.selectbox("Choose", list(opts.keys()))
                if st.button("Select"):
                    st.session_state["session_id"] = opts[sel]
                    st.success("Selected")
    except:
        pass
