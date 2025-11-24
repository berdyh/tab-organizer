import streamlit as st
import requests
import os
from config import API_URL

def render():
    st.title("📤 Export")

    if "session_id" not in st.session_state:
        st.warning("Please select or create a session first.")
        return

    session_id = st.session_state["session_id"]

    export_format = st.selectbox("Select Export Format", ["Notion", "Obsidian", "Word", "Markdown"])

    # Added options checkboxes
    col1, col2, col3 = st.columns(3)
    with col1:
        include_metadata = st.checkbox("Include Metadata", value=True)
    with col2:
        include_content = st.checkbox("Include Content", value=True)
    with col3:
        include_clusters = st.checkbox("Include Clusters", value=True)

    if st.button("Export"):
        try:
            payload = {
                "session_id": session_id,
                "format": export_format.lower(),
                "options": {
                    "include_metadata": include_metadata,
                    "include_content": include_content,
                    "include_clusters": include_clusters
                }
            }
            response = requests.post(f"{API_URL}/api/export-service/export", json=payload)

            if response.status_code == 200:
                st.success(f"Export to {export_format} successful.")
            else:
                st.error(f"Export failed: {response.text}")
        except Exception as e:
            st.error(f"Error: {e}")
