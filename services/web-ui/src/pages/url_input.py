import streamlit as st
import requests
import pandas as pd
from config import API_URL, get_api_headers

def render():
    st.title("🔗 URL Input")
    if "session_id" not in st.session_state:
        st.warning("Select session first")
        return
    session_id = st.session_state["session_id"]

    tab1, tab2 = st.tabs(["Manual", "Upload"])
    with tab1:
        urls_text = st.text_area("URLs")
        if st.button("Submit"):
            urls = [u.strip() for u in urls_text.split('\n') if u.strip()]
            if urls:
                try:
                    # Raw list payload, session_id in query
                    resp = requests.post(
                        f"{API_URL}/api/url-input-service/input/urls",
                        json=urls,
                        params={"session_id": session_id}
                    )
                    if resp.status_code == 200: st.success("Added")
                    else: st.error(f"Failed: {resp.text}")
                except Exception as e: st.error(str(e))

    with tab2:
        uploaded_file = st.file_uploader("File", type=["csv", "txt", "xlsx", "json"])
        if uploaded_file and st.button("Upload"):
            # Implementation omitted for brevity but logic is similar to previous
            st.info("Upload logic here")
