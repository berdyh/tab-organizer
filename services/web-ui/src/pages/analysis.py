import streamlit as st
import requests
import os
from config import API_URL, get_api_headers

def render():
    st.title("🧠 Content Analysis")

    if "session_id" not in st.session_state:
        st.warning("Please select or create a session first.")
        return

    session_id = st.session_state["session_id"]

    if st.button("Start Analysis"):
        try:
            payload = {"session_id": session_id}
            # Use get_api_headers to include provider config
            headers = get_api_headers()
            response = requests.post(f"{API_URL}/api/analyzer-service/analyze", json=payload, headers=headers)
            if response.status_code == 200:
                st.success("Analysis started.")
            else:
                st.error(f"Failed to start analysis: {response.text}")
        except Exception as e:
            st.error(f"Error: {e}")

    st.subheader("Analysis Status")
    try:
        response = requests.get(f"{API_URL}/api/analyzer-service/status?session_id={session_id}")
        if response.status_code == 200:
            status = response.json()
            st.json(status)
        else:
             st.info("Status information unavailable.")
    except Exception as e:
        st.error(f"Error fetching status: {e}")

    st.subheader("Search Results")
    query = st.text_input("Search Content")
    if st.button("Search"):
        if query:
            try:
                headers = get_api_headers()
                response = requests.get(
                    f"{API_URL}/api/analyzer-service/search",
                    params={"query": query, "session_id": session_id},
                    headers=headers
                )
                if response.status_code == 200:
                    results = response.json()
                    for res in results:
                        st.markdown(f"**{res.get('title', 'No Title')}**")
                        st.caption(res.get('url', ''))
                        st.write(res.get('summary', 'No summary'))
                        st.markdown("---")
                else:
                    st.error(f"Search failed: {response.text}")
            except Exception as e:
                st.error(f"Error searching: {e}")
