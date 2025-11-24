import streamlit as st
import requests
import os
import pandas as pd
from config import API_URL

def render():
    st.title("🔗 URL Input")

    if "session_id" not in st.session_state:
        st.warning("Please select or create a session first.")
        return

    session_id = st.session_state["session_id"]
    st.info(f"Adding URLs to Session: {session_id}")

    tab1, tab2, tab3 = st.tabs(["Manual Entry", "File Upload", "Manage URLs"])

    with tab1:
        st.subheader("Enter URLs Manually")
        urls_text = st.text_area("Enter URLs (one per line)", height=200)
        if st.button("Submit URLs"):
            if urls_text:
                urls = [url.strip() for url in urls_text.split('\n') if url.strip()]
                if urls:
                    try:
                        # Reverting to raw list payload as legacy backend expects: [url1, url2]
                        # Passing session_id as query param to attempt association if backend supports it
                        response = requests.post(
                            f"{API_URL}/api/url-input-service/input/urls",
                            json=urls,
                            params={"session_id": session_id}
                        )
                        if response.status_code == 200:
                            st.success(f"Successfully submitted {len(urls)} URLs.")
                        else:
                            st.error(f"Failed to submit URLs. Status: {response.status_code}. Response: {response.text}")
                    except Exception as e:
                        st.error(f"Error submitting URLs: {e}")
                else:
                    st.warning("No valid URLs found.")
            else:
                st.warning("Please enter at least one URL.")

    with tab2:
        st.subheader("Upload URL File")
        uploaded_file = st.file_uploader("Choose a file", type=["csv", "txt", "xlsx", "json"])
        if uploaded_file is not None:
            if st.button("Upload File"):
                try:
                    files = {"file": (uploaded_file.name, uploaded_file, uploaded_file.type)}
                    data = {"session_id": session_id}

                    endpoint = f"{API_URL}/api/url-input-service/input/upload/csv"

                    if uploaded_file.name.endswith('.txt'):
                         endpoint = f"{API_URL}/api/url-input-service/input/upload/text"
                    elif uploaded_file.name.endswith('.xlsx'):
                         endpoint = f"{API_URL}/api/url-input-service/input/upload/excel"
                    elif uploaded_file.name.endswith('.json'):
                         endpoint = f"{API_URL}/api/url-input-service/input/upload/json"

                    response = requests.post(endpoint, files=files, data=data)

                    if response.status_code == 200:
                        st.success("File uploaded successfully.")
                    else:
                        st.error(f"Failed to upload file. Status: {response.status_code}. Response: {response.text}")
                except Exception as e:
                    st.error(f"Error uploading file: {e}")

    with tab3:
        st.subheader("Manage URLs")
        if st.button("Refresh URL List"):
            st.rerun()

        try:
            response = requests.get(f"{API_URL}/api/url-input-service/input/list", params={"session_id": session_id})
            if response.status_code == 200:
                urls_data = response.json()
                if urls_data:
                    df = pd.DataFrame(urls_data)
                    st.dataframe(df)
                else:
                    st.info("No URLs found for this session.")
            else:
                st.error(f"Failed to fetch URLs. Status: {response.status_code}")
        except Exception as e:
            st.error(f"Error fetching URLs: {e}")
