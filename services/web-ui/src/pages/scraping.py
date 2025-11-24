import streamlit as st
import requests
import os
import pandas as pd
from config import API_URL, get_api_headers

def render():
    st.title("🕷️ Scraping Status")

    if "session_id" not in st.session_state:
        st.warning("Please select or create a session first.")
        return

    session_id = st.session_state["session_id"]

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Start Scraping"):
            try:
                # 1. Fetch URLs for the session
                # Assuming getting list via GET /api/url-input-service/input/list
                urls_response = requests.get(
                    f"{API_URL}/api/url-input-service/input/list",
                    params={"session_id": session_id}
                )

                urls = []
                if urls_response.status_code == 200:
                    urls_data = urls_response.json()
                    # Extract URLs from the response (assuming list of dicts with 'url' key or similar)
                    # Legacy code used map(u => u.url)
                    if isinstance(urls_data, list):
                         urls = [u.get("url") for u in urls_data if "url" in u]
                    else:
                        st.warning("Unexpected URL list format.")

                if not urls:
                    st.warning("No URLs found to scrape. Please add URLs first.")
                else:
                    # 2. Send URLs to scraper
                    # Legacy likely sent { "urls": [...], "session_id": ... } to /scrape/batch
                    payload = {"urls": urls, "session_id": session_id}
                    headers = get_api_headers()
                    response = requests.post(
                        f"{API_URL}/api/scraper-service/scrape/batch",
                        json=payload,
                        headers=headers
                    )
                    if response.status_code == 200:
                        st.success(f"Scraping started for {len(urls)} URLs.")
                    else:
                        st.error(f"Failed to start scraping: {response.text}")
            except Exception as e:
                st.error(f"Error: {e}")

    with col2:
         if st.button("Stop Scraping"):
             try:
                 requests.post(f"{API_URL}/api/scraper-service/stop", json={"session_id": session_id})
                 st.success("Stop command sent.")
             except Exception as e:
                 st.error(f"Error stopping: {e}")

    st.subheader("Jobs")
    if st.button("Refresh Jobs"):
        st.rerun()

    try:
        response = requests.get(f"{API_URL}/api/scraper-service/scrape/jobs")
        if response.status_code == 200:
            jobs = response.json()
            if jobs:
                df = pd.DataFrame(jobs)
                st.dataframe(df)
            else:
                st.info("No scraping jobs found.")
        else:
            st.error(f"Failed to fetch jobs. Status: {response.status_code}")
    except Exception as e:
        st.error(f"Error fetching jobs: {e}")
