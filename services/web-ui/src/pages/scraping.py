import streamlit as st
import requests
from config import API_URL, get_api_headers

def render():
    st.title("🕷️ Scraping Status")
    if "session_id" not in st.session_state: return
    session_id = st.session_state["session_id"]

    if st.button("Start Scraping"):
        # Fetch URLs first
        try:
            r = requests.get(f"{API_URL}/api/url-input-service/input/list", params={"session_id": session_id})
            urls = [u['url'] for u in r.json()] if r.status_code == 200 else []
            if urls:
                requests.post(f"{API_URL}/api/scraper-service/scrape/batch", json={"urls": urls, "session_id": session_id}, headers=get_api_headers())
                st.success("Started")
        except: pass
