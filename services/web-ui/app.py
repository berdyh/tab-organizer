"""Tab Organizer - Streamlit Web UI."""

import streamlit as st

st.set_page_config(
    page_title="Tab Organizer",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown(
    """
<style>
    .stApp {
        max-width: 1400px;
        margin: 0 auto;
    }
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
</style>
""",
    unsafe_allow_html=True,
)

# Sidebar navigation
st.sidebar.title("Tab Organizer")
st.sidebar.divider()

page = st.sidebar.radio(
    "Navigation",
    options=[
        "URL Input",
        "Scraping",
        "Clusters",
        "Chatbot",
        "Platform",
        "Settings",
    ],
    label_visibility="collapsed",
)

st.sidebar.divider()

# Session info in sidebar
if st.session_state.get("current_session_id"):
    st.sidebar.success(
        f"Session: {st.session_state.get('current_session_id', '')[:8]}..."
    )
else:
    st.sidebar.info("No session selected")

# Import and render pages
from src.pages import (
    render_chatbot_page,
    render_clustering_page,
    render_platform_page,
    render_scraping_page,
    render_settings_page,
    render_url_input_page,
)

if page == "URL Input":
    render_url_input_page()
elif page == "Scraping":
    render_scraping_page()
elif page == "Clusters":
    render_clustering_page()
elif page == "Chatbot":
    render_chatbot_page()
elif page == "Platform":
    render_platform_page()
elif page == "Settings":
    render_settings_page()

# Footer
st.sidebar.divider()
st.sidebar.caption("Tab Organizer v1.0.0")
st.sidebar.caption("Local-first tab organization with AI")
