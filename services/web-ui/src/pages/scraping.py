"""Scraping status page for Streamlit UI."""

import time
import streamlit as st
from ..api.client import SyncAPIClient


def render_scraping_page():
    """Render the scraping status page."""
    st.header("🔄 Scraping Status")
    
    # Initialize API client
    if "api_client" not in st.session_state:
        st.session_state.api_client = SyncAPIClient()
    
    api = st.session_state.api_client
    
    # Check for current session
    if not st.session_state.get("current_session_id"):
        st.warning("Please select a session first on the URL Input page.")
        return
    
    session_id = st.session_state.current_session_id
    
    # Scraping controls
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if st.button("🚀 Start Scraping", key="start_scraping", type="primary"):
            try:
                result = api.start_scraping(session_id)
                st.success(f"Started scraping {result.get('url_count', 0)} URLs")
                st.session_state.scraping_active = True
            except Exception as e:
                st.error(f"Failed to start scraping: {e}")
    
    with col2:
        auto_refresh = st.checkbox("Auto-refresh", value=False, key="auto_refresh")
    
    st.divider()
    
    # Scraping status
    st.subheader("Progress")
    
    try:
        status = api.get_scrape_status(session_id)
        
        if status.get("status") == "not_started":
            st.info("Scraping has not been started yet.")
        else:
            # Progress bar
            total = status.get("total", 0)
            completed = status.get("completed", 0)
            
            if total > 0:
                progress = completed / total
                st.progress(progress, text=f"{completed}/{total} URLs processed")
            
            # Status metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("✅ Success", status.get("success", 0))
            
            with col2:
                st.metric("❌ Failed", status.get("failed", 0))
            
            with col3:
                st.metric("🔐 Auth Required", status.get("auth_required", 0))
            
            with col4:
                st.metric("📊 Status", status.get("status", "unknown").title())
            
            # Error display
            if status.get("error"):
                st.error(f"Error: {status['error']}")
                
    except Exception as e:
        st.error(f"Failed to get scraping status: {e}")
    
    st.divider()
    
    # Authentication queue
    st.subheader("🔐 Authentication Queue")
    
    try:
        auth_data = api.get_pending_auth()
        pending = auth_data.get("pending", [])
        
        if not pending:
            st.success("No pending authentication requests")
        else:
            st.warning(f"{len(pending)} sites require authentication")
            
            for request in pending:
                with st.expander(f"🔒 {request['domain']}", expanded=True):
                    st.write(f"**URL:** {request['url']}")
                    st.write(f"**Auth Type:** {request['auth_type']}")
                    
                    if request.get("oauth_provider"):
                        st.write(f"**OAuth Provider:** {request['oauth_provider']}")
                    
                    if request.get("form_fields"):
                        st.write(f"**Form Fields:** {', '.join(request['form_fields'])}")
                    
                    # Credential input form
                    st.write("---")
                    st.write("**Provide Credentials:**")
                    
                    auth_type = request["auth_type"]
                    domain = request["domain"]
                    
                    if auth_type in ("basic", "form"):
                        username = st.text_input(
                            "Username/Email",
                            key=f"username_{domain}",
                        )
                        password = st.text_input(
                            "Password",
                            type="password",
                            key=f"password_{domain}",
                        )
                        
                        if st.button("Submit", key=f"submit_{domain}"):
                            if username and password:
                                try:
                                    api.submit_credentials(
                                        domain,
                                        {
                                            "type": auth_type,
                                            "username": username,
                                            "password": password,
                                        },
                                    )
                                    st.success("Credentials submitted!")
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"Failed to submit: {e}")
                            else:
                                st.warning("Please enter both username and password")
                    
                    elif auth_type == "cookie":
                        cookies_json = st.text_area(
                            "Cookies (JSON format)",
                            key=f"cookies_{domain}",
                            placeholder='{"session_id": "abc123", "auth_token": "xyz"}',
                        )
                        
                        if st.button("Submit Cookies", key=f"submit_cookies_{domain}"):
                            if cookies_json:
                                try:
                                    import json
                                    cookies = json.loads(cookies_json)
                                    api.submit_credentials(
                                        domain,
                                        {"type": "cookie", "cookies": cookies},
                                    )
                                    st.success("Cookies submitted!")
                                    st.rerun()
                                except json.JSONDecodeError:
                                    st.error("Invalid JSON format")
                                except Exception as e:
                                    st.error(f"Failed to submit: {e}")
                    
                    else:
                        st.info(
                            f"Authentication type '{auth_type}' requires manual handling. "
                            "Please authenticate in your browser and provide cookies."
                        )
                        
    except Exception as e:
        st.error(f"Failed to load auth queue: {e}")
    
    # Auto-refresh
    if auto_refresh and st.session_state.get("scraping_active"):
        time.sleep(2)
        st.rerun()
