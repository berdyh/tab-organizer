"""URL Input page for Streamlit UI."""

import streamlit as st
from ..api.client import SyncAPIClient


def render_url_input_page():
    """Render the URL input page."""
    st.header("📥 Add URLs")
    
    # Initialize API client
    if "api_client" not in st.session_state:
        st.session_state.api_client = SyncAPIClient()
    
    api = st.session_state.api_client
    
    # Session selection
    st.subheader("Session")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        try:
            sessions = api.list_sessions()
            session_options = {s["name"]: s["id"] for s in sessions}
            
            if session_options:
                selected_name = st.selectbox(
                    "Select session",
                    options=list(session_options.keys()),
                    key="session_select",
                )
                st.session_state.current_session_id = session_options[selected_name]
            else:
                st.info("No sessions yet. Create one below.")
                st.session_state.current_session_id = None
        except Exception as e:
            st.error(f"Failed to load sessions: {e}")
            st.session_state.current_session_id = None
    
    with col2:
        new_session_name = st.text_input("New session name", key="new_session_name")
        if st.button("Create Session", key="create_session"):
            if new_session_name:
                try:
                    result = api.create_session(new_session_name)
                    st.session_state.current_session_id = result["id"]
                    st.success(f"Created session: {new_session_name}")
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to create session: {e}")
    
    st.divider()
    
    # URL input methods
    st.subheader("Add URLs")
    
    tab1, tab2, tab3 = st.tabs(["📝 Paste URLs", "📄 Upload File", "🔗 Single URL"])
    
    with tab1:
        urls_text = st.text_area(
            "Paste URLs (one per line)",
            height=200,
            placeholder="https://example.com\nhttps://another-site.com/page",
            key="urls_textarea",
        )
        
        if st.button("Add URLs", key="add_urls_paste"):
            if urls_text and st.session_state.current_session_id:
                urls = [u.strip() for u in urls_text.split("\n") if u.strip()]
                if urls:
                    try:
                        result = api.add_urls(urls, st.session_state.current_session_id)
                        st.success(
                            f"Added {result['added']} URLs "
                            f"({result['duplicates']} duplicates skipped)"
                        )
                    except Exception as e:
                        st.error(f"Failed to add URLs: {e}")
                else:
                    st.warning("No valid URLs found")
            elif not st.session_state.current_session_id:
                st.warning("Please select or create a session first")
    
    with tab2:
        uploaded_file = st.file_uploader(
            "Upload a text file with URLs",
            type=["txt", "csv"],
            key="url_file_upload",
        )
        
        if uploaded_file is not None:
            content = uploaded_file.read().decode("utf-8")
            urls = [u.strip() for u in content.split("\n") if u.strip()]
            
            st.info(f"Found {len(urls)} URLs in file")
            
            if st.button("Add URLs from File", key="add_urls_file"):
                if urls and st.session_state.current_session_id:
                    try:
                        result = api.add_urls(urls, st.session_state.current_session_id)
                        st.success(
                            f"Added {result['added']} URLs "
                            f"({result['duplicates']} duplicates skipped)"
                        )
                    except Exception as e:
                        st.error(f"Failed to add URLs: {e}")
                elif not st.session_state.current_session_id:
                    st.warning("Please select or create a session first")
    
    with tab3:
        single_url = st.text_input(
            "Enter URL",
            placeholder="https://example.com",
            key="single_url_input",
        )
        
        if st.button("Add URL", key="add_single_url"):
            if single_url and st.session_state.current_session_id:
                try:
                    result = api.add_urls([single_url], st.session_state.current_session_id)
                    if result["added"] > 0:
                        st.success("URL added successfully")
                    else:
                        st.info("URL already exists in session")
                except Exception as e:
                    st.error(f"Failed to add URL: {e}")
            elif not st.session_state.current_session_id:
                st.warning("Please select or create a session first")
    
    st.divider()
    
    # Session stats
    if st.session_state.current_session_id:
        st.subheader("Session Statistics")
        
        try:
            stats = api.get_session(st.session_state.current_session_id)
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total URLs", stats.get("total_urls", 0))
            
            status_counts = stats.get("status_counts", {})
            
            with col2:
                st.metric("Pending", status_counts.get("pending", 0))
            
            with col3:
                st.metric("Scraped", status_counts.get("scraped", 0))
            
            with col4:
                st.metric("Auth Required", status_counts.get("auth_required", 0))
            
            # Show URLs
            with st.expander("View URLs"):
                urls = api.get_urls(st.session_state.current_session_id)
                for url_data in urls[:50]:  # Limit display
                    status_icon = {
                        "pending": "⏳",
                        "scraped": "✅",
                        "failed": "❌",
                        "auth_required": "🔐",
                    }.get(url_data["status"], "❓")
                    
                    st.write(f"{status_icon} {url_data['original']}")
                
                if len(urls) > 50:
                    st.info(f"Showing 50 of {len(urls)} URLs")
                    
        except Exception as e:
            st.error(f"Failed to load session stats: {e}")
