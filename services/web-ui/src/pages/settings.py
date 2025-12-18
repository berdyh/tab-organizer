"""Settings page for Streamlit UI."""

import streamlit as st
from ..api.client import SyncAPIClient


def render_settings_page():
    """Render the settings page."""
    st.header("⚙️ Settings")
    
    # Initialize API client
    if "api_client" not in st.session_state:
        st.session_state.api_client = SyncAPIClient()
    
    api = st.session_state.api_client
    
    # AI Provider settings
    st.subheader("🤖 AI Provider Configuration")
    
    try:
        providers = api.get_providers()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**LLM Provider**")
            current_llm = providers.get("llm", {})
            st.info(f"Current: {current_llm.get('provider', 'unknown')} / {current_llm.get('model', 'unknown')}")
            
            llm_options = ["ollama", "openai", "anthropic", "deepseek", "gemini"]
            new_llm = st.selectbox(
                "Select LLM Provider",
                options=llm_options,
                index=llm_options.index(current_llm.get("provider", "ollama")) if current_llm.get("provider") in llm_options else 0,
                key="llm_provider_select",
            )
        
        with col2:
            st.write("**Embedding Provider**")
            current_emb = providers.get("embeddings", {})
            st.info(f"Current: {current_emb.get('provider', 'unknown')} / {current_emb.get('model', 'unknown')}")
            
            emb_options = ["ollama", "openai", "deepseek", "gemini"]
            new_emb = st.selectbox(
                "Select Embedding Provider",
                options=emb_options,
                index=emb_options.index(current_emb.get("provider", "ollama")) if current_emb.get("provider") in emb_options else 0,
                key="emb_provider_select",
            )
        
        if st.button("Apply Provider Changes", key="apply_providers"):
            try:
                result = api.switch_provider(
                    llm_provider=new_llm if new_llm != current_llm.get("provider") else None,
                    embedding_provider=new_emb if new_emb != current_emb.get("provider") else None,
                )
                st.success("Provider configuration updated!")
                st.rerun()
            except Exception as e:
                st.error(f"Failed to update providers: {e}")
                
    except Exception as e:
        st.error(f"Failed to load provider info: {e}")
    
    st.divider()
    
    # API Keys
    st.subheader("🔑 API Keys")
    st.caption("API keys are configured via environment variables in Docker.")
    
    with st.expander("Environment Variables Reference"):
        st.code("""
# OpenAI
OPENAI_API_KEY=sk-...

# Anthropic
ANTHROPIC_API_KEY=sk-ant-...

# DeepSeek
DEEPSEEK_API_KEY=...

# Google Gemini
GOOGLE_API_KEY=...

# Ollama (local)
OLLAMA_HOST=http://ollama:11434
        """)
    
    st.divider()
    
    # Export settings
    st.subheader("📤 Export")
    
    session_id = st.session_state.get("current_session_id")
    
    if session_id:
        col1, col2 = st.columns(2)
        
        with col1:
            export_format = st.selectbox(
                "Export Format",
                options=["markdown", "json", "html", "obsidian"],
                key="export_format",
            )
        
        with col2:
            if st.button("Export Session", key="export_session"):
                try:
                    result = api.export_session(session_id, export_format)
                    
                    content = result.get("content", "")
                    filename = result.get("filename", f"export.{export_format}")
                    
                    st.download_button(
                        label="📥 Download Export",
                        data=content,
                        file_name=filename,
                        mime="text/plain",
                    )
                except Exception as e:
                    st.error(f"Export failed: {e}")
    else:
        st.info("Select a session to enable export.")
    
    st.divider()
    
    # Service Health
    st.subheader("🏥 Service Health")
    
    if st.button("Check Health", key="check_health"):
        try:
            health = api.check_health()
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                status = "✅ Healthy" if health.get("backend") else "❌ Unhealthy"
                st.metric("Backend Core", status)
            
            with col2:
                status = "✅ Healthy" if health.get("ai_engine") else "❌ Unhealthy"
                st.metric("AI Engine", status)
            
            with col3:
                status = "✅ Healthy" if health.get("browser_engine") else "❌ Unhealthy"
                st.metric("Browser Engine", status)
                
        except Exception as e:
            st.error(f"Health check failed: {e}")
    
    st.divider()
    
    # Session Management
    st.subheader("📁 Session Management")
    
    try:
        sessions = api.list_sessions()
        
        if sessions:
            st.write(f"**{len(sessions)} sessions found**")
            
            for session in sessions:
                col1, col2, col3 = st.columns([3, 1, 1])
                
                with col1:
                    st.write(f"📂 {session['name']}")
                    st.caption(f"ID: {session['id'][:8]}... | URLs: {session.get('total_urls', 0)}")
                
                with col2:
                    if st.button("Select", key=f"select_{session['id']}"):
                        st.session_state.current_session_id = session["id"]
                        st.success(f"Selected: {session['name']}")
                
                with col3:
                    if st.button("🗑️", key=f"delete_{session['id']}"):
                        try:
                            api.delete_session(session["id"])
                            st.success("Session deleted")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Delete failed: {e}")
        else:
            st.info("No sessions found.")
            
    except Exception as e:
        st.error(f"Failed to load sessions: {e}")
