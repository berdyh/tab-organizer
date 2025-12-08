import streamlit as st
from config import API_URL

def render():
    st.title("⚙️ Settings")
    st.subheader("AI Providers Configuration")
    llm_provider = st.selectbox("Select LLM Provider", ["Ollama", "OpenAI", "DeepSeek", "Gemini"], key="llm_prov")
    st.session_state["llm_provider"] = llm_provider.lower()
    emb_provider = st.selectbox("Select Embedding Provider", ["Local", "OpenAI", "Gemini"], key="emb_prov")
    st.session_state["embedding_provider"] = emb_provider.lower()
    with st.expander("API Keys"):
        st.session_state["openai_api_key"] = st.text_input("OpenAI Key", type="password")
        st.session_state["deepseek_api_key"] = st.text_input("DeepSeek Key", type="password")
        st.session_state["gemini_api_key"] = st.text_input("Gemini Key", type="password")
