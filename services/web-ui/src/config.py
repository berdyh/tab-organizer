import os
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8080")

def get_api_headers():
    headers = {"Content-Type": "application/json"}
    if "openai_api_key" in st.session_state and st.session_state["openai_api_key"]:
        headers["X-OpenAI-Key"] = st.session_state["openai_api_key"]
    if "deepseek_api_key" in st.session_state and st.session_state["deepseek_api_key"]:
        headers["X-DeepSeek-Key"] = st.session_state["deepseek_api_key"]
    if "gemini_api_key" in st.session_state and st.session_state["gemini_api_key"]:
        headers["X-Gemini-Key"] = st.session_state["gemini_api_key"]
    if "llm_provider" in st.session_state:
        headers["X-LLM-Provider"] = st.session_state["llm_provider"]
    if "embedding_provider" in st.session_state:
        headers["X-Embedding-Provider"] = st.session_state["embedding_provider"]
    if "llm_model_name" in st.session_state:
        headers["X-LLM-Model"] = st.session_state["llm_model_name"]
    return headers
