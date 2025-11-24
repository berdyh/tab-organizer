import streamlit as st

def render():
    st.title("⚙️ Settings")

    st.subheader("AI Providers Configuration")

    # LLM Provider
    st.markdown("### LLM Provider")
    llm_provider = st.selectbox(
        "Select LLM Provider",
        ["Ollama (Local)", "OpenAI", "DeepSeek", "Gemini"],
        index=0,
        key="llm_provider_select"
    )

    # Update session state immediately
    st.session_state["llm_provider"] = llm_provider.lower().split(" ")[0]

    # Model Name (Optional override)
    llm_model = st.text_input(
        "Model Name (e.g., gpt-4o, deepseek-chat, gemini-pro, or Ollama model tag)",
        value=st.session_state.get("llm_model_name", ""),
        help="Leave empty to use default for the provider."
    )
    if llm_model:
        st.session_state["llm_model_name"] = llm_model

    # Embedding Provider
    st.markdown("### Embedding Provider")
    embedding_provider = st.selectbox(
        "Select Embedding Provider",
        ["Local (SentenceTransformers/Ollama)", "OpenAI", "Gemini"],
        index=0,
        key="embedding_provider_select"
    )
    # Update session state
    st.session_state["embedding_provider"] = embedding_provider.lower().split(" ")[0]

    st.markdown("---")
    st.subheader("API Keys")

    with st.expander("OpenAI API Key"):
        openai_key = st.text_input("Enter OpenAI API Key", type="password", key="openai_key_input")
        if openai_key:
            st.session_state["openai_api_key"] = openai_key

    with st.expander("DeepSeek API Key"):
        deepseek_key = st.text_input("Enter DeepSeek API Key", type="password", key="deepseek_key_input")
        if deepseek_key:
            st.session_state["deepseek_api_key"] = deepseek_key

    with st.expander("Gemini API Key"):
        gemini_key = st.text_input("Enter Gemini API Key", type="password", key="gemini_key_input")
        if gemini_key:
            st.session_state["gemini_api_key"] = gemini_key

    st.success("Settings saved to session state.")

    # Display current configuration summary
    st.markdown("### Current Configuration")
    st.write(f"**LLM Provider:** {st.session_state.get('llm_provider', 'ollama')}")
    st.write(f"**LLM Model:** {st.session_state.get('llm_model_name', 'Default')}")
    st.write(f"**Embedding Provider:** {st.session_state.get('embedding_provider', 'local')}")

    has_openai = bool(st.session_state.get("openai_api_key"))
    has_deepseek = bool(st.session_state.get("deepseek_api_key"))
    has_gemini = bool(st.session_state.get("gemini_api_key"))

    st.write(f"**Keys Set:** OpenAI ({has_openai}), DeepSeek ({has_deepseek}), Gemini ({has_gemini})")
