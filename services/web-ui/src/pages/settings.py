"""Settings page for Streamlit UI."""

import streamlit as st

from ..api.client import SyncAPIClient


API_KEY_FIELDS = [
    ("OpenRouter", "OPENROUTER_API_KEY"),
    ("OpenAI", "OPENAI_API_KEY"),
    ("Anthropic", "ANTHROPIC_API_KEY"),
    ("DeepSeek", "DEEPSEEK_API_KEY"),
    ("Google Gemini", "GOOGLE_API_KEY"),
]


def _provider_options(
    available: dict,
    fallback: list[str],
    current_provider: str,
    capability: str,
) -> list[str]:
    """Return selectable providers without silently switching unknown values."""
    capability_map = available.get(capability, {}) if isinstance(available, dict) else {}
    if capability_map:
        options = list(capability_map)
        for provider in fallback:
            if provider in capability_map and provider not in options:
                options.append(provider)
    else:
        options = list(fallback)

    if (
        current_provider
        and current_provider != "unknown"
        and current_provider not in options
    ):
        options.insert(0, current_provider)

    return options or ([current_provider] if current_provider else fallback)


def _provider_index(options: list[str], current_provider: str) -> int:
    if current_provider in options:
        return options.index(current_provider)
    return 0


def _provider_label(available: dict, capability: str, provider: str) -> str:
    capability_map = available.get(capability, {}) if isinstance(available, dict) else {}
    info = capability_map.get(provider, {})
    if info.get("available", True):
        return provider
    return f"{provider} (unavailable)"


def _availability_reason(available: dict, capability: str, provider: str) -> str:
    capability_map = available.get(capability, {}) if isinstance(available, dict) else {}
    info = capability_map.get(provider, {})
    return info.get("reason") or ""


def _model_options(
    models: dict,
    provider: str,
    current_model: str,
    capability: str,
) -> list[str]:
    capability_models = models.get(capability, {}) if isinstance(models, dict) else {}
    options = list(capability_models.get(provider, []))
    if current_model and current_model not in options:
        options.insert(0, current_model)
    return options or ([current_model] if current_model else [])


def _model_index(options: list[str], current_model: str) -> int:
    if current_model in options:
        return options.index(current_model)
    return 0


def _api_key_configured(available: dict, api_key_env: str) -> bool:
    if not isinstance(available, dict):
        return False
    for capability in ("llm", "embeddings"):
        for info in available.get(capability, {}).values():
            if info.get("api_key_env") == api_key_env:
                return bool(info.get("api_key_configured"))
    return False


def render_settings_page():
    """Render the settings page."""
    st.header("Settings")

    # Initialize API client
    if "api_client" not in st.session_state:
        st.session_state.api_client = SyncAPIClient()

    api = st.session_state.api_client

    # AI Provider settings
    st.subheader("AI Provider Configuration")

    try:
        providers = api.get_providers()

        current_llm = providers.get("llm", {})
        current_llm_provider = current_llm.get("provider", "unknown")
        current_llm_model = current_llm.get("model", "")
        current_emb = providers.get("embeddings", {})
        current_emb_provider = current_emb.get("provider", "unknown")
        current_emb_model = current_emb.get("model", "")
        available = providers.get("available", {})
        models = providers.get("models", {})

        col1, col2 = st.columns(2)

        with col1:
            st.write("**LLM Provider**")
            st.info(f"Current: {current_llm_provider} / {current_llm_model}")

            llm_fallback = [
                "openrouter",
                "ollama",
                "openai",
                "claude_code",
                "codex_cli",
                "codex_acp",
                "anthropic",
                "deepseek",
                "gemini",
            ]
            llm_options = _provider_options(
                available, llm_fallback, current_llm_provider, "llm"
            )
            new_llm = st.selectbox(
                "Select LLM Provider",
                options=llm_options,
                index=_provider_index(llm_options, current_llm_provider),
                key="llm_provider_select",
                format_func=lambda provider: _provider_label(
                    available, "llm", provider
                ),
            )
            llm_reason = _availability_reason(available, "llm", new_llm)
            if llm_reason:
                st.caption(llm_reason)

            llm_current_model = (
                current_llm_model if new_llm == current_llm_provider else ""
            )
            llm_model_options = _model_options(models, new_llm, llm_current_model, "llm")
            new_llm_model = st.selectbox(
                "Select LLM Model",
                options=llm_model_options,
                index=_model_index(llm_model_options, llm_current_model),
                key="llm_model_select",
            )

        with col2:
            st.write("**Embedding Provider**")
            st.info(f"Current: {current_emb_provider} / {current_emb_model}")

            emb_options = ["openrouter", "ollama", "openai", "gemini"]
            emb_options = _provider_options(
                available, emb_options, current_emb_provider, "embeddings"
            )
            new_emb = st.selectbox(
                "Select Embedding Provider",
                options=emb_options,
                index=_provider_index(emb_options, current_emb_provider),
                key="emb_provider_select",
                format_func=lambda provider: _provider_label(
                    available, "embeddings", provider
                ),
            )
            emb_reason = _availability_reason(available, "embeddings", new_emb)
            if emb_reason:
                st.caption(emb_reason)

            emb_model_options = _model_options(
                models,
                new_emb,
                current_emb_model if new_emb == current_emb_provider else "",
                "embeddings",
            )
            emb_current_model = (
                current_emb_model if new_emb == current_emb_provider else ""
            )
            new_emb_model = st.selectbox(
                "Select Embedding Model",
                options=emb_model_options,
                index=_model_index(emb_model_options, emb_current_model),
                key="emb_model_select",
            )

        st.write("**API Keys**")
        st.caption("Leave a key field blank to keep the backend's current value.")
        api_keys = {}
        key_cols = st.columns(2)
        for index, (label, env_name) in enumerate(API_KEY_FIELDS):
            configured = _api_key_configured(available, env_name)
            with key_cols[index % 2]:
                entered = st.text_input(
                    f"{label} API Key",
                    value="",
                    type="password",
                    placeholder="Configured" if configured else env_name,
                    key=f"runtime_key_{env_name}",
                )
                if entered.strip():
                    api_keys[env_name] = entered.strip()

        submitted = st.button("Apply AI Configuration", key="apply_ai_config")

        if submitted:
            try:
                api.update_ai_config(
                    llm_provider=(
                        new_llm if new_llm != current_llm_provider else None
                    ),
                    llm_model=(
                        new_llm_model
                        if new_llm_model and new_llm_model != current_llm_model
                        else None
                    ),
                    embedding_provider=(
                        new_emb if new_emb != current_emb_provider else None
                    ),
                    embedding_model=(
                        new_emb_model
                        if new_emb_model and new_emb_model != current_emb_model
                        else None
                    ),
                    api_keys=api_keys,
                )
                st.success("AI configuration updated.")
                st.rerun()
            except Exception as e:
                st.error(f"Failed to update AI configuration: {e}")

    except Exception as e:
        st.error(f"Failed to load provider info: {e}")

    st.divider()

    # API Keys
    with st.expander("Environment Variables Reference"):
        st.code("""
# Subscription-backed CLI providers (LLM only)
# These use local CLI login state, not API keys. Embeddings still need
# ollama/openrouter/openai/gemini.
AI_PROVIDER=claude_code
LLM_MODEL=
CLAUDE_CODE_COMMAND=claude
CLAUDE_CODE_TIMEOUT=300

AI_PROVIDER=codex_cli
LLM_MODEL=
CODEX_CLI_COMMAND=codex
CODEX_CLI_TIMEOUT=300

AI_PROVIDER=codex_acp
LLM_MODEL=
CODEX_ACP_COMMAND=acpx
CODEX_ACP_TIMEOUT=300
CODEX_ACP_PERMISSION_MODE=deny-all

# OpenRouter
OPENROUTER_API_KEY=sk-or-...

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

# Host-run AI engine for subscription CLIs
AI_ENGINE_URL=http://host.docker.internal:8090
AI_ENGINE_API_TOKEN=<generated by scripts/cli.py host-ai>
        """)

    st.divider()

    # Export settings
    st.subheader("Export")

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
                        label="Download Export",
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
    st.subheader("Service Health")

    if st.button("Check Health", key="check_health"):
        try:
            health = api.check_health()

            col1, col2, col3 = st.columns(3)

            with col1:
                status = "Healthy" if health.get("backend") else "Unhealthy"
                st.metric("Backend Core", status)

            with col2:
                ai_detail = health.get("ai_engine_detail") or {}
                if health.get("ai_engine"):
                    status = "Healthy"
                elif ai_detail.get("status") == "degraded":
                    status = "Degraded"
                else:
                    status = "Unhealthy"
                st.metric("AI Engine", status)
                runtime = ai_detail.get("runtime", {})
                if runtime and not runtime.get("ready", True):
                    llm_reason = runtime.get("llm", {}).get("reason")
                    emb_reason = runtime.get("embeddings", {}).get("reason")
                    if llm_reason:
                        st.caption(f"LLM: {llm_reason}")
                    if emb_reason:
                        st.caption(f"Embeddings: {emb_reason}")

            with col3:
                status = "Healthy" if health.get("browser_engine") else "Unhealthy"
                st.metric("Browser Engine", status)

        except Exception as e:
            st.error(f"Health check failed: {e}")

    st.divider()

    # Session Management
    st.subheader("Session Management")

    try:
        sessions = api.list_sessions()

        if sessions:
            st.write(f"**{len(sessions)} sessions found**")

            for session in sessions:
                col1, col2, col3 = st.columns([3, 1, 1])

                with col1:
                    st.write(session["name"])
                    st.caption(
                        f"ID: {session['id'][:8]}... | URLs: {session.get('total_urls', 0)}"
                    )

                with col2:
                    if st.button("Select", key=f"select_{session['id']}"):
                        st.session_state.current_session_id = session["id"]
                        st.success(f"Selected: {session['name']}")

                with col3:
                    if st.button("Delete", key=f"delete_{session['id']}"):
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
