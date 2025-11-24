import streamlit as st
import requests
import os
from config import API_URL, get_api_headers

def render():
    st.title("🤖 AI Chatbot")

    session_id = st.session_state.get("session_id")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask something about your data..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""

            try:
                context = st.session_state.messages[-5:] if len(st.session_state.messages) > 1 else []

                payload = {
                    "message": prompt,
                    "context": context,
                    "session_id": session_id
                }

                # Include headers with API keys/provider info
                headers = get_api_headers()

                response = requests.post(
                    f"{API_URL}/api/chatbot-service/chat/message",
                    json=payload,
                    headers=headers
                )

                if response.status_code == 200:
                    data = response.json()
                    full_response = data.get("response", "I didn't get a response.")
                else:
                    full_response = f"Error: {response.status_code} - {response.text}"

            except Exception as e:
                full_response = f"Error communicating with chatbot: {e}"

            message_placeholder.markdown(full_response)

        st.session_state.messages.append({"role": "assistant", "content": full_response})
