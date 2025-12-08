import streamlit as st
import requests
from config import API_URL, get_api_headers

def render():
    st.title("🤖 AI Chatbot")
    session_id = st.session_state.get("session_id")

    if "messages" not in st.session_state: st.session_state.messages = []

    for m in st.session_state.messages:
        with st.chat_message(m["role"]): st.markdown(m["content"])

    if p := st.chat_input("Ask"):
        st.session_state.messages.append({"role": "user", "content": p})
        with st.chat_message("user"): st.markdown(p)

        with st.chat_message("assistant"):
            try:
                r = requests.post(
                    f"{API_URL}/api/chatbot-service/chat/message",
                    json={"message": p, "session_id": session_id, "context": st.session_state.messages[-5:]},
                    headers=get_api_headers()
                )
                resp = r.json().get("response", "Error")
            except: resp = "Error"
            st.markdown(resp)
            st.session_state.messages.append({"role": "assistant", "content": resp})
