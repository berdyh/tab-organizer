import streamlit as st
import requests
from config import API_URL

def render():
    st.title("🤖 AI Chatbot")
    prompt = st.chat_input("Ask me")
    if prompt:
        st.write(prompt)
