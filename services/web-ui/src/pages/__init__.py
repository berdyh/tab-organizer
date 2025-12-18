"""Streamlit pages module."""

from .url_input import render_url_input_page
from .scraping import render_scraping_page
from .clustering import render_clustering_page
from .chatbot import render_chatbot_page
from .settings import render_settings_page

__all__ = [
    "render_url_input_page",
    "render_scraping_page",
    "render_clustering_page",
    "render_chatbot_page",
    "render_settings_page",
]
