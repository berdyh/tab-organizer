"""Streamlit pages module."""

from .chatbot import render_chatbot_page
from .clustering import render_clustering_page
from .platform import render_platform_page
from .scraping import render_scraping_page
from .settings import render_settings_page
from .url_input import render_url_input_page

__all__ = [
    "render_url_input_page",
    "render_scraping_page",
    "render_clustering_page",
    "render_chatbot_page",
    "render_platform_page",
    "render_settings_page",
]
