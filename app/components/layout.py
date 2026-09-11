"""Single entry point for page setup — theme, navbar, CSS, header."""

from __future__ import annotations

import streamlit as st

from components.navbar import render_navbar
from components.styles import inject_custom_css, page_header


def setup_page(
    current_page: str,
    *,
    title: str | None = None,
    subtitle: str | None = None,
    page_title: str = "InfraGuard AI",
    page_icon: str | None = None,
) -> None:
    """Configure page, render top navbar + theme CSS, optional header."""
    st.set_page_config(
        page_title=page_title,
        page_icon=page_icon,
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    inject_custom_css()
    render_navbar(current_page)
    if title and subtitle:
        page_header(title, subtitle)

