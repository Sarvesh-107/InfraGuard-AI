"""Risk tier badge rendering."""

import streamlit as st

from risk_engine import RISK_COLORS


def risk_badge_html(tier: str) -> str:
    css_class = f"risk-badge-{tier.lower()}"
    return f'<span class="risk-badge {css_class}">{tier}</span>'


def render_risk_badge(tier: str) -> None:
    st.markdown(risk_badge_html(tier), unsafe_allow_html=True)


def tier_color(tier: str) -> str:
    return RISK_COLORS.get(tier, "#6c757d")
