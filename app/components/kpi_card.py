"""KPI metric display helpers."""

import html

import streamlit as st


def render_kpi_row(metrics: list[tuple[str, str, str | None]]) -> None:
    """
    Render a row of glass KPI cards.

    Each metric is (label, value, delta_or_help).
    """
    cols = st.columns(len(metrics), gap="medium")
    for col, (label, value, help_text) in zip(cols, metrics):
        with col:
            hint = html.escape(help_text) if help_text else ""
            st.markdown(
                f"""
                <div class="kpi-glass">
                    <div class="kpi-label">{html.escape(label)}</div>
                    <div class="kpi-value">{html.escape(value)}</div>
                    <div class="kpi-hint">{hint}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
