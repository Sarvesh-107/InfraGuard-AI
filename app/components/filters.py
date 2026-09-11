"""Collapsible round filter popover with vector funnel icon."""

from __future__ import annotations

from datetime import date

import pandas as pd
import streamlit as st

from data_loader import apply_filters
from risk_engine import TIERS


def render_filter_bar(
    df: pd.DataFrame,
    *,
    key_prefix: str = "filter",
    show_search: bool = False,
) -> pd.DataFrame:
    ministries = sorted(df["ministry"].dropna().unique())
    sectors = sorted(df["sector"].dropna().unique()) if "sector" in df.columns else []
    tiers = [t for t in TIERS if t in set(df["risk_tier"].astype(str).unique())] or list(TIERS)

    if "last_updated" in df.columns and df["last_updated"].notna().any():
        min_date = df["last_updated"].min().date()
        max_date = df["last_updated"].max().date()
    else:
        min_date = max_date = date.today()

    active_count = 0
    if st.session_state.get(f"{key_prefix}_ministry"):
        active_count += len(st.session_state[f"{key_prefix}_ministry"])
    if st.session_state.get(f"{key_prefix}_sector"):
        active_count += len(st.session_state[f"{key_prefix}_sector"])
    if st.session_state.get(f"{key_prefix}_tier"):
        active_count += len(st.session_state[f"{key_prefix}_tier"])
    if st.session_state.get(f"{key_prefix}_search"):
        active_count += 1

    badge_str = f" · {active_count}" if active_count > 0 else ""
    label = f"Filter ⯆{badge_str}"

    st.markdown('<div class="filter-round-anchor"></div>', unsafe_allow_html=True)
    with st.popover(label, help="Filter portfolio projects"):
        st.markdown(
            """
            <div class="filter-popover-title">Filter Criteria</div>
            """,
            unsafe_allow_html=True,
        )

        sel_ministries = st.multiselect(
            "Ministry / sector",
            ministries,
            placeholder="All ministries / sectors",
            key=f"{key_prefix}_ministry",
        )
        sel_sectors = []
        if sectors:
            sel_sectors = st.multiselect(
                "Sector",
                sectors,
                placeholder="All sectors",
                key=f"{key_prefix}_sector",
            )
        sel_tiers = st.multiselect(
            "Risk Tier",
            tiers,
            placeholder="All tiers",
            key=f"{key_prefix}_tier",
        )
        date_range = st.date_input(
            "Date Range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
            key=f"{key_prefix}_dates",
        )
        search_q = None
        if show_search:
            search_q = st.text_input(
                "Search",
                placeholder="Name, ID, sector, state…",
                key=f"{key_prefix}_search",
            )

    date_from, date_to = None, None
    if isinstance(date_range, tuple) and len(date_range) == 2:
        date_from, date_to = date_range
    elif isinstance(date_range, date):
        date_from = date_to = date_range

    return apply_filters(
        df,
        ministries=sel_ministries or None,
        sectors=sel_sectors or None,
        risk_tiers=sel_tiers or None,
        date_from=date_from,
        date_to=date_to,
        search=search_q,
    )
