"""Sector and state benchmarking from live forecast cards."""

import bootstrap  # noqa: F401

import altair as alt
import streamlit as st

from components.filters import render_filter_bar
from components.layout import setup_page
from data_loader import load_projects
from risk_engine import RISK_COLORS

setup_page(
    "bench",
    title="Benchmarking",
    subtitle="Predicted cost overrun and slippage by sector, and state-wise project scale",
    page_title="Benchmarking | PAIMANA AI",
)

projects = load_projects()
view = render_filter_bar(projects, key_prefix="bench", show_search=True)

if view.empty:
    st.info("No projects match your filters.")
    st.stop()

yax = alt.Axis(labelLimit=340)

c1, c2 = st.columns(2, gap="large")
by_sec = (
    view.groupby("sector", observed=True)
    .agg(
        approved=("cost_original_cr", "sum"),
        final=("cost_final_expected_cr", "sum"),
        slip=("slip_p50_mo", "median"),
        projects=("project_name", "size"),
    )
    .reset_index()
)
by_sec["overrun"] = by_sec.final / by_sec.approved - 1

with c1:
    with st.container(border=True):
        st.subheader("Sector-wise predicted cost overrun")
        st.altair_chart(
            alt.Chart(by_sec)
            .mark_bar(color=RISK_COLORS["Critical"])
            .encode(
                x=alt.X("overrun:Q", title="Predicted final cost vs approved", axis=alt.Axis(format="+%")),
                y=alt.Y("sector:N", sort="-x", title=None, axis=yax),
                tooltip=["sector", "projects", alt.Tooltip("overrun:Q", format="+.1%")],
            )
            .properties(height=420),
            width="stretch",
        )
with c2:
    with st.container(border=True):
        st.subheader("Sector-wise typical slippage")
        st.altair_chart(
            alt.Chart(by_sec)
            .mark_bar(color=RISK_COLORS["High"])
            .encode(
                x=alt.X("slip:Q", title="Median predicted slippage (months)"),
                y=alt.Y("sector:N", sort="-x", title=None, axis=yax),
                tooltip=["sector", "projects", alt.Tooltip("slip:Q", format=".0f")],
            )
            .properties(height=420),
            width="stretch",
        )

with st.container(border=True):
    st.subheader("State-wise project count vs cost")
    by_st = (
        view.groupby("state", observed=True)
        .agg(
            projects=("project_name", "size"),
            approved=("cost_original_cr", "sum"),
            escalation=("cost_variance_cr", "sum"),
            critical=("risk_tier", lambda s: (s.astype(str) == "Critical").sum()),
        )
        .reset_index()
    )
    st.altair_chart(
        alt.Chart(by_st)
        .mark_circle(opacity=0.7, color=RISK_COLORS["High"])
        .encode(
            x=alt.X("projects:Q", title="Projects"),
            y=alt.Y("approved:Q", title="Approved cost (₹ cr)"),
            size=alt.Size("critical:Q", title="Critical projects"),
            tooltip=[
                "state",
                "projects",
                alt.Tooltip("approved:Q", format=",.0f"),
                alt.Tooltip("escalation:Q", format="+,.0f", title="escalation (₹ cr)"),
                "critical",
            ],
        )
        .properties(height=380),
        width="stretch",
    )
