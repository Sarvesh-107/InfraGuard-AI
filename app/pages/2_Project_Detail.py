"""Single-project forecast drill-down from PAIMANA cards."""

import bootstrap  # noqa: F401

import altair as alt
import pandas as pd
import streamlit as st

from components.layout import setup_page
from components.risk_badge import render_risk_badge
from components.styles import glass_divider, page_header
from data_loader import month, load_projects
from risk_engine import RISK_COLORS

setup_page(
    "detail",
    page_title="Project Detail | PAIMANA AI",
)

project_id = st.session_state.get("selected_project_id")
if not project_id and "project_id" in st.query_params:
    project_id = st.query_params["project_id"]
    st.session_state["selected_project_id"] = project_id

projects = load_projects()
asof = projects["last_updated"].iloc[0].strftime("%Y-%m")

if not project_id:
    page_header("Project Detail", "Select a project from Overview, Early Warning, or the GIS map")
    st.info("No project selected. Open **Early Warning** or **GIS Map** and choose a project.")
    if st.button("Back to Early Warning"):
        st.switch_page("pages/1_Early_Warning.py")
    st.stop()

match = projects[projects["project_id"].astype(str) == str(project_id)]
if match.empty:
    page_header("Project Detail", "Project not found")
    st.warning(f"Project `{project_id}` was not found in the latest forecast cards.")
    if st.button("Back to Early Warning"):
        st.switch_page("pages/1_Early_Warning.py")
    st.stop()

row = match.iloc[0]
page_header(str(row["project_name"]), f"{row['sector']} · {row['state']} · {row['agency']}")

with st.container(border=True):
    meta1, meta2, meta3, meta4 = st.columns(4)
    with meta1:
        st.metric("Project ID", str(row["project_id"]))
    with meta2:
        st.metric("Risk Score", f"{row['risk_score']:.0f}")
    with meta3:
        st.markdown("**Risk Tier**")
        render_risk_badge(str(row["risk_tier"]))
    with meta4:
        prog = row["physical_progress_pct"]
        st.metric("Physical progress", f"{prog:.0f}%" if pd.notna(prog) else "—")

glass_divider()

left, right = st.columns([3, 2], gap="large")
with left:
    with st.container(border=True):
        st.subheader("Schedule & cost")
        st.markdown(
            f"**Start (approval):** {month(row.date_approval)}  \n"
            f"**Original end date:** {month(row.date_original)}  \n"
            f"**Latest revised end date:** {month(row.date_revised)}  \n"
            f"**Agency's own estimate:** {month(row.date_agency_says)}  \n"
            f"**Predicted end date:** **{month(row.date_projected_p50)}** "
            f"(worst case {month(row.date_projected_p80)})  \n"
            f"**Predicted slippage:** {row.slip_p50_mo:.0f} months"
        )
        st.markdown(
            f"**Approved cost:** ₹{row.cost_original_cr:,.0f} cr  \n"
            f"**Predicted final cost:** ₹{row.cost_final_expected_cr:,.0f} cr  \n"
            f"**Cost escalation:** ₹{row.cost_variance_cr:+,.0f} cr ({row.cost_variance_pct:+.1f}%)  \n"
            f"**Chance of finishing within 12 months:** {row.prob_finish_12mo_pct:.0f}%"
        )
        if row.reliable:
            st.success("End date reliable — usually within ~6 months")
        else:
            st.warning("End date is a rough estimate — can't be checked beyond 1 year")
        st.info(f"**Why this project is flagged:** {row.why}")

with right:
    with st.container(border=True):
        st.subheader("Original vs predicted completion")
        start = row.date_approval if pd.notna(row.date_approval) else row.date_original
        bars = pd.DataFrame(
            [
                {"schedule": "Original schedule", "start": start, "end": row.date_original, "kind": "Original"},
                {"schedule": "Revised schedule", "start": start, "end": row.date_revised, "kind": "Revised"},
                {"schedule": "Predicted", "start": start, "end": row.date_projected_p50, "kind": "Predicted"},
                {
                    "schedule": "Predicted",
                    "start": row.date_projected_p50,
                    "end": row.date_projected_p80,
                    "kind": "Worst case",
                },
            ]
        ).dropna()
        kinds = ["Original", "Revised", "Predicted", "Worst case"]
        colors = [
            RISK_COLORS["Low"],
            RISK_COLORS["Medium"],
            RISK_COLORS.get(str(row.risk_tier), RISK_COLORS["High"]),
            "#bbbbbb",
        ]
        gantt = (
            alt.Chart(bars)
            .mark_bar(height=18)
            .encode(
                x=alt.X("start:T", title=None, axis=alt.Axis(format="%Y", tickCount="year")),
                x2="end:T",
                y=alt.Y(
                    "schedule:N",
                    sort=["Original schedule", "Revised schedule", "Predicted"],
                    title=None,
                ),
                color=alt.Color(
                    "kind:N",
                    scale=alt.Scale(domain=kinds, range=colors),
                    legend=alt.Legend(orient="bottom", title=None),
                ),
            )
        )
        today = (
            alt.Chart(pd.DataFrame({"d": [pd.Timestamp(asof + "-01")]}))
            .mark_rule(strokeDash=[4, 4], color="gray")
            .encode(x="d:T")
        )
        st.altair_chart((gantt + today).properties(height=210), width="stretch")

glass_divider()
b1, b2, _ = st.columns([1.6, 1.2, 3])
with b1:
    if st.button("Ask Assistant about this project", type="primary"):
        st.session_state["assistant_prefill"] = (
            f"Summarize risk for {row.project_name} in {row.state}"
        )
        st.switch_page("pages/6_Assistant.py")
with b2:
    if st.button("Back to Early Warning"):
        st.switch_page("pages/1_Early_Warning.py")
