"""At-risk project watchlist and inline forecast drill-down."""

from urllib.parse import quote

import bootstrap  # noqa: F401

import altair as alt
import pandas as pd
import streamlit as st

from components.filters import render_filter_bar, reset_filters_button
from components.layout import setup_page
from components.styles import glass_divider, is_dark_mode
from data_loader import month, load_projects
from risk_engine import RISK_COLORS

setup_page(
    "ews",
    title="Early Warning Dashboard",
    subtitle="Watchlist of ongoing projects sorted by predicted escalation risk",
    page_title="Early Warning | PAIMANA AI",
)

with st.spinner("Loading portfolio data…"):
    projects = load_projects()
filtered = render_filter_bar(projects, key_prefix="ews", show_search=True)

if filtered.empty:
    st.info("No projects match your filters.")
    reset_filters_button("ews")
    st.stop()

asof = projects["last_updated"].iloc[0].strftime("%Y-%m")
asof_label = projects["last_updated"].iloc[0].strftime("%B %Y")

st.caption(f"{len(filtered):,} project(s) · data as of {asof_label}")
glass_divider()

st.subheader("At-risk project watchlist")
st.caption("Tick a row to load its full forecast below. Scroll right for more columns.")

table = filtered[
    [
        "project_name",
        "sector",
        "state",
        "agency",
        "date_approval",
        "date_original",
        "date_projected_p50",
        "slip_p50_mo",
        "cost_original_cr",
        "cost_final_expected_cr",
        "cost_variance_cr",
        "cost_variance_pct",
        "physical_progress_pct",
        "risk_score",
        "risk_tier",
        "project_id",
    ]
].reset_index(drop=True)

dark = is_dark_mode()
ink = "#FAFAFA" if dark else "#31333F"
track = "#3A3D46" if dark else "#D5D7DE"


def risk_bar(score, tier):
    svg = (
        f'<!--{score:05.1f}--><svg xmlns="http://www.w3.org/2000/svg" width="120" height="24">'
        f'<rect x="2" y="9" width="86" height="6" rx="3" fill="{track}"/>'
        f'<rect x="2" y="9" width="{0.86 * float(score):.1f}" height="6" rx="3" '
        f'fill="{RISK_COLORS.get(str(tier), "#EF6C00")}"/>'
        f'<text x="117" y="16.5" text-anchor="end" font-family="sans-serif" '
        f'font-size="12" fill="{ink}">{float(score):.0f}</text></svg>'
    )
    return "data:image/svg+xml;utf8," + quote(svg)


display = table.assign(
    risk_score=[risk_bar(sc, t) for sc, t in zip(table.risk_score, table.risk_tier)]
)
ink_on = {"Low": "white", "Medium": "black", "High": "black", "Critical": "white"}
tier_cells = [
    f"background-color: {RISK_COLORS.get(str(t), '#EF6C00')}; color: {ink_on.get(str(t), 'white')}"
    for t in table.risk_tier
]
styled = display.drop(columns=["project_id"]).style.apply(lambda _: tier_cells, subset=["risk_tier"])
event = st.dataframe(
    styled,
    hide_index=True,
    on_select="rerun",
    selection_mode="single-row",
    height=400,
    key="watchlist",
    column_config={
        "project_name": st.column_config.TextColumn("Project name", pinned=True, width="large"),
        "sector": "Sector (department)",
        "state": "State/UT",
        "agency": "Implementing agency",
        "date_approval": st.column_config.DateColumn(
            "Start (approval date)",
            format="MMM YYYY",
            help="The reports give the sanction/approval date, not the actual start of works.",
        ),
        "date_original": st.column_config.DateColumn("Original end date", format="MMM YYYY"),
        "date_projected_p50": st.column_config.DateColumn("Predicted end date", format="MMM YYYY"),
        "slip_p50_mo": st.column_config.NumberColumn(
            "Slippage (months)",
            format="%.0f",
            help="Predicted end date minus original end date.",
        ),
        "cost_original_cr": st.column_config.NumberColumn("Approved cost (₹ cr)", format="%,.0f"),
        "cost_final_expected_cr": st.column_config.NumberColumn(
            "Predicted final cost (₹ cr)", format="%,.0f"
        ),
        "cost_variance_cr": st.column_config.NumberColumn("Cost escalation (₹ cr)", format="%+,.0f"),
        "cost_variance_pct": st.column_config.NumberColumn("Cost overrun %", format="%+.1f%%"),
        "physical_progress_pct": st.column_config.ProgressColumn(
            "Physical progress", min_value=0, max_value=100, format="%.0f%%", color="blue"
        ),
        "risk_score": st.column_config.ImageColumn("Risk score", width=150),
        "risk_tier": "Risk tier",
    },
)
st.download_button(
    "Download watchlist (CSV)",
    table.drop(columns=["project_id"]).to_csv(index=False),
    file_name=f"paimana_watchlist_{asof}.csv",
)

rows = [r for r in event.selection.rows if r < len(table)]
p = table.iloc[rows[0] if rows else 0]

with st.container(border=True):
    st.subheader(f"Project forecast detail — {p.project_name}")
    if not rows:
        st.caption("Showing the highest-risk project by default. Tick a row above to inspect another.")
    revised = month(p.date_revised) if "date_revised" in p.index and pd.notna(p.get("date_revised")) else "not revised"
    full = filtered[filtered["project_id"] == p.project_id].iloc[0]
    d1, d2, d3 = st.columns(3)
    d1.markdown(
        f"**Sector:** {full.sector}  \n**State/UT:** {full.state}  \n"
        f"**Implementing agency:** {full.agency}  \n"
        f"**Risk:** {full.risk_score:.0f}/100 — **{full.risk_tier}**  \n"
        f"**Physical progress:** "
        + (
            f"{full.physical_progress_pct:.0f}%"
            if pd.notna(full.physical_progress_pct)
            else "not reported"
        )
    )
    d2.markdown(
        f"**Start (approval):** {month(full.date_approval)}  \n"
        f"**Original end date:** {month(full.date_original)}  \n"
        f"**Latest revised end date:** {month(full.date_revised) if pd.notna(full.date_revised) else revised}  \n"
        f"**Agency's own estimate:** {month(full.date_agency_says)}  \n"
        f"**Predicted end date:** **{month(full.date_projected_p50)}** "
        f"(worst case {month(full.date_projected_p80)})  \n"
        f"**Predicted slippage:** {full.slip_p50_mo:.0f} months"
    )
    d3.markdown(
        f"**Approved cost:** ₹{full.cost_original_cr:,.0f} cr  \n"
        f"**Predicted final cost:** ₹{full.cost_final_expected_cr:,.0f} cr  \n"
        f"**Cost escalation:** ₹{full.cost_variance_cr:+,.0f} cr ({full.cost_variance_pct:+.1f}%)  \n"
        f"**Chance of finishing within 12 months:** {full.prob_finish_12mo_pct:.0f}%"
    )
    if full.reliable:
        st.success("End date reliable — usually within ~6 months")
    else:
        st.warning("End date is a rough estimate — can't be checked beyond 1 year")
    st.info(f"**Why this project is flagged:** {full.why}")

    start = full.date_approval if pd.notna(full.date_approval) else full.date_original
    bars = pd.DataFrame(
        [
            {"schedule": "Original schedule", "start": start, "end": full.date_original, "kind": "Original"},
            {"schedule": "Revised schedule", "start": start, "end": full.date_revised, "kind": "Revised"},
            {"schedule": "Predicted", "start": start, "end": full.date_projected_p50, "kind": "Predicted"},
            {
                "schedule": "Predicted",
                "start": full.date_projected_p50,
                "end": full.date_projected_p80,
                "kind": "Worst case",
            },
        ]
    ).dropna()
    kinds = ["Original", "Revised", "Predicted", "Worst case"]
    colors = [RISK_COLORS["Low"], RISK_COLORS["Medium"], RISK_COLORS.get(str(full.risk_tier), RISK_COLORS["High"]), "#bbbbbb"]
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
            tooltip=[
                "kind",
                alt.Tooltip("start:T", format="%b %Y"),
                alt.Tooltip("end:T", format="%b %Y"),
            ],
        )
    )
    today = (
        alt.Chart(pd.DataFrame({"d": [pd.Timestamp(asof + "-01")]}))
        .mark_rule(strokeDash=[4, 4], color="gray")
        .encode(x="d:T")
    )
    st.altair_chart(
        (gantt + today).properties(title="Original vs revised vs predicted completion", height=190),
        width="stretch",
    )
    st.caption("Dashed line = data date. Grey = predicted worst case.")

    if st.button("Open full project page", type="primary"):
        st.session_state["selected_project_id"] = str(full.project_id)
        st.switch_page("pages/2_Project_Detail.py")
