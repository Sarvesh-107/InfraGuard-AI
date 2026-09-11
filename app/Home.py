"""PAIMANA AI — Portfolio Overview (entry point)."""

import bootstrap  # noqa: F401

import altair as alt
import streamlit as st

from components.filters import render_filter_bar
from components.kpi_card import render_kpi_row
from components.layout import setup_page
from components.styles import glass_divider, page_header
from data_loader import lakh_cr, load_projects
from risk_engine import RISK_COLORS, TIERS

setup_page(
    "home",
    page_title="PAIMANA AI | InfraGuard",
)

projects = load_projects()
asof_label = projects["last_updated"].iloc[0].strftime("%B %Y")

title_col, filter_col = st.columns([4.2, 1.15], gap="large")
with title_col:
    page_header(
        "Portfolio Overview",
        f"Predictive analytics & early warning for MoSPI IPMD/DIID — data as of {asof_label}",
    )
with filter_col:
    st.markdown("<div style='height:0.55rem'></div>", unsafe_allow_html=True)
    filtered = render_filter_bar(projects, key_prefix="home", show_search=True)

if filtered.empty:
    st.info("No projects match your filters. Try broadening sector, ministry, or risk tier.")
    st.stop()

approved = float(filtered["cost_original_cr"].sum())
final = float(filtered["cost_final_expected_cr"].sum())
critical = int((filtered["risk_tier"] == "Critical").sum())
overrun_pct = f"{100 * (final / approved - 1):+.1f}%" if approved else "—"
slip = f"{filtered['slip_p50_mo'].median():.0f} mo" if len(filtered) else "—"

render_kpi_row(
    [
        ("Ongoing projects", f"{len(filtered):,}", "In the current filter"),
        ("Approved cost", lakh_cr(approved), "Originally sanctioned"),
        ("Critical-risk", f"{critical:,}", "Highest escalation risk"),
        ("Predicted final cost", lakh_cr(final), "Expected vs approved"),
        ("Cost overrun", overrun_pct, "Portfolio predicted escalation"),
        ("Typical slippage", slip, "Median predicted delay"),
    ]
)

glass_divider()

tier_scale = alt.Scale(domain=TIERS, range=[RISK_COLORS[t] for t in TIERS])
yax = alt.Axis(labelLimit=340)

col_left, col_right = st.columns(2, gap="large")
with col_left:
    with st.container(border=True):
        st.subheader("Risk tier distribution")
        st.caption("Count of ongoing projects in each predicted risk tier")
        tc = (
            filtered["risk_tier"]
            .value_counts()
            .reindex(TIERS)
            .fillna(0)
            .rename_axis("tier")
            .reset_index(name="projects")
        )
        st.altair_chart(
            alt.Chart(tc)
            .mark_bar()
            .encode(
                x=alt.X("tier:N", sort=["Low", "Medium", "High", "Critical"], title=None),
                y=alt.Y("projects:Q", title="Projects"),
                color=alt.Color("tier:N", scale=tier_scale, legend=None),
                tooltip=["tier", "projects"],
            )
            .properties(height=280),
            width="stretch",
        )

with col_right:
    with st.container(border=True):
        st.subheader("Sector-wise average risk score")
        st.caption("Sectors with at least 5 projects")
        sr = filtered.groupby("sector", observed=True)["risk_score"].agg(["mean", "size"]).reset_index()
        sr = sr[sr["size"] >= 5].nlargest(12, "mean")
        if sr.empty:
            st.info("Not enough projects per sector to chart.")
        else:
            st.altair_chart(
                alt.Chart(sr)
                .mark_bar(color=RISK_COLORS["High"])
                .encode(
                    x=alt.X("mean:Q", title="Average risk score", scale=alt.Scale(domain=[0, 100])),
                    y=alt.Y("sector:N", sort="-x", title=None, axis=yax),
                    tooltip=[
                        "sector",
                        alt.Tooltip("mean:Q", format=".0f", title="avg risk"),
                        alt.Tooltip("size:Q", title="projects"),
                    ],
                )
                .properties(height=280),
                width="stretch",
            )

glass_divider()

list_head, list_action = st.columns([4, 1.2])
with list_head:
    st.subheader("Highest-risk projects")
    st.caption("Open Early Warning for the full watchlist and forecast drill-down")
with list_action:
    if st.button("Open Early Warning", use_container_width=True):
        st.switch_page("pages/1_Early_Warning.py")

top = filtered.sort_values("risk_score", ascending=False).head(8)
for _, row in top.iterrows():
    c1, c2, c3, c4 = st.columns([3.2, 1.4, 1.1, 1.1])
    with c1:
        st.markdown(f"**{row['project_name']}**")
        st.caption(f"{row['project_id']} · {row['state']}")
    with c2:
        st.caption(str(row["sector"]))
    with c3:
        st.markdown(
            f'<span class="risk-badge risk-badge-{str(row["risk_tier"]).lower()}">{row["risk_tier"]}</span>',
            unsafe_allow_html=True,
        )
    with c4:
        if st.button("Details", key=f"home_{row['project_id']}", use_container_width=True):
            st.session_state["selected_project_id"] = str(row["project_id"])
            st.switch_page("pages/2_Project_Detail.py")
