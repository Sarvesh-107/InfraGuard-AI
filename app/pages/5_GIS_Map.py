"""GIS Cluster Map showing project locations color-coded by risk tier."""

import bootstrap  # noqa: F401

import folium
from folium.plugins import MarkerCluster
import streamlit as st
from streamlit_folium import st_folium

from components.filters import render_filter_bar
from components.layout import setup_page
from components.styles import is_dark_mode
from data_loader import _ensure_lat_lon, load_projects
from risk_engine import RISK_COLORS

setup_page(
    "gis_map",
    title="GIS Project Map",
    subtitle="Geospatial distribution of infrastructure projects clustered by risk tier",
    page_title="GIS Map | InfraGuard AI",
)

COLOR_MAP = {
    "Critical": "darkred",
    "High": "red",
    "Medium": "orange",
    "Low": "green",
}

projects = load_projects()

# Reuse app's existing filter bar component
filtered = render_filter_bar(projects, key_prefix="gis_map", show_search=True)
filtered = _ensure_lat_lon(filtered)

if filtered.empty:
    st.info("No projects match your filters. Try broadening ministry, sector, or date range.")
    st.stop()

# GIS Legend & Stats section (Inline block)
crit_count = int((filtered["risk_tier"] == "Critical").sum())
high_count = int((filtered["risk_tier"] == "High").sum())
med_count = int((filtered["risk_tier"] == "Medium").sum())
low_count = int((filtered["risk_tier"] == "Low").sum())

with st.container(border=True):
    col_info, col_crit, col_high, col_med, col_low = st.columns([2.4, 2, 2, 2, 2])
    with col_info:
        st.markdown(f"**Map Portfolio Overview**")
        st.caption(f"Showing {len(filtered)} project(s) on map")
    with col_crit:
        st.markdown(
            f'<span class="risk-badge risk-badge-critical">Critical</span> <strong>{crit_count}</strong>',
            unsafe_allow_html=True,
        )
    with col_high:
        st.markdown(
            f'<span class="risk-badge risk-badge-high">High Risk</span> <strong>{high_count}</strong>',
            unsafe_allow_html=True,
        )
    with col_med:
        st.markdown(
            f'<span class="risk-badge risk-badge-medium">Medium Risk</span> <strong>{med_count}</strong>',
            unsafe_allow_html=True,
        )
    with col_low:
        st.markdown(
            f'<span class="risk-badge risk-badge-low">Low Risk</span> <strong>{low_count}</strong>',
            unsafe_allow_html=True,
        )

# Calculate map center from filtered dataset
if not filtered.empty and "latitude" in filtered.columns and "longitude" in filtered.columns:
    center_lat = float(filtered["latitude"].mean())
    center_lon = float(filtered["longitude"].mean())
else:
    center_lat, center_lon = 20.5937, 78.9629

st.caption(
    "Pin locations are illustrative approximations (placed randomly within each "
    "project's state) -- the source data has no surveyed latitude/longitude, so "
    "positions are not the projects' real-world coordinates."
)

dark = is_dark_mode()
tile_style = "OpenStreetMap" if dark else "OpenStreetMap"

with st.container(border=True):
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=5,
        tiles=tile_style,
    )

    marker_cluster = MarkerCluster(
        options={
            "maxClusterRadius": 40,
            "disableClusteringAtZoom": 13,
        }
    ).add_to(m)

    for _, row in filtered.iterrows():
        lat = float(row.get("latitude", 20.5937))
        lon = float(row.get("longitude", 78.9629))
        tier = str(row.get("risk_tier", "Low"))
        icon_color = COLOR_MAP.get(tier, "green")
        color_hex = RISK_COLORS.get(tier, "#28A745")

        pname = str(row.get("project_name", "Unknown Project"))
        pmin = str(row.get("ministry", "Unknown Ministry"))
        pid = str(row.get("project_id", ""))
        score = float(row.get("risk_score", 0.0))

        # Note: no link to Project Detail here -- st_folium renders the map inside an
        # iframe, so an <a target="_self"> navigates the iframe's own document, not the
        # parent Streamlit app. Selecting a row in the table below the map is what
        # actually opens Project Detail (see st.switch_page call further down).
        popup_html = f"""
        <div style="font-family: 'Inter', sans-serif; font-size: 13px; line-height: 1.5; color: #0f172a; min-width: 200px;">
            <strong style="font-size: 14px; display: block; margin-bottom: 4px;">{pname}</strong>
            <span style="color: #475569; font-size: 12px; display: block; margin-bottom: 6px;">{pid} &middot; {pmin}</span>
            <div>
                Risk Tier: <span style="background-color: {color_hex}; color: #ffffff; padding: 2px 8px; border-radius: 6px; font-weight: 600; font-size: 11px;">{tier}</span>
                <span style="font-weight: 600; margin-left: 4px;">({score:.1f})</span>
            </div>
        </div>
        """

        folium.Marker(
            location=[lat, lon],
            popup=folium.Popup(popup_html, max_width=320),
            tooltip=f"{pname} ({tier} Risk)",
            icon=folium.Icon(color=icon_color, icon="info-sign"),
        ).add_to(marker_cluster)

    st_folium(
        m,
        width="100%",
        height=620,
        returned_objects=[],
        use_container_width=True,
    )

st.subheader("Filtered projects")
st.caption("Select a row to open its full Project Detail page.")

table = filtered[["project_id", "project_name", "sector", "state", "risk_tier"]].reset_index(drop=True)

event = st.dataframe(
    table,
    hide_index=True,
    on_select="rerun",
    selection_mode="single-row",
    height=400,
    key="gis_map_table",
    column_config={
        "project_id": "Project ID",
        "project_name": st.column_config.TextColumn("Project name", width="large"),
        "sector": "Sector (department)",
        "state": "State/UT",
        "risk_tier": "Risk tier",
    },
)

rows = event.selection.rows
if rows:
    st.session_state["selected_project_id"] = str(table.iloc[rows[0]].project_id)
    st.switch_page("pages/2_Project_Detail.py")
