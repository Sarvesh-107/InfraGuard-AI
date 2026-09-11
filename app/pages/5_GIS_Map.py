"""GIS Cluster Map showing project locations color-coded by risk tier."""

import bootstrap  # noqa: F401

import folium
import pandas as pd
from folium.plugins import MarkerCluster
import streamlit as st
from streamlit_folium import st_folium

from components.filters import render_filter_bar, reset_filters_button
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


@st.cache_data(show_spinner=False)
def _build_marker_specs(project_ids: tuple, _filtered: pd.DataFrame) -> list[dict]:
    """Precomputes everything per-row (color lookups, popup/tooltip HTML strings)
    for the current filter selection. Measured ~104ms over 1,595 rows -- identical/
    redone on every rerun before this cache, even when the filter selection hadn't
    changed. cache_data (plain serializable list of dicts), not cache_resource --
    deliberately does NOT cache the folium.Map/Marker objects themselves. Caching
    those (an earlier version of this function did) made st_folium() return the
    exact same Python object -- with the same internal Leaflet element ids baked in
    at construction -- on every render; navigating away from this page and back
    then reused that stale object in a fresh iframe, and the frontend threw
    "marker_cluster_<hash> is not defined" / "Map container is already initialized"
    and rendered a blank map. Confirmed by testing with caching removed entirely:
    the error disappeared. Keyed on `project_ids` (cheap: ~1ms to build+hash a
    tuple of ids, measured) rather than `filtered` itself, since hashing the whole
    ~1,595-row, multi-column dataframe costs far more than the ~104ms this saves.
    `filtered` is still passed in (needed to build the specs) but excluded from
    hashing -- see its leading underscore.

    Note: the dominant GIS Map cost is NOT this function -- it's streamlit_folium's
    own st_folium() call, which re-renders the whole Leaflet document via Jinja2
    (~1.6-1.7s, measured) on every single script rerun regardless of what's cached
    on the Python side, since it's a live bidirectional component that must
    re-serialize on every render pass."""
    specs = []
    for _, row in _filtered.iterrows():
        tier = str(row.get("risk_tier", "Low"))
        pname = str(row.get("project_name", "Unknown Project"))
        pmin = str(row.get("ministry", "Unknown Ministry"))
        pid = str(row.get("project_id", ""))
        score = float(row.get("risk_score", 0.0))
        color_hex = RISK_COLORS.get(tier, "#28A745")

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
        specs.append({
            "lat": float(row.get("latitude", 20.5937)),
            "lon": float(row.get("longitude", 78.9629)),
            "icon_color": COLOR_MAP.get(tier, "green"),
            "popup_html": popup_html,
            "tooltip": f"{pname} ({tier} Risk)",
        })
    return specs


def _build_map(project_ids: tuple, tile_style: str, _filtered: pd.DataFrame) -> folium.Map:
    """The folium.Map/Marker objects themselves are built fresh on every call
    (cheap: constructing the objects from precomputed specs is not the expensive
    part -- see _build_marker_specs), specifically so st_folium() never receives
    the same Map object twice. Reusing one across reruns is what caused the
    remount bug described above."""
    if not _filtered.empty and "latitude" in _filtered.columns and "longitude" in _filtered.columns:
        center_lat = float(_filtered["latitude"].mean())
        center_lon = float(_filtered["longitude"].mean())
    else:
        center_lat, center_lon = 20.5937, 78.9629

    m = folium.Map(location=[center_lat, center_lon], zoom_start=5, tiles=tile_style)
    marker_cluster = MarkerCluster(
        options={"maxClusterRadius": 40, "disableClusteringAtZoom": 13}
    ).add_to(m)

    for spec in _build_marker_specs(project_ids, _filtered):
        folium.Marker(
            location=[spec["lat"], spec["lon"]],
            popup=folium.Popup(spec["popup_html"], max_width=320),
            tooltip=spec["tooltip"],
            icon=folium.Icon(color=spec["icon_color"], icon="info-sign"),
        ).add_to(marker_cluster)

    return m


with st.spinner("Loading portfolio data…"):
    projects = load_projects()

# Reuse app's existing filter bar component
filtered = render_filter_bar(projects, key_prefix="gis_map", show_search=True)
filtered = _ensure_lat_lon(filtered)

if filtered.empty:
    st.info("No projects match your filters. Try broadening ministry, sector, or date range.")
    reset_filters_button("gis_map")
    st.stop()

# Cap what the MAP actually draws (the table below still shows every filtered row --
# st.dataframe virtualizes cheaply, folium does not). Serialize cost scales
# ~linearly with marker count (~1.1ms/marker, measured: 1,595 markers ~1.75s vs 500
# ~0.56s), so an unfiltered "whole portfolio" view was serializing all 1,595 on
# every render regardless of what the caption claimed. Keeps the highest-risk
# projects when capped, since that's the actionable subset for this tool.
MAX_MAP_MARKERS = 500
map_subset = filtered
if len(filtered) > MAX_MAP_MARKERS:
    map_subset = filtered.sort_values("risk_score", ascending=False).head(MAX_MAP_MARKERS)

# GIS Legend & Stats section (Inline block) -- counts reflect the full filtered set,
# not just what's drawn on the map
crit_count = int((filtered["risk_tier"] == "Critical").sum())
high_count = int((filtered["risk_tier"] == "High").sum())
med_count = int((filtered["risk_tier"] == "Medium").sum())
low_count = int((filtered["risk_tier"] == "Low").sum())

with st.container(border=True):
    col_info, col_crit, col_high, col_med, col_low = st.columns([2.4, 2, 2, 2, 2])
    with col_info:
        st.markdown(f"**Map Portfolio Overview**")
        if len(map_subset) < len(filtered):
            st.caption(
                f"Showing {len(map_subset)} of {len(filtered)} projects on map "
                "(highest risk first) -- full list is in the table below"
            )
        else:
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

st.caption(
    "Pin locations are illustrative approximations (placed randomly within each "
    "project's state) -- the source data has no surveyed latitude/longitude, so "
    "positions are not the projects' real-world coordinates."
)

dark = is_dark_mode()
tile_style = "OpenStreetMap" if dark else "OpenStreetMap"

with st.container(border=True):
    with st.spinner("Rendering map…"):
        project_ids = tuple(map_subset["project_id"])
        m = _build_map(project_ids, tile_style, map_subset)
        st_folium(
            m,
            width="100%",
            height=620,
            returned_objects=[],
            use_container_width=True,
            # Without an explicit key, navigating away from this page and back caused
            # a blank map + console errors ("marker_cluster_<hash> is not defined",
            # "Map container is already initialized") -- Streamlit's default
            # auto-generated component key wasn't enough to force a clean remount of
            # the Leaflet iframe on revisit. A stable, explicit key fixes it.
            key="gis_map_folium",
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
