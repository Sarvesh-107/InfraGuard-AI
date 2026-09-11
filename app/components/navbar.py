"""Single rectangular navy blue top navigation bar with white font and spacers."""

from __future__ import annotations

import streamlit as st

NAV_ITEMS = [
    {"id": "home", "label": "Overview", "page": "Home.py"},
    {"id": "ews", "label": "Early Warning", "page": "pages/1_Early_Warning.py"},
    {"id": "detail", "label": "Project Detail", "page": "pages/2_Project_Detail.py"},
    {"id": "models", "label": "Models", "page": "pages/3_Models.py"},
    {"id": "bench", "label": "Benchmarking", "page": "pages/4_Benchmarking.py"},
    {"id": "gis_map", "label": "GIS Map", "page": "pages/5_GIS_Map.py"},
    {"id": "assistant", "label": "Assistant", "page": "pages/6_Assistant.py"},
]


def render_navbar(current_page: str) -> None:
    """Render a full-width navy slip with white tabs and | spacers."""
    with st.container():
        st.markdown('<div class="navy-bar-anchor"></div>', unsafe_allow_html=True)

        weights = [
            0.95, 0.12, 1.25, 0.12, 1.25, 0.12, 0.9, 0.12,
            1.2, 0.12, 0.85, 0.12, 1.05,
        ]
        cols = st.columns(weights)

        tab_slots = [0, 2, 4, 6, 8, 10, 12]
        divider_slots = [1, 3, 5, 7, 9, 11]

        for idx, item in enumerate(NAV_ITEMS):
            is_active = current_page == item["id"]
            with cols[tab_slots[idx]]:
                if st.button(
                    item["label"],
                    key=f"topnav_{item['id']}",
                    use_container_width=True,
                    type="primary" if is_active else "secondary",
                ):
                    if not is_active:
                        st.switch_page(item["page"])

        for d_idx in divider_slots:
            with cols[d_idx]:
                st.markdown('<div class="topbar-divider">|</div>', unsafe_allow_html=True)
