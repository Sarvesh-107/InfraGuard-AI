"""Professional glassmorphism theme, navy blue top navbar, dark mode, and layout styles."""

from __future__ import annotations

import streamlit as st

from risk_engine import RISK_COLORS

THEME_VARS = {
    "light": {
        "bg_gradient": "linear-gradient(135deg, #eef2f7 0%, #e0e7ff 50%, #e2e8f0 100%)",
        "card_bg": "rgba(255, 255, 255, 0.62)",
        "card_bg_strong": "rgba(255, 255, 255, 0.82)",
        "card_border": "rgba(255, 255, 255, 0.72)",
        "card_shadow": "0 8px 32px rgba(31, 38, 135, 0.08)",
        "glass_btn_bg": "rgba(255, 255, 255, 0.42)",
        "glass_btn_border": "rgba(255, 255, 255, 0.55)",
        "text_primary": "#0f172a",
        "text_secondary": "#475569",
        "text_muted": "#64748b",
        "accent": "#2563eb",
        "accent_soft": "rgba(37, 99, 235, 0.12)",
        "input_bg": "rgba(255, 255, 255, 0.85)",
        "divider": "rgba(15, 23, 42, 0.08)",
        "chart_line": "#2563eb",
    },
    "dark": {
        "bg_gradient": "linear-gradient(135deg, #0a0f1d 0%, #10192b 48%, #152238 100%)",
        "card_bg": "rgba(21, 34, 56, 0.55)",
        "card_bg_strong": "rgba(21, 34, 56, 0.82)",
        "card_border": "rgba(255, 255, 255, 0.14)",
        "card_shadow": "0 8px 32px rgba(0, 0, 0, 0.38)",
        "glass_btn_bg": "rgba(255, 255, 255, 0.08)",
        "glass_btn_border": "rgba(255, 255, 255, 0.16)",
        "text_primary": "#f8fafc",
        "text_secondary": "#cbd5e1",
        "text_muted": "#94a3b8",
        "accent": "#3b82f6",
        "accent_soft": "rgba(59, 130, 246, 0.18)",
        "input_bg": "rgba(15, 23, 42, 0.75)",
        "divider": "rgba(255, 255, 255, 0.10)",
        "chart_line": "#3b82f6",
    },
}


def init_theme() -> None:
    if "dark_mode" not in st.session_state:
        st.session_state.dark_mode = False


def is_dark_mode() -> bool:
    init_theme()
    return bool(st.session_state.dark_mode)


def inject_custom_css() -> None:
    dark = is_dark_mode()
    t = THEME_VARS["dark" if dark else "light"]

    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

        :root {{
            --bg-gradient: {t["bg_gradient"]};
            --card-bg: {t["card_bg"]};
            --card-bg-strong: {t["card_bg_strong"]};
            --card-border: {t["card_border"]};
            --card-shadow: {t["card_shadow"]};
            --glass-btn-bg: {t["glass_btn_bg"]};
            --glass-btn-border: {t["glass_btn_border"]};
            --text-primary: {t["text_primary"]};
            --text-secondary: {t["text_secondary"]};
            --text-muted: {t["text_muted"]};
            --accent: {t["accent"]};
            --accent-soft: {t["accent_soft"]};
            --input-bg: {t["input_bg"]};
            --divider: {t["divider"]};
        }}

        html, body, [class*="css"] {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
            color: var(--text-primary) !important;
        }}

        .stApp {{
            background: var(--bg-gradient) !important;
            background-attachment: fixed !important;
            color: var(--text-primary) !important;
        }}

        [data-testid="stSidebar"],
        section[data-testid="stSidebar"],
        [data-testid="stSidebarNav"],
        [data-testid="stSidebarCollapseButton"],
        [data-testid="stExpandSidebarButton"],
        [data-testid="collapsedControl"],
        [data-testid="stSidebarNavSeparator"] {{
            display: none !important;
            width: 0 !important;
            min-width: 0 !important;
            height: 0 !important;
            visibility: hidden !important;
        }}

        section.main, [data-testid="stMain"] {{
            background: transparent !important;
            padding-left: 1.6rem !important;
            padding-right: 1.6rem !important;
            padding-top: 0.55rem !important;
            max-width: 100% !important;
        }}

        .main .block-container,
        [data-testid="stMainBlockContainer"] {{
            padding-top: 0.15rem !important;
            padding-left: 0 !important;
            padding-right: 0 !important;
            margin-left: 0 !important;
            max-width: 100% !important;
            position: relative;
        }}

        header[data-testid="stHeader"], [data-testid="stHeader"] {{
            background: transparent !important;
            height: 0 !important;
            min-height: 0 !important;
            padding: 0 !important;
        }}

        /* Hide the tiny markdown markers used to locate the navy slip and filter */
        [data-testid="stElementContainer"]:has(.navy-bar-anchor),
        [data-testid="stElementContainer"]:has(.filter-round-anchor) {{
            display: none !important;
        }}

        /* Paint the horizontal nav row as a single navy slip */
        [data-testid="stElementContainer"]:has(.navy-bar-anchor) + [data-testid="stHorizontalBlock"],
        [data-testid="stElementContainer"]:has(.navy-bar-anchor) + div {{
            background: #0b192c !important;
            border-radius: 12px !important;
            padding: 0.28rem 0.55rem !important;
            margin: 0 0 1.05rem 0 !important;
            box-shadow: 0 10px 28px rgba(11, 25, 44, 0.42) !important;
            border: 1px solid rgba(255, 255, 255, 0.12) !important;
            align-items: center !important;
        }}

        /* Glassmorphism for ALL buttons */
        .stButton > button,
        [data-testid="stDownloadButton"] > button,
        [data-testid="stPopover"] button,
        [data-testid="stPopoverButton"] > button,
        [data-testid="stFormSubmitButton"] > button,
        [data-testid="stBaseButton-secondary"],
        [data-testid="stBaseButton-primary"],
        button[kind="secondary"],
        button[kind="primary"],
        button[data-testid="baseButton-secondary"],
        button[data-testid="baseButton-primary"] {{
            background: var(--glass-btn-bg) !important;
            backdrop-filter: blur(14px) saturate(160%) !important;
            -webkit-backdrop-filter: blur(14px) saturate(160%) !important;
            border: 1px solid var(--glass-btn-border) !important;
            border-radius: 12px !important;
            color: var(--text-primary) !important;
            font-weight: 500 !important;
            box-shadow: 0 4px 18px rgba(15, 23, 42, 0.08), inset 0 1px 0 rgba(255, 255, 255, 0.28) !important;
            transition: transform 0.18s ease, box-shadow 0.18s ease, border-color 0.18s ease, background 0.18s ease !important;
        }}

        .stButton > button p,
        [data-testid="stDownloadButton"] > button p,
        [data-testid="stPopoverButton"] > button p,
        button[kind="secondary"] p,
        button[kind="primary"] p {{
            color: inherit !important;
        }}

        .stButton > button:hover,
        [data-testid="stDownloadButton"] > button:hover,
        [data-testid="stPopoverButton"] > button:hover,
        [data-testid="stFormSubmitButton"] > button:hover,
        button[kind="secondary"]:hover {{
            background: var(--card-bg-strong) !important;
            border-color: var(--accent) !important;
            color: var(--accent) !important;
            transform: translateY(-2px);
            box-shadow: 0 10px 24px rgba(37, 99, 235, 0.18), inset 0 1px 0 rgba(255, 255, 255, 0.35) !important;
        }}

        .stButton > button[kind="primary"],
        button[kind="primary"],
        [data-testid="stBaseButton-primary"],
        button[data-testid="baseButton-primary"] {{
            background: color-mix(in srgb, var(--accent) 78%, transparent) !important;
            color: #ffffff !important;
            border: 1px solid color-mix(in srgb, var(--accent) 90%, white) !important;
            font-weight: 600 !important;
        }}

        /* Navy slip tabs: white type, no glass fill */
        [data-testid="stElementContainer"]:has(.navy-bar-anchor) + [data-testid="stHorizontalBlock"] button,
        [data-testid="stElementContainer"]:has(.navy-bar-anchor) + div button {{
            background: transparent !important;
            border: 1px solid transparent !important;
            color: #ffffff !important;
            font-size: 0.86rem !important;
            font-weight: 500 !important;
            padding: 0.32rem 0.45rem !important;
            min-height: 2.35rem !important;
            border-radius: 8px !important;
            box-shadow: none !important;
            backdrop-filter: none !important;
            -webkit-backdrop-filter: none !important;
        }}

        [data-testid="stElementContainer"]:has(.navy-bar-anchor) + [data-testid="stHorizontalBlock"] button p,
        [data-testid="stElementContainer"]:has(.navy-bar-anchor) + div button p {{
            color: #ffffff !important;
            font-size: 0.86rem !important;
        }}

        [data-testid="stElementContainer"]:has(.navy-bar-anchor) + [data-testid="stHorizontalBlock"] button[kind="primary"],
        [data-testid="stElementContainer"]:has(.navy-bar-anchor) + div button[kind="primary"] {{
            background: rgba(255, 255, 255, 0.16) !important;
            border: 1px solid rgba(255, 255, 255, 0.32) !important;
            color: #ffffff !important;
            font-weight: 600 !important;
        }}

        [data-testid="stElementContainer"]:has(.navy-bar-anchor) + [data-testid="stHorizontalBlock"] button:hover,
        [data-testid="stElementContainer"]:has(.navy-bar-anchor) + div button:hover {{
            background: rgba(255, 255, 255, 0.12) !important;
            color: #ffffff !important;
            border-color: rgba(255, 255, 255, 0.28) !important;
            transform: none;
            box-shadow: none !important;
        }}

        .topbar-divider {{
            color: rgba(255, 255, 255, 0.38) !important;
            text-align: center;
            font-weight: 300;
            font-size: 1.05rem;
            line-height: 2.15rem;
            user-select: none;
        }}

        /* Round funnel filter trigger */
        [data-testid="stPopover"] button,
        [data-testid="stPopoverButton"] button {{
            border-radius: 9999px !important;
            padding: 0.42rem 1.15rem 0.42rem 0.95rem !important;
            font-size: 0.84rem !important;
            font-weight: 600 !important;
            display: inline-flex !important;
            align-items: center !important;
            justify-content: center !important;
            gap: 0.4rem !important;
            background: var(--glass-btn-bg) !important;
            border: 1px solid var(--glass-btn-border) !important;
            color: var(--text-primary) !important;
        }}

        [data-testid="stPopover"] button [data-testid="stIconMaterial"] {{
            display: none !important;
        }}

        [data-testid="stPopover"] button::before {{
            content: "";
            display: inline-block;
            width: 14px;
            height: 14px;
            flex: 0 0 14px;
            background: currentColor;
            mask: url('data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="black" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><polygon points="22 3 2 3 10 12.46 10 19 14 21 14 12.46 22 3"/></svg>') no-repeat center / contain;
            -webkit-mask: url('data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="black" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><polygon points="22 3 2 3 10 12.46 10 19 14 21 14 12.46 22 3"/></svg>') no-repeat center / contain;
        }}

        [data-testid="stPopoverBody"],
        div[data-testid="stPopover"] [data-baseweb="popover"] > div {{
            background: var(--card-bg) !important;
            backdrop-filter: blur(16px) saturate(160%) !important;
            -webkit-backdrop-filter: blur(16px) saturate(160%) !important;
            border: 1px solid var(--card-border) !important;
            border-radius: 14px !important;
            box-shadow: var(--card-shadow) !important;
        }}

        .filter-popover-title {{
            font-weight: 700;
            font-size: 0.95rem;
            margin-bottom: 0.55rem;
            border-bottom: 1px solid var(--divider);
            padding-bottom: 0.4rem;
            color: var(--text-primary);
        }}

        [data-testid="stVerticalBlockBorderWrapper"] {{
            background: var(--card-bg) !important;
            backdrop-filter: blur(16px) saturate(150%);
            -webkit-backdrop-filter: blur(16px) saturate(150%);
            border: 1px solid var(--card-border) !important;
            border-radius: 14px !important;
            box-shadow: var(--card-shadow) !important;
            padding: 0.85rem !important;
        }}

        div[data-testid="stMetric"] {{
            background: var(--card-bg-strong) !important;
            backdrop-filter: blur(14px);
            -webkit-backdrop-filter: blur(14px);
            border-radius: 12px !important;
            padding: 0.9rem 1.1rem !important;
            box-shadow: var(--card-shadow) !important;
            border: 1px solid var(--card-border) !important;
        }}
        div[data-testid="stMetric"] label,
        div[data-testid="stMetric"] [data-testid="stMetricLabel"] p {{
            font-size: 0.78rem !important;
            color: var(--text-muted) !important;
            font-weight: 600 !important;
            text-transform: uppercase;
            letter-spacing: 0.04em;
        }}
        div[data-testid="stMetric"] div[data-testid="stMetricValue"],
        div[data-testid="stMetric"] div[data-testid="stMetricValue"] * {{
            font-size: 1.55rem !important;
            font-weight: 700 !important;
            color: var(--text-primary) !important;
        }}

        .kpi-glass {{
            background: var(--card-bg);
            backdrop-filter: blur(14px) saturate(160%);
            -webkit-backdrop-filter: blur(14px) saturate(160%);
            border: 1px solid var(--card-border);
            border-radius: 14px;
            box-shadow: var(--card-shadow);
            padding: 0.95rem 1.1rem 0.85rem;
            min-height: 6.2rem;
        }}
        .kpi-label {{
            font-size: 0.72rem;
            font-weight: 600;
            letter-spacing: 0.06em;
            text-transform: uppercase;
            color: var(--text-muted) !important;
            margin-bottom: 0.35rem;
        }}
        .kpi-value {{
            font-size: 1.55rem;
            font-weight: 700;
            letter-spacing: -0.03em;
            color: var(--text-primary) !important;
            line-height: 1.2;
        }}
        .kpi-hint {{
            margin-top: 0.28rem;
            font-size: 0.78rem;
            color: var(--text-secondary) !important;
        }}

        .project-row {{
            background: var(--card-bg);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            border: 1px solid var(--card-border);
            border-radius: 12px;
            padding: 0.65rem 0.85rem 0.2rem;
            margin-bottom: 0.55rem;
            box-shadow: var(--card-shadow);
        }}

        h1, h2, h3, h4, h5, h6, p, label, span, li, dt, dd, div {{
            color: var(--text-primary);
        }}
        .page-subtitle {{
            color: var(--text-secondary) !important;
            font-size: 0.95rem;
            margin-top: -0.15rem;
            margin-bottom: 0.85rem;
        }}

        [data-baseweb="select"] > div,
        [data-baseweb="input"] > div,
        [data-baseweb="base-input"],
        [data-baseweb="input"],
        div[role="combobox"],
        input, textarea {{
            background-color: var(--input-bg) !important;
            border-color: var(--card-border) !important;
            color: var(--text-primary) !important;
            border-radius: 8px !important;
        }}

        .alert-card {{
            border-left: 4px solid var(--accent);
            padding-left: 0.85rem;
            color: var(--text-primary);
        }}

        .risk-badge {{
            display: inline-block;
            padding: 0.2rem 0.65rem;
            border-radius: 6px;
            font-size: 0.75rem;
            font-weight: 600;
            color: #fff !important;
            white-space: nowrap;
        }}
        .risk-badge-critical {{ background: {RISK_COLORS["Critical"]}; color: #fff !important; }}
        .risk-badge-high {{ background: {RISK_COLORS["High"]}; color: #fff !important; }}
        .risk-badge-medium {{ background: {RISK_COLORS["Medium"]}; color: #1a1a1a !important; }}
        .risk-badge-low {{ background: {RISK_COLORS["Low"]}; color: #fff !important; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def page_header(title: str, subtitle: str) -> None:
    """Render page title and subtitle cleanly."""
    st.markdown(
        f"""
        <div style="margin-top: 0.1rem; margin-bottom: 0.15rem;">
            <h1 style="margin-bottom:0.1rem; font-weight:700; font-size:1.55rem; letter-spacing:-0.02em;">{title}</h1>
            <p class="page-subtitle">{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def glass_divider() -> None:
    st.markdown("---")
