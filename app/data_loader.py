"""Cached PAIMANA forecast-card loading and GIS-compatible aliases."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from risk_engine import TIERS

REPO_ROOT = Path(__file__).resolve().parents[1]
REPORTS = REPO_ROOT / "reports"
MODELS = REPO_ROOT / "models"

DATE_COLS = [
    "date_approval",
    "date_original",
    "date_revised",
    "date_agency_says",
    "date_projected_p50",
    "date_projected_p80",
]

FRIENDLY = {
    "f_vel_mean": "Average monthly progress",
    "f_vel_last": "Latest monthly progress",
    "f_vel_std": "How uneven progress is",
    "f_burn_mean": "Average monthly spending",
    "f_stall_share": "Share of months with no progress",
    "f_stall_run": "Months in a row with no progress",
    "f_n_snapshots": "Months of reporting history",
    "f_n_doc_changes": "Times the end date moved",
    "f_n_cost_changes": "Times the cost changed",
    "f_n_antic_doc_changes": "Times the agency changed its own estimate",
    "f_progress": "Physical progress",
    "f_fin_progress": "Money spent (% of cost)",
    "f_progress_gap": "Spending running ahead of building",
    "f_age_mo": "Project age",
    "f_planned_mo": "Planned duration",
    "f_elapsed_share": "Share of planned time used",
    "f_months_past_doc": "Months past original end date",
    "f_overdue": "Already past original end date",
    "f_mo_remaining": "Months left to current end date",
    "f_required_vel": "Progress needed per month",
    "f_feasibility": "Actual pace vs needed pace",
    "f_is_cost_revised": "Cost already revised",
    "f_is_sched_revised": "End date already revised",
    "f_cost_revision_pct": "Size of cost revision",
    "f_slip_days": "Delay already declared",
    "f_antic_cost_gap_pct": "Agency expects cost above plan",
    "f_antic_doc_gap_days": "Agency expects date beyond plan",
    "f_log_cost": "Project size (cost)",
}

# lat, lon, lat_radius, lon_radius — keyed lowercase for case-insensitive match
STATE_COORDS = {
    "andaman and nicobar": (11.7, 92.7, 0.8, 0.6),
    "andaman & nicobar": (11.7, 92.7, 0.8, 0.6),
    "andhra pradesh": (15.9, 79.7, 2.0, 1.6),
    "arunachal pradesh": (28.2, 94.7, 1.4, 1.8),
    "assam": (26.2, 92.9, 1.0, 1.8),
    "bihar": (25.6, 85.5, 1.0, 1.5),
    "chandigarh": (30.74, 76.79, 0.12, 0.12),
    "chhattisgarh": (21.3, 82.0, 1.6, 1.6),
    "dadra and nagar haveli": (20.2, 73.0, 0.3, 0.3),
    "daman and diu": (20.4, 72.8, 0.3, 0.3),
    "delhi": (28.61, 77.21, 0.35, 0.35),
    "goa": (15.4, 74.0, 0.4, 0.4),
    "gujarat": (22.3, 71.5, 1.5, 1.8),
    "haryana": (29.2, 76.3, 1.0, 1.2),
    "himachal pradesh": (31.9, 77.2, 1.2, 1.2),
    "jammu and kashmir": (33.5, 75.3, 1.4, 1.4),
    "jammu & kashmir": (33.5, 75.3, 1.4, 1.4),
    "jharkhand": (23.6, 85.3, 1.2, 1.2),
    "karnataka": (14.5, 76.0, 1.8, 1.5),
    "kerala": (10.5, 76.3, 1.4, 0.7),
    "ladakh": (34.2, 77.5, 1.2, 1.5),
    "lakshadweep": (10.6, 72.6, 0.4, 0.5),
    "madhya pradesh": (23.5, 78.0, 1.8, 2.4),
    "maharashtra": (19.5, 75.5, 1.8, 2.2),
    "manipur": (24.8, 93.9, 0.6, 0.6),
    "meghalaya": (25.5, 91.3, 0.6, 0.8),
    "mizoram": (23.2, 92.9, 0.6, 0.6),
    "nagaland": (26.1, 94.5, 0.6, 0.6),
    "odisha": (20.5, 84.5, 1.5, 1.5),
    "puducherry": (11.9, 79.8, 0.3, 0.3),
    "punjab": (31.1, 75.4, 1.0, 1.2),
    "rajasthan": (26.6, 73.8, 1.8, 2.2),
    "sikkim": (27.3, 88.5, 0.4, 0.4),
    "tamil nadu": (11.1, 78.7, 1.6, 1.3),
    "telangana": (17.9, 79.0, 1.2, 1.2),
    "tripura": (23.8, 91.7, 0.5, 0.5),
    "uttar pradesh": (26.8, 80.9, 1.6, 2.0),
    "uttarakhand": (30.1, 79.0, 0.9, 1.0),
    "west bengal": (23.5, 87.8, 1.4, 1.3),
}


def _latest_cards_path() -> Path:
    cards = sorted(REPORTS.glob("forecast_cards_*.csv"))
    if not cards:
        raise FileNotFoundError(f"No forecast_cards_*.csv found in {REPORTS}")
    return cards[-1]


def projects_version() -> float:
    """Cheap (just a filesystem stat, no read) cache key for "the current
    load_projects() data": changes iff a new forecast_cards_*.csv ships. Used
    instead of the projects DataFrame itself as an apply_filters() cache key --
    st.cache_data returns a fresh copy on every call (even on a cache hit, verified:
    the returned object's id() differs every rerun), so hashing/keying on the
    DataFrame's identity doesn't work across reruns, and hashing its full ~1,595-row
    content costs more (~8ms, measured) than just recomputing the filter (~2-3ms)."""
    return _latest_cards_path().stat().st_mtime


def asof_from_cards_path(path: Path | None = None) -> tuple[str, str]:
    path = path or _latest_cards_path()
    asof = path.stem.rsplit("_", 1)[-1]
    asof_label = pd.Timestamp(asof + "-01").strftime("%B %Y")
    return asof, asof_label


def month(d) -> str:
    return d.strftime("%b %Y") if pd.notna(d) else "—"


def lakh_cr(x: float) -> str:
    return f"₹{x / 1e5:,.2f} lakh cr"


def _lookup_state_coords(state: str | None, pname: str) -> tuple[str | None, tuple | None]:
    hay = f"{state or ''} {pname}".lower()
    for key, coords in STATE_COORDS.items():
        if key in hay:
            return key, coords
    return None, None


def _ensure_lat_lon(df: pd.DataFrame) -> pd.DataFrame:
    # Every project gets a jittered-random point (never a real one): the live PAIMANA
    # portal's POST /Home/GetTileData was investigated (scripts/fetch_paimana_coords.py)
    # and confirmed to have no lat/lon or district/taluk field on any project record,
    # and its numeric ProjectId has no join key back to this CSV's project_id -- there is
    # nothing to map real coordinates from. Don't re-investigate; re-run that script only
    # if GetTileData's schema is later reported to have changed.
    if "latitude" in df.columns and "longitude" in df.columns:
        return df

    import numpy as np

    out = df.copy()
    lats, lons = [], []
    for _, row in out.iterrows():
        pname = str(row.get("project_name", ""))
        pid = str(row.get("project_id", row.get("project_code", "0")))
        seed = sum(ord(c) for c in pid)
        rng = np.random.RandomState(seed)
        _, coords = _lookup_state_coords(str(row.get("state", "")), pname)
        if coords:
            lat_c, lon_c, lat_r, lon_r = coords
            lat = lat_c + rng.uniform(-lat_r, lat_r)
            lon = lon_c + rng.uniform(-lon_r, lon_r)
        else:
            lat = 20.5 + rng.uniform(-4.0, 4.0)
            lon = 78.9 + rng.uniform(-4.0, 4.0)
        lats.append(round(lat, 5))
        lons.append(round(lon, 5))
    out["latitude"] = lats
    out["longitude"] = lons
    return out


@st.cache_data(show_spinner=False)
def _load_cards(path: str, mtime: float) -> pd.DataFrame:
    df = pd.read_csv(path)
    for col in DATE_COLS:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], format="%b-%Y", errors="coerce")
    df["risk_tier"] = pd.Categorical(df["risk_tier"], TIERS, ordered=True)
    if "date_confidence" in df.columns:
        df["reliable"] = df["date_confidence"].astype(str).str.startswith("near")
    else:
        df["reliable"] = False
    return df


def load_forecast_cards() -> tuple[pd.DataFrame, str, str]:
    path = _latest_cards_path()
    asof, asof_label = asof_from_cards_path(path)
    df = _load_cards(str(path), path.stat().st_mtime)
    return df, asof, asof_label


@st.cache_data(show_spinner=False)
def _build_projects(path: str, mtime: float) -> pd.DataFrame:
    """The expensive part of load_projects() -- notably _ensure_lat_lon()'s per-row
    jitter loop (~185ms over 1,595 rows, measured) -- keyed on the same (path, mtime)
    _load_cards() uses, so it's computed once per underlying file version and never
    recomputed on an unrelated rerun (a filter change, a widget click, etc.), but still
    invalidates itself automatically the day a new forecast_cards_*.csv ships. Split
    from load_projects() because *that* function's own cache had no such key -- an
    @st.cache_data with no invalidation-relevant argument caches forever, so it would
    have kept serving the FIRST month's cards even after a new export landed."""
    df = _load_cards(path, mtime)
    asof, _ = asof_from_cards_path(Path(path))
    asof_ts = pd.Timestamp(asof + "-01")
    out = df.copy()
    out["project_id"] = out["project_code"].astype(str)
    # Sector-as-ministry placeholder, not a real ministry field. Investigated
    # (scripts/fetch_paimana_ministry.py): the live portal's LineMinistry field has
    # perfect coverage (100% of records), but there's no id to join it to this CSV's
    # project_code by (see fetch_paimana_coords.py), and a normalized-name join only
    # matches 8.2% of our projects -- the two sides genuinely list the same projects
    # (confirmed for Manipur's Tamenglong-Mahur road packages) but under completely
    # different naming conventions, with near-duplicate package variants (PKG-4A vs
    # 4B) that a fuzzy matcher could easily assign to the wrong project. Not worth
    # the risk of silently-wrong ministries for ~8% coverage -- don't re-investigate,
    # only re-run that script if the live site adds a stable cross-reference id.
    out["ministry"] = out["sector"]
    out["last_updated"] = asof_ts
    out = _ensure_lat_lon(out)
    return out


def load_projects() -> pd.DataFrame:
    """GIS-compatible view of the latest forecast cards."""
    path = _latest_cards_path()
    return _build_projects(str(path), path.stat().st_mtime)


@st.cache_data(show_spinner=False)
def load_reports(mtimes: tuple) -> tuple:
    return (
        pd.read_csv(REPORTS / "oof_predictions_v2.csv"),
        pd.read_csv(REPORTS / "backtest_enddate.csv"),
        pd.read_csv(REPORTS / "backtest_cost.csv"),
        json.loads((REPORTS / "model_comparison_v2.json").read_text(encoding="utf-8")),
    )


def report_mtimes() -> tuple:
    names = [
        "oof_predictions_v2.csv",
        "backtest_enddate.csv",
        "backtest_cost.csv",
        "model_comparison_v2.json",
    ]
    return tuple((REPORTS / f).stat().st_mtime for f in names)


@st.cache_data(show_spinner=False)
def load_drivers(mtime: float | None):
    """Feature importances from the trained XGBoost pipeline, if it can be loaded.
    Cached on the model file's mtime (already computed by the caller): unpickling
    the model and running feature_names/feature_importances_ over it is real,
    non-trivial work that was previously redone on every Models-page rerun.

    Returns None when the pickle is missing or optional ML deps (joblib / sklearn /
    xgboost) are not installed — the Models page still works from saved reports.
    """
    model_path = MODELS / "escalation_xgb_v2.joblib"
    if mtime is None or not model_path.exists():
        return None
    try:
        import joblib
    except ImportError:
        return None
    try:
        m = joblib.load(model_path)
        names = m.named_steps["prep"].get_feature_names_out()
        imp = pd.Series(m.named_steps["clf"].feature_importances_, index=names)
    except Exception:
        return None

    def label(n: str) -> str:
        n = n.split("__", 1)[1] if "__" in n else n
        for pre in ("sector_", "state_"):
            if n.startswith(pre):
                return f"{pre[:-1].title()}: {n[len(pre):].title()}"
        return FRIENDLY.get(n, n)

    return (
        imp.rename(label)
        .groupby(level=0)
        .sum()
        .sort_values(ascending=False)
        .rename_axis("factor")
        .reset_index(name="importance")
    )


@st.cache_data(show_spinner=False)
def apply_filters(
    _df: pd.DataFrame,
    version,
    ministries: list[str] | None = None,
    sectors: list[str] | None = None,
    risk_tiers: list[str] | None = None,
    date_from=None,
    date_to=None,
    search: str | None = None,
) -> pd.DataFrame:
    """Recomputed on every widget change in the filter popover, so it's the one
    function actually re-run on nearly every rerun. Measured ~2-3ms uncached on the
    1,595-row portfolio, which is already fast -- but the naive cache (hashing _df's
    full contents on every call) cost ~8ms just to check for a hit, a net LOSS.
    `_df` (leading underscore -> Streamlit never hashes it) plus the cheap `version`
    key (see projects_version()) avoids that entirely: a cache hit costs ~0.4ms,
    ~5-7x faster than uncached. Caching by _df's own identity doesn't work here --
    st.cache_data returns a fresh copy (new id()) on every call, hit or miss, so
    load_projects()'s output has a different id() every rerun regardless."""
    df = _df
    filtered = df.copy()

    if ministries:
        filtered = filtered[filtered["ministry"].isin(ministries)]
    if sectors:
        filtered = filtered[filtered["sector"].isin(sectors)]
    if risk_tiers:
        filtered = filtered[filtered["risk_tier"].astype(str).isin(risk_tiers)]
    if date_from is not None and "last_updated" in filtered.columns:
        filtered = filtered[filtered["last_updated"] >= pd.Timestamp(date_from)]
    if date_to is not None and "last_updated" in filtered.columns:
        filtered = filtered[filtered["last_updated"] <= pd.Timestamp(date_to)]
    if search:
        q = search.strip().lower()
        if q:
            parts = [filtered["project_name"].str.lower().str.contains(q, na=False)]
            if "project_id" in filtered.columns:
                parts.append(filtered["project_id"].astype(str).str.lower().str.contains(q, na=False))
            if "ministry" in filtered.columns:
                parts.append(filtered["ministry"].astype(str).str.lower().str.contains(q, na=False))
            if "state" in filtered.columns:
                parts.append(filtered["state"].astype(str).str.lower().str.contains(q, na=False))
            mask = parts[0]
            for p in parts[1:]:
                mask = mask | p
            filtered = filtered[mask]

    return filtered
