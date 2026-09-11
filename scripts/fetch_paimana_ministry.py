"""Investigates whether the live PAIMANA portal's POST /Home/GetTileData endpoint's
LineMinistry field can replace app/data_loader.py's `out["ministry"] = out["sector"]`
placeholder with a real per-project ministry.

Two separate questions, both answered by this script:
    1. Field coverage: is LineMinistry populated (non-null/non-empty) for most records?
    2. Joinability: can a live record be reliably matched back to a row in our
       forecast_cards_*.csv? (There is no shared id -- see scripts/fetch_paimana_coords.py,
       which already found the live ProjectId is a bare integer unrelated to our
       alphanumeric project_code -- so the only thing to join on is project name.)

Usage:
    python scripts/fetch_paimana_ministry.py

What it does:
    1. Session + anti-forgery token, same as fetch_paimana_coords.py.
    2. Fetches the full state list from /Home/GetStateList (35 states/UTs) instead of
       a handful -- item 1 of the ministry investigation asks for a proper coverage
       number, which needs a near-complete crawl, not a small sample.
    3. POSTs GetTileData once per state, collecting every project record (deduped by
       ProjectId, since a multi-state project could theoretically appear more than
       once -- in practice each project has exactly one StateId server-side).
    4. Reports LineMinistry non-null coverage across every record fetched.
    5. Loads reports/forecast_cards_*.csv (the file app/data_loader.py reads) and
       attempts a normalized-name join: uppercase, collapse whitespace, strip
       punctuation, on both sides. Reports what fraction of our CSV's projects find
       an exact normalized-name match among the live records, and prints a handful
       of matched pairs (to sanity-check the ministry values look right) and
       unmatched examples (to show why/whether the join is trustworthy).

Findings from the run recorded in this repo (see the comment on
`out["ministry"] = out["sector"]` in app/data_loader.py): LineMinistry itself is
populated on effectively every live record (this endpoint's field coverage is good),
but the live portal is a *current* (e.g. July 2026) live snapshot while
forecast_cards_*.csv is a point-in-time export (June 2025) -- a year of projects
completing, being added, and being renamed/re-edited separately in each system. There
is no shared id to join on (confirmed separately in fetch_paimana_coords.py), so name
matching is the only option, and the normalized-name match rate came back too low to
trust as a data source: see the printed verdict for the exact number from the run.
"""
import glob
import json
import re
import sys

import pandas as pd
import requests

BASE = "https://paimana-proj.mospi.gov.in"
DASHBOARD_URL = f"{BASE}/Home/PublicDashboardNew"
STATE_LIST_URL = f"{BASE}/Home/GetStateList"
TILE_DATA_URL = f"{BASE}/Home/GetTileData"


def get_session_and_token():
    session = requests.Session()
    resp = session.get(DASHBOARD_URL, timeout=30)
    resp.raise_for_status()
    match = re.search(r'name="__RequestVerificationToken"[^>]*value="([^"]+)"', resp.text)
    if not match:
        raise RuntimeError("Could not find __RequestVerificationToken on the dashboard page.")
    return session, match.group(1)


def get_state_list(session):
    resp = session.get(STATE_LIST_URL, timeout=30)
    resp.raise_for_status()
    return resp.json()  # [{"Value": 47, "Text": "Andaman & Nicobar"}, ...]


def fetch_tile_data(session, token, state_id, month_year="2026-07"):
    resp = session.post(
        TILE_DATA_URL,
        data={
            "Month": "", "Year": "", "__RequestVerificationToken": token,
            "SectorId": "", "PROJ_MINISTRY_ID": "", "StateId": state_id,
            "CostRange": "", "MonthYear": month_year,
        },
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def normalize_name(name):
    name = str(name or "").upper()
    name = re.sub(r"[^A-Z0-9]+", " ", name)  # punctuation/brackets -> space
    return re.sub(r"\s+", " ", name).strip()


def latest_forecast_cards_path():
    matches = sorted(glob.glob("reports/forecast_cards_*.csv"))
    if not matches:
        raise RuntimeError("No reports/forecast_cards_*.csv found.")
    return matches[-1]


def main():
    print(f"GET {DASHBOARD_URL} for a session cookie + anti-forgery token ...")
    session, token = get_session_and_token()
    print("  ok\n")

    print(f"GET {STATE_LIST_URL} for the full state/UT list ...")
    states = get_state_list(session)
    print(f"  -> {len(states)} states/UTs\n")

    live_by_id = {}
    for s in states:
        state_id, state_name = str(s["Value"]), s["Text"]
        payload = fetch_tile_data(session, token, state_id)
        records = (payload.get("data") or {}).get("ProjectsCountTabDetails") or []
        for r in records:
            live_by_id[r["ProjectId"]] = r
        print(f"  StateId={state_id:>3} {state_name:<40} -> {len(records)} record(s)")

    live_records = list(live_by_id.values())
    print(f"\n{len(live_records)} unique live project record(s) fetched across all states.\n")

    # --- 1. Field coverage ---
    has_ministry = sum(1 for r in live_records if r.get("LineMinistry"))
    print("--- LineMinistry field coverage ---")
    print(f"  {has_ministry}/{len(live_records)} live records have a non-empty LineMinistry "
          f"({has_ministry / len(live_records):.1%})")
    print("  Sample values:", sorted({r["LineMinistry"] for r in live_records[:200] if r.get("LineMinistry")})[:8])

    # --- 2. Joinability against our CSV via normalized project name ---
    csv_path = latest_forecast_cards_path()
    df = pd.read_csv(csv_path)
    print(f"\n--- Name-based join against {csv_path} ({len(df)} projects) ---")

    live_by_norm_name = {}
    for r in live_records:
        live_by_norm_name.setdefault(normalize_name(r["ProjectName"]), []).append(r)

    matched, unmatched_examples, matched_examples = 0, [], []
    for _, row in df.iterrows():
        key = normalize_name(row["project_name"])
        candidates = live_by_norm_name.get(key)
        if candidates:
            matched += 1
            if len(matched_examples) < 5:
                matched_examples.append((row["project_code"], row["project_name"],
                                          candidates[0].get("LineMinistry")))
        elif len(unmatched_examples) < 5:
            unmatched_examples.append((row["project_code"], row["project_name"]))

    print(f"  {matched}/{len(df)} of our CSV's projects find an exact normalized-name "
          f"match among the {len(live_records)} live records ({matched / len(df):.1%})")

    print("\n  Matched examples (project_code, project_name -> LineMinistry):")
    for code, name, ministry in matched_examples:
        print(f"    {code}  {name[:70]!r} -> {ministry!r}")

    print("\n  Unmatched examples (our project_code, project_name):")
    for code, name in unmatched_examples:
        print(f"    {code}  {name[:70]!r}")

    print(
        "\nSee the module docstring in this file for the recorded conclusion from "
        "the run performed for this investigation, and app/data_loader.py's "
        "comment on out[\"ministry\"] for how the codebase acts on it."
    )


if __name__ == "__main__":
    main()
