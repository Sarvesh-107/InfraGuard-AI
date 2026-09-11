"""Investigates whether the live PAIMANA portal's POST /Home/GetTileData endpoint
returns a usable per-project location field (real lat/lon, or a district/taluk name
that could be geocoded) -- something forecast_cards_*.csv does not have.

Usage:
    python scripts/fetch_paimana_coords.py

What it does:
    1. GETs the public dashboard page to pick up a session cookie and the ASP.NET
       anti-forgery token embedded in the HTML (the endpoint is behind
       [ValidateAntiForgeryToken], confirmed by inspecting the page's own AJAX call
       in Content of inline <script> tags -- it posts $("#myForm").serialize()).
    2. POSTs to GetTileData once per state (StateId 1..N from the page's own <select>)
       -- an unfiltered POST 500s with "the length of the string exceeds the value set
       on the maxJsonLength property", so the response has to be sliced by filter.
    3. Prints the full set of field names seen across every project record, and a
       sample of 5 records, so the schema is confirmed against the live response
       rather than assumed.
    4. Reports which fields look like a project id/code and which (if any) look like
       a real lat/lon or geocodable location field.

Findings from the run recorded in this repo (see _ensure_lat_lon in app/data_loader.py
for the one-line summary): every project record has this field set --

    ProjectId, ProjectName, SectorName, StateName, LineMinistry, COMPANYNAME,
    AgencyId, AgencyName, OriginalCost, RevisedCost, RevisedCostReason, Expenditure,
    SanctionDate, CreationDate, StartDate, OriginalEndDate, RevisedDate,
    RevisedDateReason, DELAYED_TIME, COST_OVERRUN_PERC, COST_OVERRUN, COR_PERC,
    TOR_PERC, PhysicalProgress, OnboardingDelay, Remarks

-- no latitude/longitude field, and no district/taluk/address field of any kind.
StateName itself is always null even when the request is filtered by StateId (i.e.
even state-level location isn't populated on this endpoint -- our own CSV's `state`
column is the only place state is reliably known). Additionally, ProjectId here is a
bare integer (e.g. 701101) with no relation to forecast_cards_*.csv's alphanumeric
project_id ("N24001567" etc.) -- there is no shared key to join on even if a location
field existed. Conclusion: Case 3, no usable location field -- see app/data_loader.py.
"""
import json
import re
import sys

import requests

BASE = "https://paimana-proj.mospi.gov.in"
DASHBOARD_URL = f"{BASE}/Home/PublicDashboardNew"
TILE_DATA_URL = f"{BASE}/Home/GetTileData"

# A handful of state ids (scraped once from the dashboard's own <select name="StateId">
# options) -- enough states to get a decently sized, varied sample without re-triggering
# the maxJsonLength 500 that an unfiltered ("StateId": "") POST hits.
SAMPLE_STATE_IDS = {
    "Andaman & Nicobar": "47",
    "Bihar": "4",
    "Delhi": "32",
    "Gujarat": "31",
}

LOCATION_FIELD_HINTS = re.compile(
    r"lat|lon|latitude|longitude|district|taluk|tehsil|block|geo|address|pincode|"
    r"pin_code|coord",
    re.I,
)
ID_FIELD_HINTS = re.compile(r"projectid|project_id|projectcode|project_code", re.I)


def get_session_and_token():
    session = requests.Session()
    resp = session.get(DASHBOARD_URL, timeout=30)
    resp.raise_for_status()
    match = re.search(
        r'name="__RequestVerificationToken"[^>]*value="([^"]+)"', resp.text
    )
    if not match:
        raise RuntimeError(
            "Could not find __RequestVerificationToken on the dashboard page -- "
            "the portal's form markup may have changed."
        )
    return session, match.group(1)


def fetch_tile_data(session, token, state_id, month_year="2026-07"):
    resp = session.post(
        TILE_DATA_URL,
        data={
            "Month": "",
            "Year": "",
            "__RequestVerificationToken": token,
            "SectorId": "",
            "PROJ_MINISTRY_ID": "",
            "StateId": state_id,
            "CostRange": "",
            "MonthYear": month_year,
        },
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def main():
    print(f"GET {DASHBOARD_URL} for a session cookie + anti-forgery token ...")
    session, token = get_session_and_token()
    print("  ok\n")

    all_records = []
    for state_name, state_id in SAMPLE_STATE_IDS.items():
        print(f"POST {TILE_DATA_URL} (StateId={state_id} / {state_name}) ...")
        payload = fetch_tile_data(session, token, state_id)
        records = (payload.get("data") or {}).get("ProjectsCountTabDetails") or []
        print(f"  -> {len(records)} project record(s)")
        all_records.extend(records)

    if not all_records:
        print("\nNo records returned at all -- can't confirm the schema. Investigate "
              "the request shape manually (browser devtools) before concluding anything.")
        sys.exit(1)

    field_names = set()
    for r in all_records:
        field_names.update(r.keys())

    print(f"\n{len(all_records)} total record(s) fetched. Full field set seen:")
    for f in sorted(field_names):
        flag = ""
        if LOCATION_FIELD_HINTS.search(f):
            flag = "  <-- possible location field"
        elif ID_FIELD_HINTS.search(f):
            flag = "  <-- possible id/code field"
        print(f"  {f}{flag}")

    print("\nSample of 5 records (raw JSON, exactly as returned):")
    for r in all_records[:5]:
        print(json.dumps(r, indent=2, ensure_ascii=False))

    location_fields = [f for f in field_names if LOCATION_FIELD_HINTS.search(f)]
    non_null_location = {
        f: sum(1 for r in all_records if r.get(f) not in (None, ""))
        for f in location_fields
    }
    print("\n--- Verdict ---")
    if not location_fields:
        print("No field name matches lat/lon/district/taluk/address/geo patterns.")
    else:
        for f, n in non_null_location.items():
            print(f"  {f}: non-null in {n}/{len(all_records)} records")
    print(
        "\nSee the module docstring in this file for the recorded conclusion from "
        "the run performed for this investigation, and app/data_loader.py's "
        "_ensure_lat_lon for how the codebase acts on it."
    )


if __name__ == "__main__":
    main()
