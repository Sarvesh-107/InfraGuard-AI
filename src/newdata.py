"""Loader for data/newdata/ -- the monthly PAIMANA table exports.

Table_3 = completed during month, Table_4 = added during month,
Table_6/7 = ongoing roster as of month end (Sept-2024 ships as Table_6).

Everything downstream reads the data through here, never raw, so the parsing
semantics below are defined exactly once.
"""
import pandas as pd, numpy as np, re, pathlib, calendar

DATA = pathlib.Path(__file__).resolve().parents[1] / "data/newdata"

# Rows in these two exports are structurally corrupt, not merely shifted: cells
# hold pipe-joined fragments ("234.27 | (N.A.) | {234.27}"), project names are
# split across several fields, and some rows interleave two projects. There is no
# parse that recovers them without inventing numbers, so they are excluded and
# the gap is declared rather than patched.
CORRUPT = {"2024-06", "2024-08"}

MONTHS = {m.lower(): i for i, m in enumerate(calendar.month_name) if m}
RENAME = {
    "Sl No": "sl_no", "Sl. No.": "sl_no", "Project Name": "project_name",
    "Agency Name": "agency", "Project Code": "project_code", "State": "state",
    "State Name": "state", "Sector": "sector",
    "Date of Approval (MM/YYYY)": "approval_date",
    "Date of Commissioning Original": "doc_original",
    "Date of Commissioning Original (MM/YYYY)": "doc_original",
    "Date of Commissioning Revised": "doc_revised",
    "Date of Commissioning Anticipated": "doc_anticipated",
    "Cost Original": "cost_original", "Cost Revised": "cost_revised",
    "Cost Anticipated": "cost_anticipated",
    "Original Cost in Rs. Crore": "cost_original",
    "Cumulative Expenditure in Rs. Crore": "expenditure",
    "Physical Progress (%)": "physical_progress_pct",
    "Progress (%)": "physical_progress_pct",      # Nov-2024 export's header
}


def _period_from_name(name):
    m = re.search(r"(January|February|March|April|May|June|July|August|September|"
                  r"October|November|December)_(\d{4})", name)
    return f"{m.group(2)}-{MONTHS[m.group(1).lower()]:02d}" if m else None


def _num(s):
    """'1,428.25' -> 1428.25 ; 'N.A.' / '' -> NaN. Never silently 0: N.A. in the
    revised columns means NOT REVISED, which is a different fact from zero cost."""
    return pd.to_numeric(
        s.astype(str).str.replace(",", "", regex=False)
         .str.replace(r"\s+", "", regex=True).replace({"N.A.": None, "": None, "nan": None, "-": None}),
        errors="coerce")


def _date(s):
    """Handles the five shapes seen in these exports: '3-2019', '9/2018', '11/2019', the
    Excel-mangled 'Dec-24', and 'Jun-2023' (May/Jun-2025 revised dates -- missing
    this silently marked every live rescheduled project as 'never revised').
    'N.A.' means not revised -> NaT, not an error."""
    # Dec-24/Jan-25/Mar-25 exports put stray spaces INSIDE values ('2 / 2 0 2 6',
    # 'J un-25'); ~880 original dates were silently lost before this line.
    v = s.astype(str).str.replace(r"\s+", "", regex=True).str.strip("{}")
    out = pd.Series(pd.NaT, index=v.index, dtype="datetime64[ns]")
    for pat, fmt in [(r"^\d{1,2}/\d{4}$", "%m/%Y"),
                     # approval dates from Dec-2024 on: '3-2019' (were all lost)
                     (r"^\d{1,2}-\d{4}$", "%m-%Y"),
                     (r"^[A-Za-z]{3}-\d{2}$", "%b-%y"),
                     (r"^[A-Za-z]{3}-\d{4}$", "%b-%Y")]:
        m = v.str.match(pat, na=False)
        out[m] = pd.to_datetime(v[m], format=fmt, errors="coerce")
    return out


# Sector labels arrive truncated at varying widths. An explicit alias map beats a
# clever prefix rule: prefix-merging silently folded 'CIVIL AVIATION' into a
# run-together value and left the suffix fragment 'HIGHWAYS' stranded.
SECTOR_ALIAS = {
    "ROAD TRANSPORT AND": "ROAD TRANSPORT AND HIGHWAYS",
    "HIGHWAYS": "ROAD TRANSPORT AND HIGHWAYS",
    "TELECOMMUNICA": "TELECOMMUNICATIONS",
    # Checked across all 36 raw exports: "DEPARTMENT OF" is the only truncated
    # "DEPARTMENT..." value that ever appears (2644 distinct raw sector strings
    # total), and every project_code that shows it also shows the full
    # "DEPARTMENT OF HIGHER EDUCATION" in some other month -- no other department
    # (school education, water resources, etc.) collides with this prefix. Safe as
    # a single generic key; re-check if a differently-truncated "DEPARTMENT..."
    # value ever shows up in a new export.
    "DEPARTMENT OF": "DEPARTMENT OF HIGHER EDUCATION",
    # 2 rows where three sector names ran together -- unresolvable, so left blank
    # rather than guessed into one of them.
    "CIVIL AVIATION RAILWAYS ROAD TRANSPORT AND HIGHWAYS": None,
}


def _norm_sector(s):
    v = s.astype(str).str.strip().str.upper().replace({"NAN": None})
    return v.map(lambda x: SECTOR_ALIAS.get(x, x) if pd.notna(x) else x)


def _read(path, kind):
    d = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    d = d.rename(columns={k: v for k, v in RENAME.items() if k in d.columns})
    d["period"] = _period_from_name(path.name)
    d["table_kind"] = kind
    for c in ["cost_original", "cost_revised", "cost_anticipated", "expenditure",
              "physical_progress_pct"]:
        if c in d:
            d[c] = _num(d[c])
    for c in ["approval_date", "doc_original", "doc_revised", "doc_anticipated"]:
        if c in d:
            d[c] = _date(d[c])
    if "sector" in d:
        d["sector"] = _norm_sector(d["sector"])
    for c in ["agency", "state", "project_name"]:
        if c in d:
            d[c] = d[c].astype(str).str.strip().str.upper().replace({"NAN": None})
    d["project_code"] = d["project_code"].astype(str).str.strip()
    # Each export ends with a TOTAL summary row (cost ~Rs 27 lakh cr, no code).
    # Kept as a project it is a portfolio-sized outlier, so drop anything whose
    # code is not a real project id.
    d = d[d["project_code"].str.fullmatch(r"N?\d{8,9}", na=False)]
    keep = [c for c in ["period", "table_kind", "project_code", "project_name",
                        "sector", "agency", "state", "approval_date", "doc_original",
                        "doc_revised", "doc_anticipated", "cost_original",
                        "cost_revised", "cost_anticipated", "expenditure",
                        "physical_progress_pct"] if c in d.columns]
    return d[keep]


def load(verbose=True):
    """-> (ongoing, completed, added) tidy frames. `completed` is sorted by
    (project_code, period) ascending, so a caller doing
    drop_duplicates("project_code", keep="first") gets the chronologically
    earliest completion, not whichever export file sorted first alphabetically."""
    frames = {"ongoing": [], "completed": [], "added": [], "skipped": []}
    for p in sorted(DATA.glob("*.csv")):
        per = _period_from_name(p.name)
        kind = ("ongoing" if p.name.startswith(("Table_6", "Table_7"))
                else "completed" if p.name.startswith("Table_3")
                else "added" if p.name.startswith("Table_4") else None)
        if kind is None:
            continue
        if kind == "ongoing" and per in CORRUPT:
            frames["skipped"].append((per, p.name))
            continue
        frames[kind].append(_read(p, kind))

    ong = pd.concat(frames["ongoing"], ignore_index=True)
    ong = (ong.drop_duplicates(["period", "project_code"])
              .sort_values(["project_code", "period"]).reset_index(drop=True))
    g = ong.groupby("project_code")

    # From Apr-2025 (and a few earlier months) the report OVERWRITES the "original"
    # cost of 132 projects with a newer sanction -- Rishikesh-Karnaprayag went
    # 16,216 -> 38,953 cr in the "original" column while "revised" stayed N.A. and
    # the agency's expected cost stayed 24,659, so it looked 14,294 cr UNDER budget.
    # The first original ever reported is the approved cost; a later rewrite is a
    # revision, and is recorded as one.
    first = g.cost_original.transform("first")
    restated = (ong.cost_original - first).abs() > 0.001 * first
    ong.loc[restated & ong.cost_revised.isna(), "cost_revised"] = ong.cost_original
    ong["cost_original"] = first      # also absorbs sub-0.1% rounding re-prints

    # 2025 exports write 0% progress as "-" (143 of 170 such projects were at 0%
    # the month before). Read as unknown, the model imputed an average and made
    # not-started projects look well under way. So "-" is 0% -- unless the project
    # already had real progress, where it just means "not updated this month".
    # Uses the HIGHEST progress reported so far: Rishikesh-Karnaprayag read 82% ->
    # 0% -> 0% -> "-", and a last-value rule turned the dashes into a false 0%.
    best = g.physical_progress_pct.transform(lambda s: s.cummax().ffill().shift())
    # a reported 0% after real progress is a reporting glitch, not works undone
    ong.loc[(ong.physical_progress_pct == 0) & (best > 20), "physical_progress_pct"] = np.nan
    ong.loc[ong.physical_progress_pct.isna() & ~(best > 0.5), "physical_progress_pct"] = 0.0
    # DATA.glob() + sorted() orders files ALPHABETICALLY, not chronologically
    # ("...April_2025.csv" < "...December_2024.csv"), so a project completed in
    # more than one monthly export would keep whichever file happened to sort
    # first -- not its true earliest completion month -- wherever a caller later
    # does drop_duplicates("project_code") on this frame. `period` is parsed from
    # each file's actual year/month (zero-padded "YYYY-MM", see
    # _period_from_name), so sorting by it string-sorts chronologically too.
    comp = (pd.concat(frames["completed"], ignore_index=True)
              .sort_values(["project_code", "period"]).reset_index(drop=True))
    add = pd.concat(frames["added"], ignore_index=True)

    # 'Revised' N.A. means not revised -> the effective plan of record is the
    # revised value when one exists, else the original.
    ong["effective_cost"] = ong.cost_revised.fillna(ong.cost_original)
    ong["effective_doc"] = ong.doc_revised.fillna(ong.doc_original)
    ong["is_cost_revised"] = ong.cost_revised.notna().astype(int)
    ong["is_schedule_revised"] = ong.doc_revised.notna().astype(int)

    if verbose:
        print(f"ongoing   : {len(ong):5d} rows  {ong.period.nunique()} periods "
              f"{sorted(ong.period.unique())}")
        print(f"completed : {len(comp):5d} rows  {comp.project_code.nunique()} unique")
        print(f"added     : {len(add):5d} rows")
        print(f"skipped   : {[s[0] for s in frames['skipped']]} (corrupt exports)")
    return ong, comp, add


if __name__ == "__main__":
    ong, comp, add = load()
    print(f"\nunique projects: {ong.project_code.nunique()}  sectors: {ong.sector.nunique()}")
    print(f"sectors: {sorted(ong.sector.dropna().unique())}")
    n = ong.groupby("project_code").size()
    print(f"\nsnapshots per project: {n.describe()[['mean','50%','max']].to_dict()}")
    print(f"projects seen in all 10: {(n == 10).sum()}")
    # how often does the agency's own anticipated figure already exceed the plan?
    a = ong.dropna(subset=["cost_anticipated", "effective_cost"])
    print(f"\nanticipated cost > effective plan: {(a.cost_anticipated > a.effective_cost*1.001).mean():.1%}")
    b = ong.dropna(subset=["doc_anticipated", "effective_doc"])
    print(f"anticipated date > effective plan: {((b.doc_anticipated - b.effective_doc).dt.days > 30).mean():.1%}")
