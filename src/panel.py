"""Silver -> gold panel for the escalation-risk model.

Design (see reports/MODEL_CARD.md):
  features are strictly as-of T0 = 2026-04, labels come from 2026-05..2026-07.
  One row per project. No panel expansion -- with only 4 report periods it would
  buy nothing and risk same-project rows landing on both sides of a split.
"""
import pandas as pd, numpy as np, pathlib, sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
SILVER = ROOT / "sih1/data/silver/projects_long.csv"
OUT = pathlib.Path(__file__).resolve().parents[1] / "data/gold"

T0 = "2026-04"
LABEL_PERIODS = ["2026-05", "2026-06", "2026-07"]
TEND = "2026-07"
SLIP_DAYS = 30          # DoC pushed further than this counts as a real slip
COST_TOL = 0.001        # 0.1% guards against rounding noise in reprinted costs


def _d(s):
    return pd.to_datetime(s, format="%d-%m-%Y", errors="coerce")


def load_silver():
    d = pd.read_csv(SILVER, encoding="utf-8-sig", low_memory=False)
    # ne_region is a strict SUBSET of all_ongoing (verified: 229/229 overlap in
    # 2026-04). Concatenating the kinds double-counts every NE project.
    ong = d[d.table_kind == "all_ongoing"].drop_duplicates(["period", "project_code"])
    ne = set(d.loc[d.table_kind == "ne_region", "project_code"])
    new = d[d.table_kind == "newly_added"].drop_duplicates(["period", "project_code"])
    comp = d[d.table_kind == "completed_during_month"].drop_duplicates(["project_code"])
    return d, ong, ne, new, comp


def features(base, ne, new, period):
    """Feature frame strictly as-of `period`. Used for both training (T0) and
    live scoring (latest roster) -- one definition, so the two can never drift."""
    f = base.copy()
    num = ["original_cost_cr", "expenditure_cr", "physical_progress_pct",
           "financial_progress_pct", "progress_gap_pct", "cost_overrun_pct",
           "schedule_slip_days", "planned_duration_days", "age_days",
           "effective_cost_cr"]
    for c in num:
        f[c] = pd.to_numeric(f[c], errors="coerce")

    X = pd.DataFrame(index=f.index)
    X["sector"] = f.sector.fillna("Unknown")
    X["ministry"] = f.line_ministry.fillna("Unknown")
    X["agency"] = f.agency.fillna("Unknown").str.lower().str.strip()
    X["state"] = f.state.fillna("Unknown")
    X["log_cost"] = np.log1p(f.original_cost_cr.clip(lower=0))
    X["planned_duration_days"] = f.planned_duration_days
    X["age_days"] = f.age_days
    X["physical_progress_pct"] = f.physical_progress_pct
    X["financial_progress_pct"] = f.financial_progress_pct
    X["progress_gap_pct"] = f.progress_gap_pct
    X["is_cost_revised"] = (f.is_cost_revised.astype(str).str.upper() == "TRUE").astype(int)
    X["is_schedule_revised"] = (f.is_schedule_revised.astype(str).str.upper() == "TRUE").astype(int)
    X["declared_cost_overrun_pct"] = f.cost_overrun_pct
    X["declared_slip_days"] = f.schedule_slip_days
    X["is_ne"] = f.index.isin(ne).astype(int)

    # elapsed_share: how much of the planned window is already spent. The axis
    # every evaluation gets stratified on, so it is metadata as well as a feature.
    X["elapsed_share"] = (f.age_days / f.planned_duration_days.replace(0, np.nan)).clip(0, 5)
    doc_o = _d(f.original_doc)
    asof = pd.Timestamp(period + "-01")
    X["months_past_original_doc"] = ((asof - doc_o).dt.days / 30.44).clip(lower=-120)
    X["already_overdue"] = (X.months_past_original_doc > 0).astype(int)
    doc_e = _d(f.effective_doc)
    rem = (doc_e - asof).dt.days / 30.44
    X["months_remaining"] = rem
    X["required_velocity"] = (100 - f.physical_progress_pct) / rem.where(rem > 0)
    X["is_newly_added"] = f.index.isin(new[new.period == period].project_code).astype(int)
    X["project_name"] = f.project_name
    return X


def build():
    d, ong, ne, new, comp = load_silver()
    base = ong[ong.period == T0].set_index("project_code")
    end = ong[ong.period == TEND].set_index("project_code")

    # --- labels -------------------------------------------------------------
    # Survivors: still on the roster at TEND -> compare their as-of state.
    surv = base.index.intersection(end.index)
    doc0, doc1 = _d(base.loc[surv, "effective_doc"]), _d(end.loc[surv, "effective_doc"])
    slip = ((doc1 - doc0).dt.days > SLIP_DAYS)
    c0 = base.loc[surv, "effective_cost_cr"].astype(float)
    c1 = end.loc[surv, "effective_cost_cr"].astype(float)
    costup = c1 > c0 * (1 + COST_TOL)

    lab = pd.DataFrame(index=surv)
    lab["y_slip"] = slip.fillna(False).astype(int)
    lab["y_cost_up"] = costup.fillna(False).astype(int)
    lab["exit_status"] = "on_roster"

    # Exits: a project that left the roster because it COMMISSIONED is a success,
    # not an escalation. Anything else that vanished is unexplained -> dropped,
    # never silently folded into the negative class.
    gone = base.index.difference(end.index)
    done = comp[comp.period.isin(LABEL_PERIODS)].set_index("project_code").index
    fin = gone.intersection(done)
    lf = pd.DataFrame(index=fin)
    lf["y_slip"] = 0
    lf["y_cost_up"] = 0
    lf["exit_status"] = "completed"
    lab = pd.concat([lab, lf])
    lab["y_adverse"] = ((lab.y_slip == 1) | (lab.y_cost_up == 1)).astype(int)

    X = features(base.loc[lab.index], ne, new, T0)

    g = X.join(lab)
    g.index.name = "project_code"
    OUT.mkdir(parents=True, exist_ok=True)
    g.to_csv(OUT / "panel_t0.csv")

    print(f"panel rows           : {len(g)}")
    print(f"  survivors          : {(g.exit_status=='on_roster').sum()}")
    print(f"  completed in window: {(g.exit_status=='completed').sum()}")
    print(f"  roster at T0       : {len(base)}   unexplained exits dropped: "
          f"{len(base.index.difference(end.index)) - len(fin)}")
    for c in ["y_slip", "y_cost_up", "y_adverse"]:
        print(f"  {c:10s} positives: {g[c].sum():5d}  base rate {g[c].mean():.3f}")
    print(f"  agencies           : {g.agency.nunique()}   sectors: {g.sector.nunique()}")
    return g


if __name__ == "__main__":
    build()
