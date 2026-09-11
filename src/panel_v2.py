"""Gold panel from the 10-month newdata roster.

Training unit = (project, as-of month t). Features use ONLY snapshots <= t;
labels are read from snapshots ~H months after t. Columns are named by
convention so downstream code needs no hardcoded feature list:
    f_*  feature      y_*  label      everything else is metadata
"""
import pandas as pd, numpy as np, pathlib
from newdata import load

OUT = pathlib.Path(__file__).resolve().parents[1] / "data/gold"
HORIZON = 3          # months ahead the label looks
SLIP_DAYS = 30       # a date push beyond this counts as a real slip
COST_TOL = 0.001


def _mo(p):
    y, m = p.split("-")
    return int(y) * 12 + int(m)


def feature_frame(max_period=None):
    """All roster rows with as-of features and no labels. Scoring reuses this so
    train-time and serve-time features cannot drift apart. `max_period` truncates
    the input history, which is how the as-of guarantee is actually tested."""
    ong, comp, add = load(verbose=False)
    ong = ong.copy()
    if max_period is not None:
        ong = ong[ong.period <= max_period]
    ong["mo"] = ong.period.map(_mo)
    ong = ong.sort_values(["project_code", "mo"]).reset_index(drop=True)
    g = ong.groupby("project_code", sort=False)

    # ---- Tier C: velocity. The whole reason a 10-month panel beats a snapshot.
    ong["d_mo"] = g.mo.diff()
    ong["d_prog"] = g.physical_progress_pct.diff()
    ong["d_exp"] = g.expenditure.diff()
    ong["prog_per_mo"] = ong.d_prog / ong.d_mo
    ong["exp_per_mo"] = ong.d_exp / ong.d_mo
    # expanding = strictly as-of: mean/std over this snapshot and earlier only
    ong["f_vel_mean"] = g.prog_per_mo.transform(lambda s: s.expanding().mean())
    ong["f_vel_last"] = ong.prog_per_mo
    ong["f_vel_std"] = g.prog_per_mo.transform(lambda s: s.expanding().std())
    ong["f_burn_mean"] = g.exp_per_mo.transform(lambda s: s.expanding().mean())
    # A missing progress change (first snapshot, or a month with no progress
    # figure) is UNKNOWN, not a stall. Treating it as 0 once marked every project
    # stalled in its first month and across the Nov-2024 progress gap.
    ong["is_stall"] = (ong.prog_per_mo <= 0.01).astype(float).where(ong.prog_per_mo.notna())
    ong["f_stall_share"] = g.is_stall.transform(lambda s: s.expanding().mean())
    ong["f_stall_run"] = g.is_stall.transform(
        lambda s: s.groupby((s != s.shift()).cumsum()).cumcount().add(1).where(s == 1, 0))
    ong["f_n_snapshots"] = g.cumcount() + 1

    # ---- Tier D: how often the plan of record has already moved, as-of t
    ong["chg_doc"] = (g.effective_doc.diff().dt.days.abs() > SLIP_DAYS).astype(float)
    ong["chg_cost"] = (g.effective_cost.diff().abs() > 0.01).astype(float)
    ong["chg_adoc"] = (g.doc_anticipated.diff().dt.days.abs() > SLIP_DAYS).astype(float)
    for src, dst in [("chg_doc", "f_n_doc_changes"), ("chg_cost", "f_n_cost_changes"),
                     ("chg_adoc", "f_n_antic_doc_changes")]:
        ong[dst] = ong.groupby("project_code", sort=False)[src].transform(
            lambda s: s.fillna(0).expanding().sum())

    # ---- Tier B: cumulative state at t
    asof = pd.to_datetime(ong.period + "-01")
    ong["f_progress"] = ong.physical_progress_pct
    ong["f_fin_progress"] = 100 * ong.expenditure / ong.effective_cost.replace(0, np.nan)
    ong["f_progress_gap"] = ong.f_fin_progress - ong.f_progress
    ong["f_age_mo"] = (asof - ong.approval_date).dt.days / 30.44
    ong["f_planned_mo"] = (ong.doc_original - ong.approval_date).dt.days / 30.44
    ong["f_elapsed_share"] = ong.f_age_mo / ong.f_planned_mo.replace(0, np.nan)
    ong["f_months_past_doc"] = (asof - ong.doc_original).dt.days / 30.44
    ong["f_overdue"] = (ong.f_months_past_doc > 0).astype(int)
    ong["f_mo_remaining"] = (ong.effective_doc - asof).dt.days / 30.44
    # f_required_vel is undefined (NaN) once the project is past even its current plan
    # (f_mo_remaining <= 0) -- there is no forward window left to divide by. Two ways to
    # handle that were considered:
    #   (a) substitute a fixed "should have finished by now" reference pace (e.g.
    #       remaining progress / months already overdue) so the feature stays numeric
    #       for every row, clearly flagged as a different regime.
    #   (b) leave it NaN here (mathematically honest -- there is no real "pace still
    #       needed" once the target date is in the past) and give score_v2.reasons() a
    #       distinct reason code for this case instead.
    # Went with (b): f_required_vel/f_feasibility feed the model as plain f_* features
    # (train_v2.load_panel() picks up every f_ column), so a synthetic proxy pace would
    # quietly become a model input with no real-world meaning, need retraining to
    # validate, and risk officers reading it as directly comparable to the real
    # remaining-time-based number on other rows. NaN costs nothing model-side (XGBoost
    # treats missing values as a legitimate, learnable signal) and matches this
    # codebase's stance elsewhere (src/newdata.py's CORRUPT set) of declaring a gap
    # rather than inventing a number to fill it. See score_v2.reasons() for the
    # diagnostic this produces instead.
    ong["f_required_vel"] = ((100 - ong.f_progress)
                             / ong.f_mo_remaining.where(ong.f_mo_remaining > 0))
    # feasibility: what the plan demands vs what the project has actually done
    ong["f_feasibility"] = ong.f_vel_mean / ong.f_required_vel.replace(0, np.nan)
    ong["f_is_cost_revised"] = ong.is_cost_revised
    ong["f_is_sched_revised"] = ong.is_schedule_revised
    ong["f_cost_revision_pct"] = 100 * (ong.effective_cost / ong.cost_original - 1)
    ong["f_slip_days"] = (ong.effective_doc - ong.doc_original).dt.days

    # ---- the agency's own live forecast vs its plan of record. This moves BEFORE
    # a formal revision, so it is the earliest honest signal in the file.
    ong["f_antic_cost_gap_pct"] = 100 * (ong.cost_anticipated / ong.effective_cost.replace(0, np.nan) - 1)
    ong["f_antic_doc_gap_days"] = (ong.doc_anticipated - ong.effective_doc).dt.days
    ong["f_log_cost"] = np.log1p(ong.cost_original.clip(lower=0))

    return ong, comp


def build(horizon=HORIZON):
    ong, comp = feature_frame()

    # ---- labels: state at t+horizon vs state at t --------------------------
    key = ong.set_index(["project_code", "mo"])
    fut = ong[["project_code", "mo", "effective_cost", "effective_doc",
               "cost_anticipated", "doc_anticipated"]].copy()
    fut["mo"] = fut.mo - horizon          # join a future snapshot onto the row at t
    fut = fut.rename(columns=lambda c: "fut_" + c if c not in ("project_code", "mo") else c)
    p = ong.merge(fut, on=["project_code", "mo"], how="left")

    cost_up = (p.fut_effective_cost > p.effective_cost * (1 + COST_TOL)) | \
              (p.fut_cost_anticipated > p.cost_anticipated * (1 + COST_TOL))
    time_up = ((p.fut_effective_doc - p.effective_doc).dt.days > SLIP_DAYS) | \
              ((p.fut_doc_anticipated - p.doc_anticipated).dt.days > SLIP_DAYS)
    p["y_cost_esc"] = cost_up.fillna(False).astype(int)
    p["y_time_esc"] = time_up.fillna(False).astype(int)
    p["y_adverse"] = (cost_up.fillna(False) | time_up.fillna(False)).astype(int)
    p["has_future"] = p.fut_effective_cost.notna() | p.fut_effective_doc.notna()

    # A project that left the roster because it COMMISSIONED is a success, not an
    # escalation: label it negative instead of dropping it with the unexplained exits.
    # sort_values("period") here even though newdata.load() already returns comp
    # pre-sorted: keeps this call correct on its own if that contract ever changes.
    done = comp.assign(mo=comp.period.map(_mo))[["project_code", "period", "mo"]] \
               .sort_values("period").drop_duplicates("project_code", keep="first") \
               .drop(columns="period").rename(columns={"mo": "done_mo"})
    p = p.merge(done, on="project_code", how="left")
    completes_in_window = (p.done_mo > p.mo) & (p.done_mo <= p.mo + horizon)
    p.loc[completes_in_window & ~p.has_future, ["y_cost_esc", "y_time_esc", "y_adverse"]] = 0
    p.loc[completes_in_window, "exit_status"] = "completed"
    p["exit_status"] = p.exit_status.fillna(
        pd.Series(np.where(p.has_future, "on_roster", "unexplained_exit"), index=p.index))

    # Only origin months whose t+horizon snapshot actually EXISTS may contribute.
    # Feb-2025 is absent and the last two months run past the data, so those
    # origins would otherwise keep just their completions -- pure negatives with
    # no escalating counterparts, which quietly biases the base rate down.
    obs = set(ong.mo.unique())
    valid_origin = p.mo.isin({m for m in obs if m + horizon in obs})
    print(f"valid origin months: {sorted(p.loc[valid_origin, 'period'].unique())}")
    keep = p[valid_origin & (p.has_future | completes_in_window)].copy()
    fcols = [c for c in keep.columns if c.startswith("f_")]
    keep[fcols] = keep[fcols].replace([np.inf, -np.inf], np.nan)
    cols = (["project_code", "period", "mo", "project_name", "sector", "agency",
             "state", "exit_status", "effective_cost"]
            + fcols + ["y_cost_esc", "y_time_esc", "y_adverse"])
    out = keep[cols]
    OUT.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT / "panel_v2.csv", index=False)

    print(f"panel rows        : {len(out)}   projects: {out.project_code.nunique()}")
    print(f"features (f_*)    : {len(fcols)}")
    print(f"periods           : {sorted(out.period.unique())}")
    print(f"exit_status       : {out.exit_status.value_counts().to_dict()}")
    for c in ["y_cost_esc", "y_time_esc", "y_adverse"]:
        print(f"  {c:12s} positives {out[c].sum():5d}  base rate {out[c].mean():.3f}")
    print(f"rows per period   : {out.groupby('period').size().to_dict()}")
    return out


if __name__ == "__main__":
    build()
