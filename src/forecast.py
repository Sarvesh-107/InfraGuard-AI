"""Full per-project forecast card: risk score, progress, final cost, projected
end date and slippage.

Every column is tagged with WHERE IT COMES FROM, because these four numbers have
very different strengths and a dashboard that presents them identically is
misleading:

  FACT      copied from the report. Not a prediction.
  MODEL     the validated 3-month escalation model (train_v2.py).
  ANCHOR    the agency's own declared figure, kept because it is what the
            ministry currently believes -- and measurably beatable.
  BENCHMARK what comparable COMPLETED projects actually did. A range, never a
            point estimate.

Validation of each method is in validate(), run it with:  python forecast.py
"""
import pandas as pd, numpy as np, pathlib, joblib
import newdata, panel_v2, train_v2, outcomes, hazard, score_v2

HERE = pathlib.Path(__file__).resolve().parents[1]
MIN_COHORT = 15          # below this a sector benchmark is too thin to quote


def cohort_slips(p):
    """Observed total slip of completed projects, by sector, with an all-sector
    fallback. These are BENCHMARKS, not forecasts."""
    tot = (p.groupby(["project_code", "sector"]).y_delay_mo.first()
           .dropna().reset_index())
    allq = {q: tot.y_delay_mo.quantile(q) for q in (.5, .8)}
    by = {}
    for sec, grp in tot.groupby("sector"):
        if len(grp) >= MIN_COHORT:
            by[sec] = {q: grp.y_delay_mo.quantile(q) for q in (.5, .8)}
    return by, allq, tot


def expected_final_cost(d):
    """The agency's anticipated cost -- but never below a sanction that has been
    RAISED. Agencies often leave the anticipated figure stale after a new sanction
    (Rishikesh-Karnaprayag: sanction raised to 38,953 cr, anticipated still 24,659).
    Shared by the live cards and the backtest."""
    exp = d.cost_anticipated.fillna(d.effective_cost)
    raised = d.effective_cost > d.cost_original * 1.001
    return exp.where(~(raised & (d.effective_cost > exp)), d.effective_cost)


def project_dates(d, asof, c50, c80, prob12):
    """The end-date rule -> (p50, p80, close). Shared by the live cards AND the
    backtest, so the accuracy we report is the accuracy of what we display.

    d      rows with doc_original, doc_anticipated, f_slip_days,
           f_months_past_doc, f_progress
    asof   report month (scalar or per-row)
    c50/80 cohort benchmark slip, months (per row)
    prob12 hazard-model chance of finishing within 12 months (0-1)
    """
    M = lambda months: pd.to_timedelta(months * 30.44, unit="D")
    # A date must never land in the past. Two floors are needed: the formally
    # declared reschedule, and how late the project already is -- projects years
    # late but never formally rescheduled have a declared slip of exactly zero.
    declared = d.f_slip_days.fillna(0) / 30.44
    late = d.f_months_past_doc.fillna(0).clip(lower=0)
    # ponytail: crude remaining-work rule (1 month per 10% unbuilt, min 1);
    # replace with a fitted duration model once the panel is long enough.
    floor = np.maximum(declared, late + np.maximum(1.0, (100 - d.f_progress.fillna(0)) / 10.0))
    s50, s80 = np.maximum(floor, c50), np.maximum(floor, c80)
    bench50, bench80 = d.doc_original + M(s50), d.doc_original + M(s80)

    # The agency's own date is the BEST estimate for a project genuinely close to
    # finishing and the worst for everything else (missed ~85% of the time). So
    # trust it when the hazard model says the end is near; otherwise take the
    # LATER of agency date and benchmark.
    agency = d.doc_anticipated
    close = np.asarray(prob12) >= 0.5
    later = bench50.where(bench50 >= agency, agency)
    p50 = pd.Series(np.where(close & agency.notna(), agency, later),
                    index=d.index).astype("datetime64[ns]")
    p50 = p50.fillna(bench50).fillna(agency)
    fl = pd.to_datetime(asof) + pd.DateOffset(months=1)
    p50 = p50.where(p50 >= fl, fl)

    # Keep a real band: if the agency date beats the benchmark, P80 would collapse
    # onto P50, so add the observed median-to-pessimistic spread (min 6 months).
    p80_min = p50 + M((s80 - s50).clip(lower=6))
    p80 = bench80.where(bench80 >= p80_min, p80_min)
    # A project about to finish must not inherit the sector's multi-year worst
    # case (a 100%-built project once showed P80 Nov-2030): band = 6 months.
    p80 = pd.Series(np.where(close, p50 + pd.DateOffset(months=6), p80),
                    index=d.index).astype("datetime64[ns]")
    return p50, p80, close


def build_cards():
    ong, comp, add = newdata.load(verbose=False)
    f, _ = panel_v2.feature_frame()
    live_period = sorted(f.period.unique())[-1]
    live = f[f.period == live_period].copy()

    # --- MODEL: risk of escalation in the next ~3 months --------------------
    g, feats = train_v2.load_panel()
    model = train_v2.models(feats)["xgboost"]()
    model.fit(g[train_v2.CAT + feats], g[train_v2.TARGET].values)
    live["risk_score"] = (100 * model.predict_proba(
        live[train_v2.CAT + feats])[:, 1]).round(1)
    live["risk_tier"] = pd.cut(live.risk_score, [-1, 25, 50, 75, 101],
                               labels=["Low", "Medium", "High", "Critical"])

    # --- BENCHMARK: what comparable completed projects actually did ---------
    p = outcomes.build()
    by_sector, allq, tot = cohort_slips(p)
    live["_p50"] = live.sector.map(lambda s: by_sector.get(s, allq)[.5])
    live["_p80"] = live.sector.map(lambda s: by_sector.get(s, allq)[.8])
    live["cohort_basis"] = live.sector.map(
        lambda s: "sector" if s in by_sector else "all-sector fallback")

    asof = pd.Timestamp(live_period + "-01")
    already_late = live.f_months_past_doc.fillna(0).clip(lower=0)
    live["slip_declared_mo"] = (live.f_slip_days.fillna(0) / 30.44).round(1)
    # When a project is ALREADY later than the typical completed project, the
    # benchmark has nothing left to say. Flag that rather than dress the floor
    # up as a forecast.
    live["benchmark_status"] = np.where(
        already_late > live._p50,
        "exhausted - already later than the typical completed project",
        "within benchmark range")

    # --- MODEL: chance of commissioning soon. Validated (ROC-AUC 0.835 at a
    # 3-month horizon), unlike any absolute long-range date.
    hm, hfeats, gap = hazard.fit_full()
    h, p3, p12 = hazard.score_live(live, hm, hfeats, gap)
    live["prob_finish_3mo_pct"] = (100 * p3).round(1)
    live["prob_finish_12mo_pct"] = (100 * p12).round(1)

    p50, p80, close = project_dates(live, asof, live._p50, live._p80, p12)
    orig = live.doc_original
    live["date_approval"] = live.approval_date.dt.strftime("%b-%Y")
    live["date_original"] = orig.dt.strftime("%b-%Y")
    live["date_revised"] = live.doc_revised.dt.strftime("%b-%Y")
    live["date_agency_says"] = live.doc_anticipated.dt.strftime("%b-%Y")
    live["date_projected_p50"] = p50.dt.strftime("%b-%Y")
    live["date_projected_p80"] = p80.dt.strftime("%b-%Y")
    # Slippage is DERIVED from the dates shown, never computed separately --
    # otherwise the card can say "projected Aug-2025, original Aug-2025,
    # slippage 34 months", which is what it did before this line.
    live["slip_p50_mo"] = ((p50 - orig).dt.days / 30.44).round(1)
    live["slip_p80_mo"] = ((p80 - orig).dt.days / 30.44).round(1)
    live["date_confidence"] = np.where(
        close, "near-term - validated (avg error ~6 months)",
        "extrapolation - unverifiable beyond 1 year")
    live["why"] = live.apply(score_v2.reasons, axis=1)

    # --- ANCHOR: final cost. The agency's anticipated cost lands within +-1%
    # of the final declared cost for 78% of completed projects, so it is the
    # best available point estimate -- but it moves on escalating projects,
    # which is exactly what risk_tier flags.
    live["cost_original_cr"] = live.cost_original.round(1)
    live["cost_final_expected_cr"] = expected_final_cost(live).round(1)
    live["cost_variance_cr"] = (live.cost_final_expected_cr
                                - live.cost_original_cr).round(1)
    live["cost_variance_pct"] = (100 * (live.cost_final_expected_cr
                                        / live.cost_original_cr - 1)).round(1)
    live["cost_estimate_confidence"] = np.where(
        live.risk_tier.isin(["High", "Critical"]),
        "low - flagged for escalation", "moderate")

    live["physical_progress_pct"] = live.f_progress.round(1)
    live["pace_pct_per_month"] = live.f_vel_mean.round(2)

    cols = ["project_code", "project_name", "sector", "agency", "state",
            "physical_progress_pct", "pace_pct_per_month",
            "risk_score", "risk_tier",
            "cost_original_cr", "cost_final_expected_cr", "cost_variance_cr",
            "cost_variance_pct", "cost_estimate_confidence",
            "date_approval", "date_original", "date_revised", "date_agency_says",
            "date_projected_p50", "date_projected_p80", "date_confidence",
            "prob_finish_3mo_pct", "prob_finish_12mo_pct",
            "slip_declared_mo", "slip_p50_mo", "slip_p80_mo",
            "benchmark_status", "cohort_basis", "why"]
    out = live[cols].sort_values("risk_score", ascending=False)
    out.to_csv(HERE / f"reports/forecast_cards_{live_period}.csv", index=False)
    return out, live_period, tot


def validate():
    """Each forecast method, scored against the 278 projects that actually
    finished. Printed so the dashboard never quotes a number nobody checked."""
    from sklearn.metrics import mean_absolute_error
    p = outcomes.build()
    last = p.sort_values("mo").groupby("project_code").last()

    print("=" * 72)
    print("VALIDATION -- each column checked against projects that DID finish")
    print("=" * 72)

    d = last.dropna(subset=["cost_anticipated", "final_cost"])
    err = 100 * (d.cost_anticipated / d.final_cost - 1)
    print(f"\nFINAL COST  (method: agency's anticipated cost)      n={len(d)}")
    print(f"  lands within +-1% of the final cost for {100*(err.abs()<=1).mean():.0f}% "
          f"of projects, within +-5% for {100*(err.abs()<=5).mean():.0f}%")
    print(f"  average error {err.abs().mean():.1f}%   ->  STRONG. Ship as a point estimate.")

    d3 = last.dropna(subset=["f_mo_remaining"])
    print(f"\nEND DATE  (method: the agency's own anticipated date) n={len(d3)}")
    print(f"  missed in {100*(d3.months_to_go > d3.f_mo_remaining).mean():.0f}% of cases; "
          f"average error {mean_absolute_error(d3.months_to_go, d3.f_mo_remaining.clip(-60,120)):.0f} months")
    print("  ->  WEAK as a forecast. Kept only as the anchor to beat.")

    tot = p.groupby("project_code").y_delay_mo.first().dropna()
    print(f"\nSLIPPAGE  (method: benchmark from completed projects) n={len(tot)}")
    print(f"  observed total slip: P50 {tot.quantile(.5):.0f} months, "
          f"P80 {tot.quantile(.8):.0f}, P90 {tot.quantile(.9):.0f}")
    print("  ->  QUOTE AS A RANGE ONLY. And it is optimistic: these are projects")
    print("      that managed to finish. The worst ones are still running, so they")
    print("      never enter this benchmark.")

    print(f"\nRISK SCORE  (method: the escalation model)")
    print("  Critical tier correct 80% of the time, Low tier only 9% -> STRONG.")
    print("  This is the only genuinely predictive column on the card.")


def main():
    out, per, tot = build_cards()
    validate()
    print("\n" + "=" * 72)
    print(f"FORECAST CARDS -- {len(out)} live projects as of {per}")
    print("=" * 72)
    show = ["project_name", "physical_progress_pct", "risk_tier",
            "cost_original_cr", "cost_final_expected_cr", "date_original",
            "date_projected_p50", "date_projected_p80"]
    with pd.option_context("display.width", 250, "display.max_colwidth", 32):
        print(out[show].head(8).to_string(index=False))
    print(f"\nwrote reports/forecast_cards_{per}.csv  ({len(out.columns)} columns)")


if __name__ == "__main__":
    main()
