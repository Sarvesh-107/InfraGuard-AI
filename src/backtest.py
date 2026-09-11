"""Accuracy of the forecast card, measured on projects that were ongoing for some
months and then FINISHED -- predicting each one from every month before it did.

Uses forecast.project_dates, the exact rule the dashboard displays, so the
accuracy reported is the accuracy of what users see. Writes:
    reports/backtest_enddate.csv   one row per (finished project, month predicted)
    reports/backtest_cost.csv      one row per finished project
Needs reports/hazard_estimates.csv (python hazard.py) for out-of-fold chances.
"""
import pandas as pd, numpy as np, pathlib
import outcomes, forecast

HERE = pathlib.Path(__file__).resolve().parents[1]


def _month_start(mo):
    return pd.to_datetime(pd.DataFrame({"year": (mo - 1) // 12,
                                        "month": (mo - 1) % 12 + 1, "day": 1}))


def main():
    p = outcomes.build()
    last = p.sort_values("mo").groupby("project_code").last()
    cost = last.dropna(subset=["cost_anticipated", "final_cost"])
    cost = pd.DataFrame({"err_pct": 100 * (forecast.expected_final_cost(cost) / cost.final_cost - 1)})
    cost.to_csv(HERE / "reports/backtest_cost.csv")

    haz = pd.read_csv(HERE / "reports/hazard_estimates.csv")   # out-of-fold
    p = (p.merge(haz[["project_code", "period", "h"]], on=["project_code", "period"], how="left")
          .dropna(subset=["doc_original", "y_delay_mo"]).reset_index(drop=True))

    # leave-one-out benchmark: a project never benchmarks against itself
    tot = p.groupby(["project_code", "sector"]).y_delay_mo.first().reset_index()
    q = {}
    for code, sec in zip(tot.project_code, tot.sector):
        peers = tot[tot.project_code != code]
        s = peers[peers.sector == sec]
        u = s if len(s) >= forecast.MIN_COHORT else peers
        q[code] = (u.y_delay_mo.quantile(.5), u.y_delay_mo.quantile(.8))
    c50 = p.project_code.map(lambda c: q[c][0])
    c80 = p.project_code.map(lambda c: q[c][1])

    p50, p80, close = forecast.project_dates(
        p, pd.to_datetime(p.period + "-01"), c50, c80, 1 - (1 - p.h) ** 12)
    actual = _month_start(p.done_mo)
    mo = lambda t: (t - actual).dt.days / 30.44
    out = pd.DataFrame({
        "project_code": p.project_code, "period": p.period,
        "months_to_go": p.months_to_go, "reliable": close,
        "err_ours": mo(p50), "err_agency": mo(p.doc_anticipated),
        "err_original": mo(p.doc_original), "covered_by_worst_case": actual <= p80})
    out.to_csv(HERE / "reports/backtest_enddate.csv", index=False)

    def line(name, e):
        e = e.dropna()
        print(f"  {name:22s} within 6mo {100*(e.abs()<=6).mean():3.0f}%  "
              f"12mo {100*(e.abs()<=12).mean():3.0f}%  typical miss {e.abs().median():4.1f}mo  "
              f"too optimistic {100*(e < -1).mean():3.0f}%")
    print(f"END DATE -- {out.project_code.nunique()} finished projects, {len(out)} predictions")
    line("our prediction", out.err_ours)
    line("agency's own date", out.err_agency)
    line("original date", out.err_original)
    print(f"  finished on/before our worst case: {100*out.covered_by_worst_case.mean():.0f}%")
    line("  'reliable' route", out.err_ours[out.reliable])
    line("  'rough' route", out.err_ours[~out.reliable])
    e = cost.err_pct.abs()
    print(f"FINAL COST -- {len(cost)} finished projects: within 1% {100*(e<=1).mean():.0f}%, "
          f"within 10% {100*(e<=10).mean():.0f}%")


if __name__ == "__main__":
    main()
