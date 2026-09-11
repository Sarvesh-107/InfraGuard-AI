"""Fit on the full panel, score the LIVE roster, emit the watchlist.

Score = 100 x P(cost or schedule escalation declared within ~3 months).
XGBoost is used because it wins protocol A and is the best calibrated
(Brier 0.138); the score is its probability alone, with no weight blend on top.
"""
import pandas as pd, numpy as np, pathlib, joblib
from xgboost import XGBClassifier
import panel_v2, train_v2

HERE = pathlib.Path(__file__).resolve().parents[1]


def reasons(r):
    """Fixed reason-code vocabulary: every flagged project says WHY in terms an
    officer can check against the file."""
    o = []
    if r.f_antic_doc_gap_days > 60:
        o.append(f"agency's own forecast is {r.f_antic_doc_gap_days/30.44:.0f} months "
                 f"past its approved date")
    if r.f_antic_cost_gap_pct > 2:
        o.append(f"agency anticipates +{r.f_antic_cost_gap_pct:.0f}% over approved cost")
    if r.f_months_past_doc > 24:
        o.append(f"{r.f_months_past_doc/12:.1f}y past original commissioning date")
    elif r.f_overdue == 1:
        o.append(f"{r.f_months_past_doc:.0f} months past original date")
    if r.f_stall_run >= 3:
        o.append(f"no physical progress for {r.f_stall_run:.0f} consecutive reports")
    if pd.notna(r.f_feasibility) and 0 <= r.f_feasibility < 0.5:
        o.append(f"pace is {r.f_feasibility:.0%} of what the current date needs")
    elif pd.notna(r.f_mo_remaining) and r.f_mo_remaining <= 0:
        # f_required_vel/f_feasibility are NaN here by design (see panel_v2.py) --
        # there's no forward-looking pace target once the current plan date has
        # already passed, so say that plainly instead of staying silent on it.
        o.append("already past planned completion, no revised pace available")
    if r.f_progress_gap > 15:
        o.append(f"spend outruns works by {r.f_progress_gap:.0f}pp")
    if r.f_n_doc_changes >= 2:
        o.append(f"date already moved {r.f_n_doc_changes:.0f}x in 10 months")
    return "; ".join(o[:3]) or "elevated risk on combined profile"


def main():
    g, feats = train_v2.load_panel()
    X, y = g[train_v2.CAT + feats], g[train_v2.TARGET].values
    model = train_v2.models(feats)["xgboost"]()
    model.fit(X, y)
    joblib.dump(model, HERE / "models/escalation_xgb_v2.joblib")

    ong, comp = panel_v2.feature_frame()
    live_period = sorted(ong.period.unique())[-1]
    live = ong[ong.period == live_period].copy()
    live["risk_score"] = (100 * model.predict_proba(live[train_v2.CAT + feats])[:, 1]).round(1)
    live["tier"] = pd.cut(live.risk_score, [-1, 25, 50, 75, 101],
                          labels=["Low", "Medium", "High", "Critical"])
    live["why"] = live.apply(reasons, axis=1)

    cols = ["project_code", "project_name", "sector", "agency", "state",
            "effective_cost", "f_progress", "risk_score", "tier", "why"]
    out = (live[cols].rename(columns={"effective_cost": "cost_cr",
                                      "f_progress": "physical_progress_pct"})
           .sort_values("risk_score", ascending=False))
    out.to_csv(HERE / f"reports/watchlist_{live_period}.csv", index=False)

    print(f"live roster {live_period}: {len(out)} projects\n")
    print(out.tier.value_counts().reindex(["Critical", "High", "Medium", "Low"]).to_string())
    hi = out[out.tier.isin(["Critical", "High"])]
    print(f"\nRs at risk (High+Critical): Rs {hi.cost_cr.sum():,.0f} cr across {len(hi)} projects")
    print(f"\nTop 10:")
    for _, r in out.head(10).iterrows():
        print(f"  {r.risk_score:5.1f}  {str(r.project_name)[:56]:56s} {str(r.sector)[:20]:20s}")
        print(f"         -> {r.why}")
    print(f"\nwrote reports/watchlist_{live_period}.csv")


if __name__ == "__main__":
    main()
