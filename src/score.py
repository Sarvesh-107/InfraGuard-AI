"""Fit the chosen model on the full panel and score the LIVE roster.

Score = 100 x P(cost or schedule escalation declared within ~3 months).
It is the model's calibrated probability, nothing else -- no hand-tuned weight
blend on top, because a weight nobody validated is a number nobody can defend.
"""
import pandas as pd, numpy as np, pathlib, joblib
from sklearn.ensemble import RandomForestClassifier
from panel import load_silver, features
from train import make_pipe, CAT, NUM, TARGET

HERE = pathlib.Path(__file__).resolve().parents[1]
LIVE = "2026-07"


def reasons(r):
    """Fixed reason-code vocabulary -- every flagged project must say WHY in
    words an officer can check against the file."""
    out = []
    if r.months_past_original_doc > 24:
        out.append(f"{r.months_past_original_doc/12:.1f}y past original commissioning date")
    elif r.already_overdue:
        out.append(f"{r.months_past_original_doc:.0f} months past original date")
    if r.progress_gap_pct is not None and r.progress_gap_pct > 15:
        out.append(f"spend outruns works by {r.progress_gap_pct:.0f}pp")
    if r.is_schedule_revised and r.declared_slip_days > 730:
        out.append(f"already rescheduled, {r.declared_slip_days/365:.1f}y slip declared")
    if r.is_cost_revised and r.declared_cost_overrun_pct > 10:
        out.append(f"cost already revised +{r.declared_cost_overrun_pct:.0f}%")
    if r.physical_progress_pct < 25 and r.elapsed_share > 1:
        out.append(f"only {r.physical_progress_pct:.0f}% built with planned time exhausted")
    if r.required_velocity is not None and r.required_velocity > 5:
        out.append(f"needs {r.required_velocity:.1f}%/month to hit current date")
    return "; ".join(out[:3]) or "elevated risk on combined profile"


def main():
    panel = pd.read_csv(HERE / "data/gold/panel_t0.csv", index_col="project_code")
    model = make_pipe(RandomForestClassifier(n_estimators=400, min_samples_leaf=8,
                                             class_weight="balanced",
                                             random_state=0, n_jobs=-1))
    model.fit(panel[CAT + NUM], panel[TARGET].values)
    joblib.dump(model, HERE / "models/escalation_rf.joblib")

    d, ong, ne, new, comp = load_silver()
    live = ong[ong.period == LIVE].set_index("project_code")
    X = features(live, ne, new, LIVE)
    p = model.predict_proba(X[CAT + NUM])[:, 1]

    w = X.copy()
    w["risk_score"] = (100 * p).round(1)
    w["tier"] = pd.cut(w.risk_score, [-1, 25, 50, 75, 101],
                       labels=["Low", "Medium", "High", "Critical"])
    w["cost_cr"] = pd.to_numeric(live.effective_cost_cr, errors="coerce")
    w["why"] = w.apply(reasons, axis=1)

    cols = ["project_name", "sector", "agency", "state", "cost_cr",
            "physical_progress_pct", "risk_score", "tier", "why"]
    out = w.sort_values("risk_score", ascending=False)[cols]
    out.to_csv(HERE / "reports/watchlist_2026_07.csv")

    print(f"scored {len(out)} live projects (roster {LIVE})\n")
    print(out.tier.value_counts().reindex(["Critical", "High", "Medium", "Low"]).to_string())
    crit = w[w.tier.isin(["Critical", "High"])]
    print(f"\nRs at risk (High+Critical): Rs {crit.cost_cr.sum():,.0f} cr "
          f"across {len(crit)} projects")
    print("\nTop 10 watchlist:")
    for c, r in out.head(10).iterrows():
        print(f"  {r.risk_score:5.1f} {str(r.project_name)[:52]:52s} {str(r.sector)[:18]:18s}")
        print(f"        -> {r.why}")
    print(f"\nwrote reports/watchlist_2026_07.csv + models/escalation_rf.joblib")


if __name__ == "__main__":
    main()
