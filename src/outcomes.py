"""PS part (a): can we predict REALISED cost / time overrun from completions?

Verdict: NO -- not from a 10-month window. This script exists to prove that with
numbers rather than assert it, because the naive versions of both models produce
excellent-looking scores that are artifacts.

Run it to reproduce the three findings below.

  1. FINAL DELAY is un-validatable here. The label decomposes into a feature plus
     a window-truncated remainder:
         y_delay = f_months_past_doc  +  months_to_go
                   \\_ already a feature   \\_ capped at 11 by a 10-month panel
     So the model only ever guesses a number in 1..11 and scores brilliantly.
     A project that will really commission in 2030 needs months_to_go = 60, which
     the model has never seen and cannot output. A trivial "delay so far + median"
     baseline lands within ~1 month of the tuned Random Forest.

  2. COST OVERRUN vs original is very nearly TAUTOLOGICAL. Whether the final cost
     exceeds the original sanction is all but determined by whether the cost has
     already been revised at t -- which is an input feature.

  3. COST OVERRUN beyond what is already declared is honest but far too thin:
     16 positive projects out of 278.

What IS defensible from this data needs no model at all -- see headline_finding().
The model that does work, on a target this window can actually support, is the
3-month escalation model in train_v2.py.
"""
import pandas as pd, numpy as np, pathlib, json, warnings
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import newdata, panel_v2

warnings.filterwarnings("ignore")
HERE = pathlib.Path(__file__).resolve().parents[1]
CAT = ["sector", "state"]


def build():
    """One row per (completed project, as-of snapshot before it commissioned)."""
    ong, comp, add = newdata.load(verbose=False)
    f, _ = panel_v2.feature_frame()
    # sort_values("period") here even though newdata.load() already returns comp
    # pre-sorted: keeps this call correct on its own if that contract ever changes.
    c = comp.sort_values("period").drop_duplicates("project_code", keep="first").copy()
    c["done_mo"] = c.period.map(panel_v2._mo)
    c["done_date"] = pd.to_datetime(c.period + "-01")

    last = f.sort_values("mo").groupby("project_code").last()
    c = c.merge(last[["effective_cost"]], left_on="project_code",
                right_index=True, how="left")
    # Realised cost = last DECLARED effective cost, never cumulative expenditure:
    # expenditure at the completion report sits ~18% BELOW sanction for most
    # projects, so an expenditure-based ratio measures reporting lag, not overrun.
    c["y_cost_pct"] = 100 * (c.effective_cost / c.cost_original - 1)
    c["y_cost_up"] = (c.y_cost_pct > 0.5).astype(float)
    c["y_delay_mo"] = (c.done_date - c.doc_original).dt.days / 30.44

    p = f.merge(c[["project_code", "done_mo", "y_delay_mo", "y_cost_pct",
                   "y_cost_up", "effective_cost"]].rename(
                       columns={"effective_cost": "final_cost"}),
                on="project_code")
    p = p[p.mo < p.done_mo].copy()
    p["months_to_go"] = p.done_mo - p.mo
    return p


def _pipe(clf, feats, scale=False):
    num = [("imp", SimpleImputer(strategy="median"))]
    if scale:
        num.append(("sc", StandardScaler()))
    return Pipeline([("prep", ColumnTransformer([
        ("c", OneHotEncoder(handle_unknown="infrequent_if_exist",
                            min_frequency=25, sparse_output=False), CAT),
        ("n", Pipeline(num), feats)])), ("clf", clf)])


def diagnose_delay(p, feats):
    d = p.dropna(subset=["y_delay_mo"])
    X, y, gp = d[CAT + feats], d.y_delay_mo.values, d.project_code
    print("=" * 72)
    print("1. FINAL DELAY -- why the good-looking score is an artifact")
    print("=" * 72)
    print(f"rows={len(d)}  projects={d.project_code.nunique()}  "
          f"median delay={np.median(y):.0f} months")

    ident = d.f_months_past_doc.fillna(0) + d.months_to_go
    print(f"\n  y_delay == f_months_past_doc + months_to_go ?  "
          f"mean abs difference = {(d.y_delay_mo - ident).abs().mean():.2f} months")
    print(f"  months_to_go: max={d.months_to_go.max():.0f}, mean={d.months_to_go.mean():.1f}"
          f"   <- capped by the 10-month panel, NOT by the projects")

    triv = (d.f_months_past_doc.fillna(0) + d.months_to_go.median()).values
    rf = np.zeros(len(y))
    for tr, te in GroupKFold(n_splits=5).split(X, y, gp):
        m = _pipe(RandomForestRegressor(n_estimators=400, min_samples_leaf=5,
                                        random_state=0, n_jobs=-1), feats)
        m.fit(X.iloc[tr], y[tr]); rf[te] = m.predict(X.iloc[te])

    rows = {
        "baseline: predict 0": np.zeros(len(y)),
        "baseline: global median": np.full(len(y), np.median(y)),
        "TRIVIAL: delay so far + median remainder": triv,
        "tuned random forest": rf,
    }
    print(f"\n  {'model':44s} {'MAE(mo)':>9s} {'R2':>7s}")
    print("  " + "-" * 62)
    out = {}
    for k, v in rows.items():
        out[k] = {"mae": float(mean_absolute_error(y, v)), "r2": float(r2_score(y, v))}
        print(f"  {k:44s} {out[k]['mae']:9.1f} {out[k]['r2']:7.3f}")
    gap = out["TRIVIAL: delay so far + median remainder"]["mae"] - out["tuned random forest"]["mae"]
    print(f"\n  => the tuned model buys {gap:.1f} months over arithmetic.")
    print("     VERDICT: do not ship a final-delay model on a 10-month window.")
    return out


def diagnose_cost(p):
    print("\n" + "=" * 72)
    print("2. COST OVERRUN vs original sanction -- near-tautological")
    print("=" * 72)
    c = p.dropna(subset=["y_cost_up"])
    print(f"  P(final > original | cost ALREADY revised at t) = "
          f"{c[c.f_is_cost_revised == 1].y_cost_up.mean():.3f}")
    print(f"  P(final > original | NOT yet revised at t)      = "
          f"{c[c.f_is_cost_revised == 0].y_cost_up.mean():.3f}")
    pos = int(c.groupby('project_code').y_cost_up.max().sum())
    print(f"  {int(c[c.f_is_cost_revised == 1].y_cost_up.sum())} of "
          f"{int(c.y_cost_up.sum())} positive rows come from already-revised projects.")
    print("  => a model here mostly re-reads its own input feature.")

    print("\n" + "=" * 72)
    print("3. FURTHER cost rise beyond what is declared at t -- honest but thin")
    print("=" * 72)
    p = p.copy()
    p["y_further"] = (p.final_cost > p.effective_cost * 1.001).astype(float)
    npos = int(p.groupby("project_code").y_further.max().sum())
    print(f"  positive projects: {npos} of {p.project_code.nunique()}  "
          f"({100 * p.y_further.mean():.1f}% of rows)")
    print(f"  P(further | already revised) = {p[p.f_is_cost_revised == 1].y_further.mean():.3f}"
          f"   P(further | not revised) = {p[p.f_is_cost_revised == 0].y_further.mean():.3f}"
          f"   <- tautology gone")
    print(f"  => only {npos} positive projects. Too thin to model honestly.")
    return {"further_positive_projects": npos,
            "total_projects": int(p.project_code.nunique())}


def headline_finding(p):
    """The defensible number from the completions -- and it needs no model."""
    print("\n" + "=" * 72)
    print("4. WHAT THIS DATA DOES SUPPORT (no model required)")
    print("=" * 72)
    p = p.copy()
    p["missed"] = (p.months_to_go > p.f_mo_remaining.fillna(0)).astype(float)
    per_proj = p.groupby("project_code").missed.max()
    n, tot = int(per_proj.sum()), len(per_proj)
    print(f"  Completed projects that commissioned LATER than the date they were")
    print(f"  declaring while still in flight: {n} of {tot} ({100*n/tot:.0f}%).")
    last = p.sort_values("mo").groupby("project_code").last()
    print(f"  Median delay at commissioning: "
          f"{p.groupby('project_code').y_delay_mo.first().median():.0f} months.")
    print("  => 'the declared completion date is missed ~9 times in 10' is a")
    print("     defensible, model-free headline. The forward-looking model that")
    print("     DOES validate on this window is train_v2.py (3-month escalation).")
    return {"missed_declared_date": n, "of_projects": tot}


def main():
    p = build()
    feats = [c for c in p.columns if c.startswith("f_")]
    r = {"delay_diagnosis": diagnose_delay(p, feats),
         "cost_diagnosis": diagnose_cost(p),
         "headline": headline_finding(p)}
    json.dump(r, open(HERE / "reports/outcome_models.json", "w"), indent=2, default=float)
    print("\nwrote reports/outcome_models.json")


if __name__ == "__main__":
    main()
