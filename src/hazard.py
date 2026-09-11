"""Projected completion date via a discrete-time hazard model.

Why not just regress "months until completion":
    Only projects that finished INSIDE our 10-month window have an observed
    remaining life, and it is 11 months at most. A regression fitted on those
    can never output "this finishes in 4 years" -- it has never seen one.

What this does instead:
    For every (project, month) we ask a much smaller question, one the data can
    actually answer: *did this project finish THIS month?* Roughly 1.6% did. A
    project still running is not a missing outcome -- it is direct evidence that
    it did NOT finish that month. That is how the ~1,900 unfinished projects
    become usable training data instead of being thrown away.

    The model learns a monthly completion chance h from the project's state.
    A survival curve follows: S(k) = prod(1 - h) over k months, and the
    projected date is where S crosses 50% (and 20% for the pessimistic band).

    Because a low monthly chance compounds into a long wait, this CAN return
    "another 5 years" even though no training example ran that long. That is
    the whole point -- the horizon comes from arithmetic on the hazard, not
    from having observed it.
"""
import pandas as pd, numpy as np, pathlib, warnings
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score
import newdata, panel_v2, train_v2

warnings.filterwarnings("ignore")
HERE = pathlib.Path(__file__).resolve().parents[1]
MAX_MONTHS = 240          # 20 years: a cap for display, not a belief
CAT = train_v2.CAT


def build_hazard_panel():
    """One row per (project, observed month). y = 1 if it commissioned in the
    interval ending at the next observed snapshot."""
    ong, comp, add = newdata.load(verbose=False)
    f, _ = panel_v2.feature_frame()
    periods = sorted(f.mo.unique())
    nxt = {m: n for m, n in zip(periods, periods[1:])}

    done = (comp.drop_duplicates("project_code")
            .assign(done_mo=lambda d: d.period.map(panel_v2._mo))
            .set_index("project_code").done_mo)
    f = f.copy()
    f["done_mo"] = f.project_code.map(done)
    f["next_mo"] = f.mo.map(nxt)
    # the final snapshot has no following observation, so it can be neither a
    # confirmed completion nor a confirmed survival -> it is censored, drop it
    f = f[f.next_mo.notna()].copy()
    # completed in the interval (mo, next_mo]
    f["y"] = ((f.done_mo > f.mo) & (f.done_mo <= f.next_mo)).fillna(False).astype(int)
    # a project already finished before this snapshot is no longer at risk
    f = f[~(f.done_mo <= f.mo).fillna(False)]
    f["gap"] = f.next_mo - f.mo
    return f


def fit_and_eval(fp, feats):
    from xgboost import XGBClassifier
    X, y, gp = fp[CAT + feats], fp.y.values, fp.project_code
    oof = np.zeros(len(y))
    for tr, te in GroupKFold(n_splits=5).split(X, y, gp):
        m = train_v2.make_pipe(XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8,
            colsample_bytree=0.8, min_child_weight=15, reg_lambda=2.0,
            eval_metric="logloss", random_state=0), feats)
        m.fit(X.iloc[tr], y[tr])
        oof[te] = m.predict_proba(X.iloc[te])[:, 1]
    print(f"  monthly completion rate (base) : {y.mean():.3%}")
    print(f"  ranking quality  ROC-AUC {roc_auc_score(y, oof):.3f}   "
          f"PR-AUC {average_precision_score(y, oof):.3f}")
    return oof


def hazard_to_months(h, q):
    """Months until the chance of still being unfinished falls to q."""
    h = np.clip(h, 1e-4, 0.95)
    return np.clip(np.log(q) / np.log(1 - h), 1, MAX_MONTHS)


def main():
    fp = build_hazard_panel()
    feats = [c for c in fp.columns if c.startswith("f_")]
    print("=" * 70)
    print("DISCRETE-TIME HAZARD MODEL -- 'did it finish this month?'")
    print("=" * 70)
    print(f"  rows {len(fp)}  projects {fp.project_code.nunique()}  "
          f"completions {fp.y.sum()}")
    oof = fit_and_eval(fp, feats)

    # per-month hazard, corrected for uneven gaps between reports
    fp = fp.assign(h_raw=oof)
    fp["h"] = 1 - (1 - fp.h_raw) ** (1 / fp.gap.clip(lower=1))
    fp["p50_mo"] = hazard_to_months(fp.h.values, 0.5)
    fp["p80_mo"] = hazard_to_months(fp.h.values, 0.2)

    print(f"\n  implied remaining life (out-of-fold, all projects):")
    for lbl, col in [("median (P50)", "p50_mo"), ("pessimistic (P80)", "p80_mo")]:
        s = fp[col]
        print(f"    {lbl:20s} median {s.median():5.0f} mo   "
              f"25th {s.quantile(.25):4.0f}   75th {s.quantile(.75):5.0f}   "
              f"at cap {100*(s >= MAX_MONTHS).mean():.0f}%")

    print(f"\n  Sanity: does a HIGHER predicted hazard really finish sooner?")
    d = fp[fp.done_mo.notna()].copy()
    d["actual_remaining"] = d.done_mo - d.mo
    q = pd.qcut(d.h, 4, labels=["slowest 25%", "slow", "fast", "fastest 25%"])
    print(f"    {'group':>14s} {'n':>5s} {'actual months to finish':>25s}")
    for g in q.cat.categories:
        m = (q == g).values
        print(f"    {g:>14s} {m.sum():5d} {d.actual_remaining[m].median():>18.1f} (median)")

    print(f"\n  NOTE: that check can only use projects that DID finish, so all four")
    print(f"  groups look fast. The ranking is the meaningful part, not the level.")
    fp[["project_code", "period", "h", "p50_mo", "p80_mo"]].to_csv(
        HERE / "reports/hazard_estimates.csv", index=False)
    print(f"\nwrote reports/hazard_estimates.csv")
    return fp


if __name__ == "__main__":
    main()


def fit_full():
    """Fit on every (project, month) row and return the model, for scoring the
    live roster (whose own rows are censored and so absent from training)."""
    from xgboost import XGBClassifier
    fp = build_hazard_panel()
    feats = [c for c in fp.columns if c.startswith("f_")]
    m = train_v2.make_pipe(XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8,
        colsample_bytree=0.8, min_child_weight=15, reg_lambda=2.0,
        eval_metric="logloss", random_state=0), feats)
    m.fit(fp[CAT + feats], fp.y.values)
    # average months between reports, so a per-interval hazard becomes monthly
    return m, feats, float(fp.gap.mean())


def score_live(live, model, feats, gap):
    """-> monthly completion chance, and P(finished within 3 / 12 months)."""
    h_interval = model.predict_proba(live[CAT + feats])[:, 1]
    h = 1 - (1 - np.clip(h_interval, 1e-6, 0.95)) ** (1 / max(gap, 1))
    return h, 1 - (1 - h) ** 3, 1 - (1 - h) ** 12
