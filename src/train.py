"""Model ladder for near-term escalation risk.

Rungs, each judged on the SAME protocol (GroupKFold by agency -- the real
deployment condition, since 93 of 179 agencies hold a single project and a
model that has memorised NHAI tells us nothing about the tail):

  0  naive baselines            <- the honesty anchor, always published
  1  logistic regression        <- PS part (b), statistical arm
  2  random forest / xgboost    <- PS part (b), ML arm

Survival and sequence rungs from docs/ML_PLAN.md are deliberately NOT built:
with one 3-month label window they cannot be validated, let alone beat rung 0.
"""
import pandas as pd, numpy as np, pathlib, json, warnings
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
HERE = pathlib.Path(__file__).resolve().parents[1]
TARGET = "y_adverse"
CAT = ["sector", "ministry", "state"]
NUM = ["log_cost", "planned_duration_days", "age_days", "physical_progress_pct",
       "financial_progress_pct", "progress_gap_pct", "is_cost_revised",
       "is_schedule_revised", "declared_cost_overrun_pct", "declared_slip_days",
       "is_ne", "elapsed_share", "months_past_original_doc", "already_overdue",
       "months_remaining", "required_velocity", "is_newly_added"]


def prep():
    g = pd.read_csv(HERE / "data/gold/panel_t0.csv", index_col="project_code")
    return g, g[CAT + NUM], g[TARGET].values, g["agency"].values


def make_pipe(clf, scale=False):
    # min_frequency pools the long tail of rare sectors/states instead of
    # exploding into columns that appear in one fold only.
    cat = OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=20,
                        sparse_output=False)
    num = [("imp", SimpleImputer(strategy="median"))]
    if scale:
        num.append(("sc", StandardScaler()))
    return Pipeline([
        ("prep", ColumnTransformer([("c", cat, CAT), ("n", Pipeline(num), NUM)])),
        ("clf", clf)])


def metrics(y, p, k=50):
    order = np.argsort(-p)
    out = {"pr_auc": average_precision_score(y, p),
           "roc_auc": roc_auc_score(y, p),
           f"precision_at_{k}": float(y[order[:k]].mean()),
           "lift_at_50": float(y[order[:k]].mean() / y.mean())}
    # Rank-only baselines (e.g. raw slip days) are scores, not probabilities;
    # Brier is meaningless for them, so report it only when it is well defined.
    out["brier"] = brier_score_loss(y, p) if p.min() >= 0 and p.max() <= 1 else float("nan")
    return out


def cv(model_fn, X, y, groups, n=5):
    oof = np.zeros(len(y))
    for tr, te in GroupKFold(n_splits=n).split(X, y, groups):
        m = model_fn()
        m.fit(X.iloc[tr], y[tr])
        oof[te] = m.predict_proba(X.iloc[te])[:, 1]
    return oof


def main():
    g, X, y, groups = prep()
    print(f"n={len(y)}  positives={y.sum()} ({y.mean():.1%})  agencies={len(set(groups))}\n")
    res, preds = {}, {}

    # ---- rung 0: naive baselines (no learning at all) ----------------------
    res["baseline_always_no"] = metrics(y, np.zeros(len(y)) + 1e-9)
    res["baseline_already_overdue"] = metrics(y, g.already_overdue.values.astype(float))
    res["baseline_schedule_revised"] = metrics(y, g.is_schedule_revised.values.astype(float))
    # "current slip persists" -- the embarrassingly strong one the plan warns about
    res["baseline_declared_slip"] = metrics(y, g.declared_slip_days.fillna(0).values.astype(float))
    # sector historical rate, computed out-of-fold so it is a fair comparison
    sec = np.zeros(len(y))
    for tr, te in GroupKFold(n_splits=5).split(X, y, groups):
        rate = pd.Series(y[tr]).groupby(g.sector.values[tr]).mean()
        sec[te] = pd.Series(g.sector.values[te]).map(rate).fillna(y[tr].mean()).values
    res["baseline_sector_rate"] = metrics(y, sec)

    # ---- rung 1: statistical arm ------------------------------------------
    preds["logistic"] = cv(lambda: make_pipe(
        LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced"), scale=True),
        X, y, groups)
    res["logistic"] = metrics(y, preds["logistic"])

    # ---- rung 2: ML arm ----------------------------------------------------
    preds["random_forest"] = cv(lambda: make_pipe(
        RandomForestClassifier(n_estimators=400, min_samples_leaf=8,
                               class_weight="balanced", random_state=0, n_jobs=-1)),
        X, y, groups)
    res["random_forest"] = metrics(y, preds["random_forest"])

    preds["xgboost"] = cv(lambda: make_pipe(
        XGBClassifier(n_estimators=350, max_depth=4, learning_rate=0.05,
                      subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
                      reg_lambda=2.0, eval_metric="logloss", random_state=0)),
        X, y, groups)
    res["xgboost"] = metrics(y, preds["xgboost"])

    print(f"{'model':28s} {'PR-AUC':>8s} {'ROC-AUC':>8s} {'Brier':>7s} {'P@50':>6s} {'lift':>6s}")
    print("-" * 68)
    for k, v in res.items():
        print(f"{k:28s} {v['pr_auc']:8.3f} {v['roc_auc']:8.3f} {v['brier']:7.3f} "
              f"{v['precision_at_50']:6.2f} {v['lift_at_50']:6.2f}")

    # ---- horizon stratification: a pooled number hides where it works ------
    best = max(preds, key=lambda k: res[k]["pr_auc"])
    bins = pd.cut(g.elapsed_share, [-.01, .5, 1., 1.5, 99],
                  labels=["<50%", "50-100%", "100-150%", ">150%"])
    print(f"\nbest = {best}, by elapsed_share (share of planned duration used):")
    print(f"{'bucket':10s} {'n':>5s} {'base':>6s} {'PR-AUC':>8s}")
    for b in bins.cat.categories:
        m = (bins == b).values
        if m.sum() > 30 and 0 < y[m].mean() < 1:
            print(f"{b:10s} {m.sum():5d} {y[m].mean():6.2f} "
                  f"{average_precision_score(y[m], preds[best][m]):8.3f}")

    (HERE / "reports").mkdir(exist_ok=True)
    json.dump(res, open(HERE / "reports/model_comparison.json", "w"), indent=2)
    pd.DataFrame(preds, index=g.index).assign(y=y, agency=groups).to_csv(
        HERE / "reports/oof_predictions.csv")
    print(f"\nwrote reports/model_comparison.json + oof_predictions.csv")


if __name__ == "__main__":
    main()
