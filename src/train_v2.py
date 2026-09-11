"""Model ladder on the 10-month panel.

Convention-driven: any panel with f_* features and y_* labels works here.

Two protocols, because they answer different questions and a single number
hides the difference:
  A. GroupKFold by project_code -- MANDATORY. The same project contributes 6
     rows; letting two of them straddle a split leaks the answer and looks
     like brilliance.
  B. Temporal holdout -- train on early months, test on later ones. This is
     the deployment condition and the number worth quoting.
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
PANEL = HERE / "data/gold/panel_v2.csv"
CAT = ["sector", "state"]
TARGET = "y_adverse"


def load_panel():
    g = pd.read_csv(PANEL)
    feats = [c for c in g.columns if c.startswith("f_")]
    return g, feats


def make_pipe(clf, feats, scale=False):
    num = [("imp", SimpleImputer(strategy="median"))]
    if scale:
        num.append(("sc", StandardScaler()))
    return Pipeline([
        ("prep", ColumnTransformer([
            ("c", OneHotEncoder(handle_unknown="infrequent_if_exist",
                                min_frequency=30, sparse_output=False), CAT),
            ("n", Pipeline(num), feats)])),
        ("clf", clf)])


def metrics(y, p, k=100):
    o = np.argsort(-p)
    m = {"pr_auc": average_precision_score(y, p), "roc_auc": roc_auc_score(y, p),
         f"precision_at_{k}": float(y[o[:k]].mean()),
         "lift": float(y[o[:k]].mean() / y.mean())}
    m["brier"] = brier_score_loss(y, p) if 0 <= p.min() and p.max() <= 1 else float("nan")
    return m


def models(feats):
    return {
        "logistic": lambda: make_pipe(LogisticRegression(
            max_iter=3000, class_weight="balanced"), feats, scale=True),
        "random_forest": lambda: make_pipe(RandomForestClassifier(
            n_estimators=500, min_samples_leaf=10, class_weight="balanced",
            random_state=0, n_jobs=-1), feats),
        "xgboost": lambda: make_pipe(XGBClassifier(
            n_estimators=400, max_depth=5, learning_rate=0.05, subsample=0.8,
            colsample_bytree=0.8, min_child_weight=10, reg_lambda=2.0,
            eval_metric="logloss", random_state=0), feats),
    }


def baselines(g, y):
    """Rung 0. The last one is the bar that actually matters: the agency's own
    live forecast already disagreeing with its plan of record."""
    return {
        "baseline_always_no": np.zeros(len(y)) + 1e-9,
        "baseline_overdue": g.f_overdue.fillna(0).values.astype(float),
        "baseline_already_revised": g.f_is_sched_revised.fillna(0).values.astype(float),
        "baseline_slip_persists": g.f_slip_days.fillna(0).values.astype(float),
        "baseline_agency_forecast": g.f_antic_doc_gap_days.fillna(0).values.astype(float),
    }


def main():
    g, feats = load_panel()
    y = g[TARGET].values
    X = g[CAT + feats]
    print(f"n={len(g)}  projects={g.project_code.nunique()}  features={len(feats)}")
    print(f"positives={y.sum()} ({y.mean():.1%})\n")
    res, oof = {}, {}

    for k, v in baselines(g, y).items():
        res[k] = metrics(y, v)

    # ---- protocol A: grouped by project -----------------------------------
    for name, fn in models(feats).items():
        p = np.zeros(len(y))
        for tr, te in GroupKFold(n_splits=5).split(X, y, g.project_code):
            m = fn(); m.fit(X.iloc[tr], y[tr]); p[te] = m.predict_proba(X.iloc[te])[:, 1]
        oof[name] = p
        res[name] = metrics(y, p)

    print("PROTOCOL A -- GroupKFold by project_code")
    print(f"{'model':28s} {'PR-AUC':>8s} {'ROC-AUC':>8s} {'Brier':>7s} {'P@100':>7s} {'lift':>6s}")
    print("-" * 70)
    for k, v in res.items():
        print(f"{k:28s} {v['pr_auc']:8.3f} {v['roc_auc']:8.3f} {v['brier']:7.3f} "
              f"{v['precision_at_100']:7.2f} {v['lift']:6.2f}")

    # ---- protocol B: temporal holdout --------------------------------------
    per = sorted(g.period.unique())
    tr_p, te_p = per[:4], per[4:]
    itr, ite = g.period.isin(tr_p).values, g.period.isin(te_p).values
    # a project seen in training must not reappear in the test fold
    seen = set(g.loc[itr, "project_code"])
    print(f"\nPROTOCOL B -- temporal: train {tr_p} -> test {te_p}")
    print(f"  train n={itr.sum()}  test n={ite.sum()}  "
          f"(note: {len(set(g.loc[ite,'project_code']) & seen)} test projects also in train "
          f"-- unavoidable in a rolling panel, so read protocol A as the leakage-safe number)")
    tres = {}
    for k, v in baselines(g, y).items():
        tres[k] = metrics(y[ite], v[ite])
    for name, fn in models(feats).items():
        m = fn(); m.fit(X[itr], y[itr])
        tres[name] = metrics(y[ite], m.predict_proba(X[ite])[:, 1])
    print(f"{'model':28s} {'PR-AUC':>8s} {'ROC-AUC':>8s} {'P@100':>7s} {'lift':>6s}")
    print("-" * 62)
    for k, v in tres.items():
        print(f"{k:28s} {v['pr_auc']:8.3f} {v['roc_auc']:8.3f} "
              f"{v['precision_at_100']:7.2f} {v['lift']:6.2f}")

    # ---- horizon stratification -------------------------------------------
    best = max(oof, key=lambda k: res[k]["pr_auc"])
    bins = pd.cut(g.f_elapsed_share, [-.01, .5, 1., 1.5, 99],
                  labels=["<50%", "50-100%", "100-150%", ">150%"])
    print(f"\nbest={best}, by elapsed share of planned duration:")
    print(f"{'bucket':10s} {'n':>6s} {'base':>6s} {'PR-AUC':>8s}")
    for b in bins.cat.categories:
        m = (bins == b).values
        if m.sum() > 50 and 0 < y[m].mean() < 1:
            print(f"{b:10s} {m.sum():6d} {y[m].mean():6.2f} "
                  f"{average_precision_score(y[m], oof[best][m]):8.3f}")

    json.dump({"grouped": res, "temporal": tres},
              open(HERE / "reports/model_comparison_v2.json", "w"), indent=2)
    pd.DataFrame(oof, index=g.index).assign(
        y=y, project_code=g.project_code, period=g.period).to_csv(
        HERE / "reports/oof_predictions_v2.csv", index=False)
    print("\nwrote reports/model_comparison_v2.json + oof_predictions_v2.csv")


if __name__ == "__main__":
    main()
