"""One runnable check. Guards the claims the model card makes.

    python test_prototype.py
"""
import numpy as np, pandas as pd, pathlib, json
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score
import panel as P, train as T

HERE = pathlib.Path(__file__).resolve().parents[1]


def test_no_double_count():
    d, ong, ne, new, comp = P.load_silver()
    assert not ong.duplicated(["period", "project_code"]).any(), "duplicate roster rows"
    # ne_region must be a subset of all_ongoing, never appended to it: every NE
    # code has to appear in the roster, or concatenating the kinds would add rows.
    assert ne <= set(ong.project_code), f"{len(ne - set(ong.project_code))} ne rows outside roster"
    print("ok  no double counting")


def test_features_match_between_train_and_serve():
    d, ong, ne, new, comp = P.load_silver()
    tr = P.features(ong[ong.period == P.T0].set_index("project_code"), ne, new, P.T0)
    lv = P.features(ong[ong.period == "2026-07"].set_index("project_code"), ne, new, "2026-07")
    assert list(tr.columns) == list(lv.columns), "train/serve feature skew"
    print("ok  train and serve build identical features")


def test_no_label_in_features():
    g = pd.read_csv(HERE / "data/gold/panel_t0.csv", index_col="project_code")
    for c in T.CAT + T.NUM:
        assert not c.startswith("y_") and c not in ("exit_status",), f"label leaked: {c}"
    assert set(g.y_adverse.unique()) <= {0, 1}
    print("ok  no label column reachable from features")


def test_shuffle_control():
    """The real leakage test: on shuffled labels the model must collapse to the
    base rate. If it still scores well, something in X encodes the answer."""
    g, X, y, groups = T.prep()
    rng = np.random.default_rng(0)
    ys = rng.permutation(y)
    oof = T.cv(lambda: T.make_pipe(RandomForestClassifier(
        n_estimators=200, min_samples_leaf=8, random_state=0, n_jobs=-1)),
        X, ys, groups)
    ap = average_precision_score(ys, oof)
    assert ap < ys.mean() + 0.05, f"leakage: shuffled PR-AUC {ap:.3f} vs base {ys.mean():.3f}"
    print(f"ok  shuffle control PR-AUC {ap:.3f} ~ base rate {ys.mean():.3f}")


def test_beats_best_baseline():
    r = json.load(open(HERE / "reports/model_comparison.json"))
    base = max(v["pr_auc"] for k, v in r.items() if k.startswith("baseline"))
    best = max(v["pr_auc"] for k, v in r.items() if not k.startswith("baseline"))
    assert best > base + 0.10, f"model {best:.3f} does not clear baseline {base:.3f}"
    print(f"ok  best model {best:.3f} clears best baseline {base:.3f}")


if __name__ == "__main__":
    test_no_double_count()
    test_features_match_between_train_and_serve()
    test_no_label_in_features()
    test_beats_best_baseline()
    test_shuffle_control()
    print("\nall checks passed")
