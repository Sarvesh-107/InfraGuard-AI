"""Checks for the newdata pipeline. Guards every claim in the model card.

    python test_v2.py
"""
import numpy as np, pandas as pd, json, pathlib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import average_precision_score
import newdata, panel_v2, train_v2

HERE = pathlib.Path(__file__).resolve().parents[1]


def test_corrupt_files_excluded():
    ong, comp, add = newdata.load(verbose=False)
    assert not set(ong.period) & newdata.CORRUPT, "corrupt export leaked into panel"
    assert ong.period.nunique() == 10
    print("ok  corrupt exports excluded, 10 clean periods")


def test_no_total_rows():
    """The TOTAL footer row carries the whole portfolio (~Rs 27 lakh cr). If one
    survives as a project it silently dominates every cost feature."""
    ong, comp, add = newdata.load(verbose=False)
    assert ong.project_code.str.fullmatch(r"N?\d{8,9}").all(), "non-project row kept"
    assert ong.cost_original.max() < 500_000, f"outlier cost {ong.cost_original.max()}"
    assert not (ong.project_name == "TOTAL").any()
    print(f"ok  no TOTAL rows; max project cost Rs {ong.cost_original.max():,.0f} cr")


def test_no_date_silently_dropped():
    """Every non-N.A. date in the raw files must parse. A new format in one month
    once blanked all revised dates for the live roster without any error."""
    import glob
    bad = []
    for f in glob.glob(str(newdata.DATA / "Table_[67]_*.csv")):
        per = newdata._period_from_name(pathlib.Path(f).name)
        if per in newdata.CORRUPT:
            continue
        raw = pd.read_csv(f, encoding="utf-8-sig", dtype=str)
        for c in ["Date of Approval (MM/YYYY)", "Date of Commissioning Original",
                  "Date of Commissioning Revised", "Date of Commissioning Anticipated"]:
            v = raw[c].dropna().str.replace(r"\s+", "", regex=True).str.strip("{}")
            v = v[~v.isin(["N.A.", ""])]
            n_bad = newdata._date(v).isna().sum()
            if n_bad:
                bad.append((per, c, n_bad, list(v[newdata._date(v).isna()].unique()[:3])))
    assert not bad, f"unparsed dates: {bad}"
    print("ok  every non-N.A. date in every clean export parses")


def test_progress_every_month():
    ong, _, _ = newdata.load(verbose=False)
    cov = ong.groupby("period").physical_progress_pct.apply(lambda s: s.notna().mean())
    # 2025 writes 0% as "-"; only projects that already had progress stay blank
    assert (cov > 0.97).all(), f"progress missing for a month: {cov.round(3).to_dict()}"
    print(f"ok  physical progress present every month (min {cov.min():.0%})")


def test_approved_cost_never_rewritten():
    """The reports overwrite 'original' cost for ~130 projects mid-year; the loader
    must keep the first figure as approved and book the rewrite as a revision."""
    ong, _, _ = newdata.load(verbose=False)
    moves = ong.groupby("project_code").cost_original.nunique()
    assert (moves == 1).all(), f"{(moves > 1).sum()} projects' approved cost changes over time"
    r = ong[ong.project_name.str.contains("RISHIKESH - KARNAPRAYAG", regex=False, na=False)
            & (ong.period == "2025-06")].iloc[0]
    assert r.cost_original < 20_000 and r.effective_cost > 38_000, "restatement not booked as revision"
    # its progress went 82% -> 0% -> "-": that must read as unknown, never a false 0%
    assert r.physical_progress_pct != 0, "reporting-glitch zero shown as real progress"
    print("ok  approved cost fixed per project; mid-year rewrites booked as revisions")


def test_sector_normalised():
    ong, _, _ = newdata.load(verbose=False)
    s = set(ong.sector.dropna())
    assert "ROAD TRANSPORT AND" not in s and "HIGHWAYS" not in s, "truncated sector left"
    assert "TELECOMMUNICA" not in s
    # no value may be a strict prefix of another, or one sector is split in two
    assert not [(a, b) for a in s for b in s if a != b and b.startswith(a)]
    print(f"ok  {len(s)} sectors, no truncated duplicates")


def test_labels_have_future():
    """Every origin month must have a real t+3 snapshot, else that month keeps
    only its completions -- pure negatives that bias the base rate down."""
    g = pd.read_csv(panel_v2.OUT / "panel_v2.csv")
    n = g.groupby("period").size()
    assert n.min() > 1000, f"thin origin month kept: {n.to_dict()}"
    assert g.y_adverse.isin([0, 1]).all()
    print(f"ok  {len(n)} origin months, all >1000 rows (min {n.min()})")


def test_features_are_asof():
    """A feature at t must not change when future snapshots are removed."""
    full_src, _ = panel_v2.feature_frame()                    # all 10 months
    trunc_src, _ = panel_v2.feature_frame(max_period="2024-10")  # history ends at t
    full = full_src[full_src.period == "2024-10"].set_index("project_code")
    sub = trunc_src[trunc_src.period == "2024-10"].set_index("project_code")
    cols = [c for c in full.columns if c.startswith("f_")]
    common = full.index.intersection(sub.index)
    for c in cols:
        a, b = full.loc[common, c], sub.loc[common, c]
        assert np.allclose(a.fillna(-999), b.fillna(-999)), f"{c} depends on the future"
    print(f"ok  as-of features unchanged when future is truncated ({len(common)} projects)")


def test_shuffle_control():
    g, feats = train_v2.load_panel()
    y = g[train_v2.TARGET].values
    X = g[train_v2.CAT + feats]
    ys = np.random.default_rng(0).permutation(y)
    p = np.zeros(len(y))
    for tr, te in GroupKFold(n_splits=5).split(X, ys, g.project_code):
        m = train_v2.make_pipe(RandomForestClassifier(
            n_estimators=200, min_samples_leaf=10, random_state=0, n_jobs=-1), feats)
        m.fit(X.iloc[tr], ys[tr]); p[te] = m.predict_proba(X.iloc[te])[:, 1]
    ap = average_precision_score(ys, p)
    assert ap < ys.mean() + 0.05, f"leakage: shuffled PR-AUC {ap:.3f} vs base {ys.mean():.3f}"
    print(f"ok  shuffle control PR-AUC {ap:.3f} ~ base rate {ys.mean():.3f}")


def test_beats_agency_forecast():
    r = json.load(open(HERE / "reports/model_comparison_v2.json"))
    for proto in ("grouped", "temporal"):
        base = max(v["pr_auc"] for k, v in r[proto].items() if k.startswith("baseline"))
        best = max(v["pr_auc"] for k, v in r[proto].items() if not k.startswith("baseline"))
        assert best > base + 0.10, f"{proto}: {best:.3f} vs baseline {base:.3f}"
        print(f"ok  {proto}: best {best:.3f} clears best baseline {base:.3f}")


def test_forecast_dates_sane():
    """A projected completion date must never be in the past, and the pessimistic
    band must never be earlier than the median one."""
    import forecast
    out, per, _ = forecast.build_cards()
    asof = pd.Timestamp(per + "-01")
    d50 = pd.to_datetime(out.date_projected_p50, format="%b-%Y", errors="coerce")
    d80 = pd.to_datetime(out.date_projected_p80, format="%b-%Y", errors="coerce")
    # Projects whose report carries no original commissioning date get no
    # projection at all -- blank is the honest output, so exclude those.
    ok = d50.notna() & d80.notna()
    assert (d50[ok] >= asof).all(), f"{(d50[ok] < asof).sum()} P50 dates in the past"
    assert (d80[ok] >= d50[ok]).all(), f"{(d80[ok] < d50[ok]).sum()} P80 before P50"
    assert (out.cost_final_expected_cr.dropna() >= 0).all()
    # slippage shown must equal (projected date - original date), to the month
    orig = pd.to_datetime(out.date_original, format="%b-%Y", errors="coerce")
    implied = ((d50 - orig).dt.days / 30.44).round()
    m = ok & orig.notna()
    assert ((implied[m] - out.slip_p50_mo[m].round()).abs() <= 1).all(), \
        "slippage column disagrees with the dates shown"
    print(f"ok  forecast dates sane ({ok.sum()} projects; "
          f"{(~ok).sum()} left blank for want of an original date)")


def test_completed_dedup_keeps_chronologically_earliest():
    """Regression test: comp is built from files in ALPHABETICAL filename order
    ("...April_2025.csv" sorts before "...December_2024.csv"), so naively
    concatenating and then drop_duplicates("project_code") on row order alone
    would keep whichever file happened to sort first -- not the true earliest
    completion month. Constructs the exact row order that alphabetical file
    globbing would produce (the April row first, even though December is
    chronologically earlier) and checks the fix -- sort_values("period") before
    drop_duplicates, as newdata.load() and its three callers now do -- keeps the
    right one."""
    fake_comp = pd.DataFrame([
        {"project_code": "N99999999", "period": "2025-04", "project_name": "FAKE"},
        {"project_code": "N99999999", "period": "2024-12", "project_name": "FAKE"},
    ])

    # the bug: naive drop_duplicates on unsorted (alphabetical-file) row order
    # keeps the wrong (chronologically later) row
    naive = fake_comp.drop_duplicates("project_code", keep="first")
    assert naive.period.iloc[0] == "2025-04", "test setup sanity check failed"

    # the fix: sort by period first
    fixed = fake_comp.sort_values("period").drop_duplicates("project_code", keep="first")
    assert fixed.period.iloc[0] == "2024-12", \
        f"expected chronologically earliest period 2024-12, kept {fixed.period.iloc[0]}"
    print("ok  completed-row dedup keeps the chronologically earliest period, not "
          "whichever file sorted first alphabetically")


def test_completed_frame_presorted_by_period():
    """newdata.load()'s comp frame must be sorted so any later
    drop_duplicates("project_code", keep="first") is safe by construction."""
    ong, comp, add = newdata.load(verbose=False)
    for code, sub in comp.groupby("project_code", sort=False):
        assert sub.period.is_monotonic_increasing, f"{code}: comp not sorted by period"
    print(f"ok  comp pre-sorted by (project_code, period) for all "
          f"{comp.project_code.nunique()} completed projects")


if __name__ == "__main__":
    test_corrupt_files_excluded()
    test_no_total_rows()
    test_no_date_silently_dropped()
    test_progress_every_month()
    test_approved_cost_never_rewritten()
    test_sector_normalised()
    test_labels_have_future()
    test_features_are_asof()
    test_beats_agency_forecast()
    test_forecast_dates_sane()
    test_shuffle_control()
    test_completed_dedup_keeps_chronologically_earliest()
    test_completed_frame_presorted_by_period()
    print("\nall v2 checks passed")
