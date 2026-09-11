"""Prediction-model accuracy and drivers from PAIMANA backtests."""

import bootstrap  # noqa: F401

import altair as alt
import pandas as pd
import streamlit as st

from components.layout import setup_page
from components.styles import glass_divider
from data_loader import MODELS, load_drivers, load_reports, report_mtimes
from risk_engine import RISK_COLORS, TIERS

setup_page(
    "models",
    title="Prediction Models",
    subtitle="Accuracy on projects the system had not seen, and on projects that actually finished",
    page_title="Models | PAIMANA AI",
)

oof, ends, cost, comp = load_reports(report_mtimes())
yax = alt.Axis(labelLimit=340)

st.markdown(
    "Every number below comes from testing on projects the system **had not seen**, "
    "or on projects that **actually finished**, predicted from the months before they did."
)
glass_divider()

with st.container(border=True):
    st.subheader("How accurate is each prediction")
    o = oof.assign(
        tier=pd.cut(100 * oof.xgboost, [-1, 25, 50, 75, 101], labels=["Low", "Medium", "High", "Critical"])
    )
    hit = o.groupby("tier", observed=True).y.agg(["size", "mean"]).reindex(TIERS)
    st.markdown(
        "**Risk score** — share of projects in each tier that really declared a cost "
        "increase or delay within 3 months:"
    )
    cols = st.columns(4)
    for col, (t, h) in zip(cols, hit.iterrows()):
        with col:
            if pd.isna(h["mean"]):
                st.metric(str(t), "—")
            else:
                st.metric(str(t), f"{100 * h['mean']:.0f}%", f"{int(h['size']):,} tested")

    e = cost.err_pct.abs()
    v = ends.err_ours.abs()
    m1, m2, m3 = st.columns(3)
    m1.metric(
        "Predicted final cost within ±1% of actual",
        f"{100 * (e <= 1).mean():.0f}%",
        f"{len(cost)} finished projects",
    )
    m2.metric(
        "Predicted end date within 6 months",
        f"{100 * (v <= 6).mean():.0f}%",
        f"typical miss {v.median():.0f} months",
    )
    m3.metric(
        "Agency's own date within 6 months",
        f"{100 * (ends.err_agency.abs() <= 6).mean():.0f}%",
        f"too optimistic {100 * (ends.err_agency < -1).mean():.0f}% of the time",
    )
    st.caption(
        f"End dates: {ends.project_code.nunique()} finished projects, {len(ends):,} predictions "
        f"made 1–11 months before they finished. The actual finish fell on or before our "
        f"worst-case date {100 * ends.covered_by_worst_case.mean():.0f}% of the time."
    )

c1, c2 = st.columns(2, gap="large")
with c1:
    with st.container(border=True):
        st.subheader("Simple rules vs statistical vs machine learning")
        names = {
            "baseline_overdue": ("Rule: already overdue", "Simple rule"),
            "baseline_already_revised": ("Rule: already rescheduled", "Simple rule"),
            "baseline_slip_persists": ("Rule: current delay continues", "Simple rule"),
            "baseline_agency_forecast": ("Agency's own estimate", "Simple rule"),
            "logistic": ("Logistic regression", "Statistical"),
            "random_forest": ("Random forest", "Machine learning"),
            "xgboost": ("XGBoost", "Machine learning"),
        }
        mc = pd.DataFrame(
            [
                {"method": names[k][0], "type": names[k][1], "hit": v["precision_at_100"]}
                for k, v in comp["grouped"].items()
                if k in names
            ]
        )
        st.altair_chart(
            alt.Chart(mc)
            .mark_bar()
            .encode(
                x=alt.X(
                    "hit:Q",
                    title="Of the 100 projects flagged riskiest, share that really slipped",
                    axis=alt.Axis(format="%"),
                    scale=alt.Scale(domain=[0, 1]),
                ),
                y=alt.Y("method:N", sort="-x", title=None, axis=yax),
                color=alt.Color(
                    "type:N",
                    title=None,
                    legend=alt.Legend(orient="bottom"),
                    scale=alt.Scale(
                        domain=["Simple rule", "Statistical", "Machine learning"],
                        range=["#9e9e9e", "#1976d2", RISK_COLORS["Critical"]],
                    ),
                ),
                tooltip=["method", "type", alt.Tooltip("hit:Q", format=".0%")],
            )
            .properties(height=300),
            width="stretch",
        )
        g = comp["grouped"]
        ml_hit = max(g["random_forest"]["precision_at_100"], g["xgboost"]["precision_at_100"])
        st.caption(
            f"Machine learning flags risk better than the statistical model — "
            f"{100 * ml_hit:.0f}% vs {100 * g['logistic']['precision_at_100']:.0f}% of top-100 flags correct."
        )

with c2:
    with st.container(border=True):
        st.subheader("What drives the risk score")
        model_path = MODELS / "escalation_xgb_v2.joblib"
        mtime = model_path.stat().st_mtime if model_path.exists() else None
        try:
            dr = load_drivers(mtime)
        except Exception:
            dr = None
        if dr is None:
            if model_path.exists():
                st.info(
                    "The trained model file is present, but it could not be loaded "
                    "(install `joblib`, `scikit-learn`, and `xgboost`, or use Python 3.11/3.12). "
                    "Accuracy charts on the left still use the saved backtest reports."
                )
            else:
                st.info(
                    "Trained model file `models/escalation_xgb_v2.joblib` is not in this workspace, "
                    "so live feature importances cannot be shown. Accuracy charts on the left still "
                    "use the saved backtest reports."
                )
        else:
            st.altair_chart(
                alt.Chart(dr.head(12))
                .mark_bar(color=RISK_COLORS["High"])
                .encode(
                    x=alt.X("importance:Q", title="Influence on the model", axis=alt.Axis(format="%")),
                    y=alt.Y("factor:N", sort="-x", title=None, axis=yax),
                    tooltip=["factor", alt.Tooltip("importance:Q", format=".1%")],
                )
                .properties(height=300),
                width="stretch",
            )
            st.caption(
                "Land acquisition, court cases and contractor changes are not in the monthly "
                "reports, so their effect cannot be measured from this data yet."
            )
