"""Model & Monitoring view: how good the model is, what drives it, and how it would be watched."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from components import BLUE, INK_MUTED, ORANGE, SERIES, horizontal_bars, percent, plain_table, show_chart, style_figure

MONITORING_PLAN = [
    ("Score distribution (PSI)", "Monthly", "0.10 watch, 0.25 act", "Find what moved. Recalibrate if it lasts."),
    ("Input features (PSI), nulls, schema", "Every data load", "0.10 watch, 0.25 act; any schema change", "Fix the feed, or refit if the shift is real."),
    ("AUC, KS, Gini by vintage", "Quarterly, once outcomes mature", "AUC down 0.03 from validation", "Refit and compare against the challenger."),
    ("Calibration: predicted vs observed by risk band", "Quarterly", "A band off by more than 20% relative", "Recalibrate the intercept."),
    ("Fairness: error by age, gender, marital status, region", "Quarterly", "A gap of 3 points or more", "Compliance review."),
    ("Anomaly detectors: dismiss rate", "Monthly", "Dismiss rate above 80%", "Tighten or retire that detector."),
    ("Assistant: declines, blocked answers, marked wrong", "Weekly", "Any answer marked wrong; more than 5% blocked", "Review the catalogue and prompts."),
]


def _status(psi: float, watch: float, act: float) -> str:
    if psi >= act:
        return "Act"
    if psi >= watch:
        return "Watch"
    return "Stable"


def _headline(report: dict) -> None:
    champion = report["models"]["Logistic regression"]
    columns = st.columns(4)
    columns[0].metric("AUC", f"{champion['auc']:.3f}", help=f"95% interval {champion['auc_95_interval'][0]:.3f} to {champion['auc_95_interval'][1]:.3f}, resampling whole customers.")
    columns[1].metric("Gini", f"{champion['gini']:.3f}", help="2 x AUC - 1")
    columns[2].metric("KS", f"{champion['ks']:.3f}", help="Largest gap between the score distributions of defaulters and non-defaulters.")
    columns[3].metric("Out-of-time AUC", f"{champion['out_of_time']['auc']:.3f}", help=f"Trained on 2018-2021, tested on {champion['out_of_time']['test_accounts']:,} accounts from 2023 whose customers were not in training.")
    st.markdown(
        f"**Logistic regression is the main model.** {report['champion_rule']} LightGBM gained {report['lightgbm_gain_over_logistic']:+.3f}, "
        "so the simpler model that can be explained term by term stays. This data has modest signal: an AUC far above this would suggest leakage rather than skill."
    )


def _comparison(report: dict) -> None:
    rows = []
    for name, metrics in report["models"].items():
        rows.append(
            {
                "Model": name,
                "AUC": f"{metrics['auc']:.3f}",
                "95% interval": f"{metrics['auc_95_interval'][0]:.3f} to {metrics['auc_95_interval'][1]:.3f}",
                "Gini": f"{metrics['gini']:.3f}",
                "KS": f"{metrics['ks']:.3f}",
                "Fold AUC spread (SD)": f"{metrics['auc_fold_sd']:.3f}",
                "Out-of-time AUC": f"{metrics['out_of_time']['auc']:.3f}",
                "Out-of-time KS": f"{metrics['out_of_time']['ks']:.3f}",
            }
        )
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    st.caption(f"{report['validation']} {report['accounts']:,} accounts, {report['base_default_rate']:.1%} default rate, {report['feature_count']} features.")


def _curves(report: dict) -> None:
    left, right = st.columns(2)
    with left:
        st.markdown("**ROC curve**")
        fig = go.Figure()
        for (name, points), colour in zip(report["curves"]["roc"].items(), [BLUE, ORANGE]):
            fig.add_scatter(x=points["fpr"], y=points["tpr"], mode="lines", name=name, line=dict(color=colour, width=2),
                            hovertemplate=name + "<br>False positive rate %{x:.0%}<br>True positive rate %{y:.0%}<extra></extra>")
        fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random", line=dict(color=INK_MUTED, width=1), hoverinfo="skip")
        fig.update_xaxes(title_text="False positive rate", tickformat=".0%")
        fig.update_yaxes(title_text="True positive rate", tickformat=".0%")
        show_chart(style_figure(fig, height=340, show_legend=True), key="roc")
    with right:
        st.markdown("**KS: how far the two score distributions separate**")
        curve = report["curves"]["ks"]
        fig = go.Figure()
        fig.add_scatter(x=curve["score"], y=curve["non_defaults_cdf"], mode="lines", name="Non-defaulters", line=dict(color=BLUE, width=2), hovertemplate="Score %{x:.1%}<br>%{y:.0%} of non-defaulters below<extra></extra>")
        fig.add_scatter(x=curve["score"], y=curve["defaults_cdf"], mode="lines", name="Defaulters", line=dict(color=ORANGE, width=2), hovertemplate="Score %{x:.1%}<br>%{y:.0%} of defaulters below<extra></extra>")
        fig.update_xaxes(title_text="Predicted probability of default", tickformat=".0%", range=[0, 0.6])
        fig.update_yaxes(title_text="Share of accounts below this score", tickformat=".0%")
        show_chart(style_figure(fig, height=340, show_legend=True), key="ks")


def _deciles(report: dict) -> None:
    table = pd.DataFrame(report["deciles"])
    top = table.iloc[0]
    top3 = table.iloc[2]
    st.markdown("**Does the score rank accounts well?**")
    st.caption(
        f"The riskiest 10% of accounts default at {percent(top['default_rate'])}, {top['lift']:.1f} times the average. "
        f"The riskiest 30% contain {percent(top3['cumulative_capture'])} of all defaults."
    )
    left, right = st.columns(2)
    with left:
        fig = go.Figure()
        fig.add_bar(x=table["decile"], y=table["default_rate"], name="Observed", marker=dict(color=BLUE, cornerradius=4), hovertemplate="Decile %{x}: %{y:.1%} observed<extra></extra>")
        fig.add_scatter(x=table["decile"], y=table["mean_pd"], name="Predicted", mode="lines+markers", line=dict(color=ORANGE, width=2), marker=dict(size=8, line=dict(color="#fcfcfb", width=2)), hovertemplate="Decile %{x}: %{y:.1%} predicted<extra></extra>")
        fig.update_xaxes(title_text="Score decile (1 = riskiest)", dtick=1, showgrid=False)
        fig.update_yaxes(title_text="Default rate", tickformat=".0%", rangemode="tozero")
        show_chart(style_figure(fig, height=320, show_legend=True), key="deciles")
    with right:
        display = pd.DataFrame(
            {
                "Decile": table["decile"],
                "Accounts": table["accounts"].map("{:,.0f}".format),
                "Defaults": table["defaults"].map("{:,.0f}".format),
                "Default rate": table["default_rate"].map(percent),
                "Predicted": table["mean_pd"].map(percent),
                "Lift": table["lift"].map("{:.2f}".format),
                "Defaults captured (cumulative)": table["cumulative_capture"].map(percent),
            }
        )
        st.dataframe(display, width="stretch", hide_index=True, height=320)


def _importance(report: dict) -> None:
    st.markdown("**What drives the score**")
    st.caption(
        "Logistic regression uses the average size of each feature's contribution. LightGBM uses SHAP. Income and years employed lead both. "
        "Missed payments, late payments and average days past due carry the same signal, so the two models split it differently."
    )
    table = pd.DataFrame(report["importance"])
    left, right = st.columns(2)
    with left:
        st.caption("Logistic regression (main model): mean absolute contribution, in log-odds")
        top = table.sort_values("logistic_mean_abs_log_odds", ascending=False).head(12)
        show_chart(horizontal_bars(top["label"].tolist(), top["logistic_mean_abs_log_odds"], [f"{v:.3f}" for v in top["logistic_mean_abs_log_odds"]]), key="importance_logistic")
    with right:
        st.caption("LightGBM (challenger): mean absolute SHAP value")
        top = table.sort_values("lightgbm_mean_abs_shap", ascending=False).head(12)
        show_chart(horizontal_bars(top["label"].tolist(), top["lightgbm_mean_abs_shap"], [f"{v:.3f}" for v in top["lightgbm_mean_abs_shap"]], colour=SERIES[1]), key="importance_shap")


def _sensitivity(report: dict) -> None:
    st.markdown("**How each modelling choice moves the result**")
    st.caption("Logistic regression throughout. The final model deliberately gives up a little AUC to avoid look-ahead and sensitive attributes.")
    table = pd.DataFrame(report["sensitivity"])
    final_auc = table.iloc[0]["auc"]
    plain_table(
        pd.DataFrame(
            {
                "Variant": table["variant"],
                "AUC": table["auc"].map("{:.3f}".format),
                "Change": (table["auc"] - final_auc).map("{:+.3f}".format),
                "Gini": table["gini"].map("{:.3f}".format),
                "KS": table["ks"].map("{:.3f}".format),
                "Features": table["feature_count"].astype(str),
                "Why it is here": table["note"],
            }
        )
    )


def _monitoring(report: dict) -> None:
    stability = report["stability"]
    watch, act = stability["watch_threshold"], stability["act_threshold"]
    vintages = pd.DataFrame(stability["by_vintage"])
    features = pd.DataFrame(stability["by_feature"])
    st.subheader("Monitoring after deployment")
    st.caption(f"A live example on this data, against a baseline of {stability['baseline']}. PSI measures how far a distribution has moved: under {watch} is stable, {act} or more needs action.")

    alerts = features[features["psi"] >= watch].sort_values("psi", ascending=False)
    first_year, last_year = vintages.iloc[0], vintages.iloc[-1]
    if not alerts.empty:
        names = ", ".join(f"{row.label} (PSI {row.psi:.2f})" for row in alerts.itertuples())
        st.warning(
            f"**Alert on this data:** {names}. The score itself is stable (PSI at most {vintages['psi_vs_baseline'].max():.3f}). The cause is structural: bureau open dates run through 2025, "
            f"so newer loans can see more lines by their scoring date, {last_year['mean_visible_bureau_lines']:.2f} on average for {last_year['origination_year']:.0f} against "
            f"{first_year['mean_visible_bureau_lines']:.2f} for {first_year['origination_year']:.0f}. It also explains why AUC rises by vintage, from {first_year['auc']:.3f} to {last_year['auc']:.3f}. "
            "This is exactly the kind of shift monitoring is for: in production it would prompt a question to the bureau data team.",
            icon=":material/warning:",
        )

    left, right = st.columns(2)
    with left:
        st.markdown("**By origination year**")
        display = pd.DataFrame(
            {
                "Year": vintages["origination_year"].astype(str),
                "Accounts": vintages["accounts"].map("{:,}".format),
                "Score PSI": vintages["psi_vs_baseline"].map("{:.3f}".format),
                "AUC": vintages["auc"].map("{:.3f}".format),
                "KS": vintages["ks"].map("{:.3f}".format),
                "Observed": vintages["observed_default_rate"].map(percent),
                "Predicted": vintages["predicted_default_rate"].map(percent),
            }
        )
        st.dataframe(display, width="stretch", hide_index=True)
    with right:
        st.markdown("**Top features, 2022-2023 against the baseline**")
        display = features.sort_values("psi", ascending=False)
        st.dataframe(
            pd.DataFrame(
                {
                    "Feature": display["label"],
                    "PSI": display["psi"].map("{:.3f}".format),
                    "Status": display["psi"].map(lambda v: _status(v, watch, act)),
                }
            ),
            width="stretch", hide_index=True,
        )

    st.markdown("**Monitoring plan**")
    plain_table(pd.DataFrame(MONITORING_PLAN, columns=["What", "How often", "Alert at", "Action"]))


def _fairness(report: dict) -> None:
    st.markdown("**Fairness check on the excluded attributes**")
    table = pd.DataFrame(report["fairness"])
    ages = table[table["attribute"] == "age_band"]
    youngest, oldest = ages.iloc[0], ages.iloc[-1]
    st.caption(
        f"Gender, marital status, region and age are not model inputs, but error is still checked across them. Leaving age out has a visible cost: "
        f"the {youngest['value']} band is under-predicted ({percent(youngest['mean_predicted_pd'])} predicted, {percent(youngest['observed_default_rate'])} observed) and "
        f"{oldest['value']} is over-predicted ({percent(oldest['mean_predicted_pd'])} predicted, {percent(oldest['observed_default_rate'])} observed). "
        f"That is a decision for compliance, not a modelling detail. Years employed is only weakly linked to age (r = {report['age_proxy']['years_employed_vs_age']:.2f})."
    )
    st.dataframe(
        pd.DataFrame(
            {
                "Attribute": table["attribute"].str.replace("_", " ").str.capitalize(),
                "Value": table["value"],
                "Accounts": table["accounts"].map("{:,.0f}".format),
                "Observed default rate": table["observed_default_rate"].map(percent),
                "Average predicted": table["mean_predicted_pd"].map(percent),
                "Gap (points)": ((table["mean_predicted_pd"] - table["observed_default_rate"]) * 100).map("{:+.1f}".format),
            }
        ),
        width="stretch", hide_index=True, height=300,
    )


def render(context: dict) -> None:
    report = context["model_report"]
    st.title("Model & Monitoring")
    st.caption("Predicts target_default using customer and loan data, 12 months of repayment history and date-limited bureau data.")
    _headline(report)
    _comparison(report)
    st.divider()
    _curves(report)
    st.divider()
    _deciles(report)
    st.divider()
    _importance(report)
    st.divider()
    _sensitivity(report)
    st.divider()
    _monitoring(report)
    st.divider()
    _fairness(report)
