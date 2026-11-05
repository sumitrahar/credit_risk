"""Account view: one account's risk score, what drives it, and its history."""
from __future__ import annotations

import json

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from components import (
    INK_MUTED,
    LOWERS_RISK,
    RAISES_RISK,
    money,
    plain,
    repayment_timeline,
    risk_pill,
    show_chart,
    style_figure,
)
from src.anomaly import DETECTORS


def _drivers_chart(drivers: list[dict]) -> go.Figure:
    labels = [f"{d['label']}: {d['value']}" for d in drivers]
    values = [d["pd_points"] for d in drivers]
    fig = go.Figure(
        go.Bar(
            x=values, y=labels, orientation="h",
            text=[f"{v:+.1f} pts" for v in values], textposition="outside", cliponaxis=False,
            marker=dict(color=[RAISES_RISK if v > 0 else LOWERS_RISK for v in values], cornerradius=4),
            hovertext=[f"{label}<br>{'Raises' if v > 0 else 'Lowers'} the score by about {abs(v):.1f} points" for label, v in zip(labels, values)],
            hoverinfo="text",
        )
    )
    fig.update_yaxes(autorange="reversed", showgrid=False, type="category")
    fig.update_xaxes(title_text="Effect on predicted probability of default, in percentage points")
    fig.add_vline(x=0, line=dict(color=INK_MUTED, width=1))
    # Leave room on both sides, so the labels at the bar ends are never clipped.
    fig.update_xaxes(range=[min(0, min(values)) * 1.35 - 1, max(0, max(values)) * 1.35 + 1])
    return style_figure(fig, height=90 + 42 * len(drivers))


def _facts(row: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:
    loan = pd.DataFrame(
        [
            ("Loan type", row["loan_type"]),
            ("Stated purpose", row["loan_purpose"]),
            ("Original amount", money(row["loan_amount"])),
            ("Latest balance (approximate)", money(row["latest_balance"])),
            ("Interest rate", f"{row['interest_rate_pct']:.2f}%" + (" (recovered)" if row["interest_rate_recovered"] else "")),
            ("Term", f"{row['tenure_months']:.0f} months"),
            ("Originated", str(pd.Timestamp(row["origination_date"]).date())),
            ("Disbursal channel", row["disbursal_channel"]),
            ("Secured", "Yes" if row["collateral_flag"] else "No"),
        ],
        columns=["Loan", "Value"],
    )
    customer = pd.DataFrame(
        [
            ("Customer", row["customer_id"]),
            ("Age", f"{row['age']:.0f}"),
            ("Employment", row["employment_type"]),
            ("Annual income", money(row["annual_income"])),
            ("Years employed", f"{row['years_employed']:.1f}" + (" (imputed)" if row["years_employed_missing"] else "")),
            ("Region", f"{row['region']}, {row['city_tier']}"),
            ("Housing", row["housing_type"]),
            ("Bureau record", row["bureau_record"]),
        ],
        columns=["Customer", "Value"],
    )
    return loan, customer


def render(context: dict) -> None:
    portfolio = context["portfolio"]
    st.title("Account")

    selected = st.session_state.get("selected_account")
    default = selected or portfolio.sort_values("pd_score", ascending=False)["account_id"].iloc[0]
    if st.session_state.get("account_box_source") != default:
        st.session_state["account_box"] = default
        st.session_state["account_box_source"] = default
    typed = st.text_input("Account ID", key="account_box", help="For example ACC200001. Opens with the riskiest account until you choose one.")
    account_id = typed.strip().upper()
    match = portfolio[portfolio["account_id"] == account_id]
    if match.empty:
        st.warning(f"There is no account {account_id}. Account IDs run from ACC200001 to ACC214685.")
        return
    st.session_state["selected_account"] = account_id
    st.session_state["account_box_source"] = account_id
    row = match.iloc[0]

    columns = st.columns(5)
    columns[0].metric(
        "Predicted default", f"{row['pd_score'] * 100:.1f}%",
        help="Predicted probability of default. An out-of-fold score: from a model that did not see this customer in training.",
    )
    with columns[1]:
        st.markdown('<p class="eyebrow">Risk band</p>', unsafe_allow_html=True)
        st.markdown(f"<div style='font-size:1.35rem;margin-top:6px'>{risk_pill(row['risk_band'])}</div>", unsafe_allow_html=True)
    columns[2].metric(
        "Risk percentile", f"{min(row['pd_percentile'], 99.9):.1f}",
        help=f"Riskier than this percentage of all accounts. Within {row['loan_type']}: {min(row['pd_percentile_in_loan_type'], 99.9):.1f}.",
    )
    columns[3].metric("Latest balance", money(row["latest_balance"]))
    columns[4].metric("Recorded outcome", row["default_status"], help="target_default in the data. In live use this would not yet be known.")

    drivers = json.loads(row["drivers_json"])
    st.divider()
    left, right = st.columns([3, 2])
    with left:
        st.markdown("**Key drivers of this score**")
        st.caption("Red raises the score and blue lowers it, compared with the average account. Points are approximate.")
        show_chart(_drivers_chart(drivers), key="drivers")
    with right:
        st.markdown("**Anomaly status**")
        fired = json.loads(row["detectors_json"])
        reasons = json.loads(row["reasons_json"])
        if row["flagged"]:
            st.warning(f"In the review list, rank {row['review_rank']:,}. {len(fired)} detectors agree.", icon=":material/flag:")
        elif fired:
            st.info("One detector fired, which is not enough for the review list.", icon=":material/info:")
        else:
            st.success("No detector fired.", icon=":material/check_circle:")
        for detector, reason in zip(fired, reasons):
            st.markdown(f"- **{DETECTORS[detector]['name']}:** {plain(reason)}")
        if fired and st.button("Open in Anomaly Review", key="to_review"):
            st.session_state["review_account"] = account_id
            st.session_state["review_scope"] = "All accounts with a signal" if not row["flagged"] else "Review list"
            st.session_state["review_status"] = "All"  # so the account appears even if already reviewed
            st.session_state["pending_page"] = "Anomaly Review"
            st.rerun()

    st.divider()
    st.markdown("**Repayment history**")
    st.caption(f"Peers are {row['peer_group']} accounts: their typical average days past due is the line.")
    months = context["repayment"]
    months = months[months["account_id"] == account_id].sort_values("payment_month")
    show_chart(repayment_timeline(months, row["peer_median_dpd"]), key="timeline")
    with st.expander("View repayment months as a table"):
        st.dataframe(
            months[["payment_month", "emi_due", "emi_paid", "days_past_due", "payment_status", "outstanding_balance"]].rename(
                columns={
                    "payment_month": "Month", "emi_due": "Due", "emi_paid": "Paid", "days_past_due": "Days past due",
                    "payment_status": "Status", "outstanding_balance": "Balance after",
                }
            ),
            width="stretch", hide_index=True,
        )

    st.divider()
    left, right = st.columns(2)
    loan, customer = _facts(row)
    left.dataframe(loan, width="stretch", hide_index=True)
    right.dataframe(customer, width="stretch", hide_index=True)

    st.markdown("**Other lenders (bureau)**")
    bureau = context["bureau"]
    lines = bureau[bureau["customer_id"] == row["customer_id"]].sort_values("date_opened")
    if lines.empty:
        st.caption("This customer has no bureau record.")
    else:
        scoring_date = pd.Timestamp(row["origination_date"]) + pd.DateOffset(months=12)
        st.caption(f"Only lines opened by {scoring_date.date()} (12 months after origination) can feed this account's score.")
        st.dataframe(
            pd.DataFrame(
                {
                    "Credit type": lines["credit_type"],
                    "Lender": lines["lender_category"],
                    "Status": lines["account_status"],
                    "Limit": lines["sanctioned_amount"],
                    "Outstanding": lines["outstanding_amount"],
                    "Days past due": lines["days_past_due"],
                    "Opened": lines["date_opened"].dt.date,
                    "Closed": lines["date_closed"].dt.date,
                    "Used in score": (lines["date_opened"] <= scoring_date).map({True: "Yes", False: "No: opened later"}),
                    "Data flag": lines["date_closed_after_snapshot"].map({1: "Closure date is in the future", 0: ""}),
                }
            ),
            width="stretch", hide_index=True,
            column_config={
                "Limit": st.column_config.NumberColumn(format="$%,.0f"),
                "Outstanding": st.column_config.NumberColumn(format="$%,.0f"),
            },
        )

    st.markdown("**Collections activity**")
    collections = context["collections"]
    events = collections[collections["account_id"] == account_id].sort_values("event_date")
    if events.empty:
        st.caption("No collections activity on this account.")
    else:
        st.caption("Shown as provided. Collections are never used in the score.")
        st.dataframe(
            pd.DataFrame(
                {
                    "Date": events["event_date"].dt.date,
                    "Action": events["action_type"],
                    "Outcome": events["outcome"],
                    "Recovered": events["amount_recovered"],
                    "Data flag": events["event_before_origination"].map({1: "Dated before the loan began", 0: ""}),
                }
            ),
            width="stretch", hide_index=True,
            column_config={"Recovered": st.column_config.NumberColumn(format="$%,.0f")},
        )
