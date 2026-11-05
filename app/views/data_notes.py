"""Data Notes view: join logic, every data-quality decision, and the assumptions behind the numbers."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from components import plain_table

JOINS = [
    ("customers.csv", "1 per customer", "loan_accounts.csv", "customer_id", "One customer to many loans. Left join from loans, so no account is lost."),
    ("repayment_history.csv", "12 per account", "loan_accounts.csv", "account_id", "Aggregated to one row per account first: missed and late counts, average and worst days past due, longest missed run, trend."),
    ("bureau_credit.csv", "0-9 per customer", "loan_accounts.csv", "customer_id, limited by date", "Aggregated per account. Only lines opened by origination + 12 months count, and later closures are rewound to Active."),
    ("bureau_credit.csv", "0-9 per customer", "customers.csv", "customer_id", "A second, current view for anomaly detection and monitoring. Never a model input."),
    ("collections_events.csv", "0-3 per account", "loan_accounts.csv", "account_id", "Aggregated per account and left-joined. Display only: never a model input."),
]

ASSUMPTIONS = [
    "The data extract is taken at 2025-12-31, because the latest bureau line opened on 2025-12-29.",
    "Default is observed after month 12, because the target has no date and many defaulters never missed a payment. That makes the model a month-12 behaviour score. Sensitivity for 3 and 6 months is on the Model view.",
    "Status, balance and days past due on Active bureau lines are as at last reporting, which has no date. This small look-ahead cannot be removed with the data provided.",
    "Exposure is shown as both originated amount and latest balance. Latest balance is approximate, because balances fall by a fixed 60% of each payment.",
    "Amounts are in USD, as the data dictionary states.",
    "Credit cards are treated as installment products, because the data gives them a term and installment.",
]


def render(context: dict) -> None:
    log = context["dq_log"]
    portfolio = context["portfolio"]

    st.title("Data Notes")
    st.caption(f"How the five raw tables became one clean dataset of {len(portfolio):,} accounts, one row per account.")

    st.subheader("Join logic")
    plain_table(pd.DataFrame(JOINS, columns=["Table", "Rows", "Joined to", "On", "How"]))
    st.caption("All keys are unique and no record is orphaned: the build stops if that ever changes. A test confirms the join stays at 14,685 rows.")

    st.divider()
    st.subheader("Every data-quality decision")
    st.caption(f"{len(log)} entries, with counts computed from the data at build time. Nothing was silently dropped: no rows were removed from any table.")
    only = st.segmented_control(
        "Show", ["All", "In the data dictionary", "Found during analysis"], default="All", key="dq_scope", label_visibility="collapsed",
    ) or "All"
    shown = log
    if only == "In the data dictionary":
        shown = log[log["in_dictionary"]]
    elif only == "Found during analysis":
        shown = log[~log["in_dictionary"]]
    display = shown.drop(columns="in_dictionary").assign(rows_affected=lambda t: t["rows_affected"].map("{:,}".format))
    plain_table(
        display.rename(
            columns={"table": "Table", "column": "Column", "issue": "Issue", "rows_affected": "Rows", "decision": "Decision"}
        )
    )

    st.divider()
    st.subheader("Key assumptions")
    for assumption in ASSUMPTIONS:
        st.markdown(f"- {assumption}")
