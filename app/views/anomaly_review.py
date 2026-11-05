"""Anomaly Review view: the list of flagged accounts, why each was flagged, and the decision."""
from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from components import feature_statement, money, open_account, percent, plain, plain_table, repayment_timeline, show_chart
from src.anomaly import DETECTORS
from src.decisions import VALID_DECISIONS, all_decisions, detector_outcomes, latest_decisions, record_decision

SCOPES = ["Review list", "All accounts with a signal"]
STATUS_OPTIONS = ["Not yet reviewed", "All", "Confirmed", "Dismissed"]


def _review_table(queue: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Rank": queue["review_rank"],
            "Account": queue["account_id"],
            "Signals": queue["n_detectors"],
            "Why (first reason)": queue["reasons_json"].map(lambda v: (json.loads(v) or [""])[0]),
            "Loan type": queue["loan_type"],
            "Latest balance": queue["latest_balance"],
            "Predicted default": queue["pd_score"] * 100,
            "Decision": queue["decision"].fillna("Not reviewed"),
        }
    )


def _detail(row: pd.Series, context: dict) -> None:
    account_id = row["account_id"]
    st.subheader(f"Why {account_id} was flagged")
    fired = json.loads(row["detectors_json"])
    reasons = json.loads(row["reasons_json"])

    left, right = st.columns([3, 2])
    with left:
        for detector, reason in zip(fired, reasons):
            info = DETECTORS[detector]
            st.markdown(f"- **{info['name']}** ({info['compared_with'].lower()}): {plain(reason)}")
        months = context["repayment"]
        months = months[months["account_id"] == account_id].sort_values("payment_month")
        st.markdown("**Own history against peers**")
        st.caption(
            f"Average days past due: {row['first_half_mean_dpd']:.1f} in months 1-6, {row['second_half_mean_dpd']:.1f} in months 7-12. "
            f"The line is the median for {row['peer_group']} peers."
        )
        show_chart(repayment_timeline(months, row["peer_median_dpd"], height=270), key=f"review_timeline_{account_id}")
    with right:
        columns = st.columns(2)
        columns[0].metric("Predicted default", f"{row['pd_score'] * 100:.1f}%")
        columns[1].metric("Latest balance", money(row["latest_balance"]))
        columns = st.columns(2)
        columns[0].metric("Unusual percentile", f"{min(row['anomaly_score'], 99.9):.1f}", help="Isolation Forest rank across all accounts: more unusual than this percentage of them.")
        columns[1].metric("Risk band", row["risk_band"])
        st.caption(plain(f"{row['loan_type']}, {row['employment_type']}, income band {row['income_band']}."))

        current = row["decision"] if pd.notna(row["decision"]) else None
        if current:
            st.markdown(f"Latest decision: **{current}**" + (f". Note: {plain(row['note'])}" if row["note"] else ""))
        with st.form(f"decision_form_{account_id}", clear_on_submit=True):
            decision = st.radio("Your decision", VALID_DECISIONS, horizontal=True, index=None, help="Confirm: this needs follow-up. Dismiss: not a real concern.")
            note = st.text_input("Note", placeholder="What did you see?")
            saved = st.form_submit_button("Record decision", type="primary")
        if saved:
            if decision is None:
                st.warning("Choose Confirm or Dismiss first.")
            else:
                record_decision(account_id, decision, note, fired)
                st.session_state["review_message"] = f"Recorded {decision} for {account_id}."
                st.session_state.pop("review_account", None)
                st.rerun()
        if st.button("Open full account view", key=f"open_account_{account_id}"):
            open_account(account_id)


def _detector_table(report: dict) -> None:
    st.markdown("**Detectors**")
    st.caption(
        "Default rate is only a sanity check: the target is never used to fit a detector, and unusual does not have to mean default. "
        "Two detectors show no lift on their own, which is why the list needs two signals to agree."
    )
    outcomes = detector_outcomes().set_index("detector") if not latest_decisions().empty else pd.DataFrame()
    rows = []
    for item in report["detectors"]:
        decided = outcomes.loc[item["detector"]] if item["detector"] in outcomes.index else None
        rows.append(
            {
                "Detector": item["name"],
                "Compared with": item["compared_with"],
                "Rule": item["rule"],
                "Accounts": f"{item['accounts']:,}",
                "In review list": f"{item['in_review_list']:,}",
                "Default rate": percent(item["default_rate"]),
                "Confirmed": str(int(decided["Confirm"])) if decided is not None else "0",
                "Dismissed": str(int(decided["Dismiss"])) if decided is not None else "0",
            }
        )
    plain_table(pd.DataFrame(rows))


def render(context: dict) -> None:
    report = context["anomaly_report"]
    frame = context["filtered"]
    st.title("Anomaly Review")
    st.caption("Accounts whose repayment or bureau pattern is unusual against their own history or their peers.")

    feature_statement(
        trigger=(
            f"The batch run checks every account with {len(DETECTORS)} detectors. An account joins the review list when two or more agree: "
            f"{report['accounts_flagged']:,} accounts, {percent(report['share_flagged'])} of the portfolio."
        ),
        false_positive=(
            f"A one-off payment-date slip, an agreed hardship plan or a data error looks unusual but is not a real risk. "
            f"{percent(1 - report['flagged_default_rate'])} of listed accounts did not default."
        ),
        false_negative=(
            f"A borrower who is steadily poor but normal for their peers, or a defaulter with a clean record. "
            f"{percent(1 - report['share_of_defaults_in_list'])} of defaults are not on the list: it finds unusual patterns and does not replace the model."
        ),
        human_step="The analyst reads the reasons and records Confirm or Dismiss with a note. Nothing reaches a customer without that decision, and dismiss rates per detector feed threshold reviews.",
    )
    st.write("")

    if message := st.session_state.pop("review_message", None):
        st.toast(message, icon=":material/check_circle:")

    decisions = latest_decisions()
    merged = frame.merge(decisions[["decision", "note"]], left_on="account_id", right_index=True, how="left")
    in_list = merged[merged["flagged"] == 1]

    columns = st.columns(5)
    columns[0].metric("In review list", f"{len(in_list):,}", help="Within the current sidebar filters.")
    columns[1].metric(
        "Default rate on list", percent(report["flagged_default_rate"]),
        help=f"For the whole list, compared with {percent(report['base_default_rate'])} across the portfolio. A sanity check only.",
    )
    columns[2].metric("Reviewed", f"{in_list['decision'].notna().sum():,}")
    columns[3].metric("Confirmed", f"{(in_list['decision'] == 'Confirm').sum():,}")
    columns[4].metric("Dismissed", f"{(in_list['decision'] == 'Dismiss').sum():,}")

    left, right = st.columns(2)
    # The Account view can preset these when it links here, so they live in session state.
    st.session_state.setdefault("review_scope", SCOPES[0])
    st.session_state.setdefault("review_status", STATUS_OPTIONS[0])
    scope = left.segmented_control("Accounts", SCOPES, key="review_scope", help="All accounts with a signal adds those where only one detector fired.") or SCOPES[0]
    status = right.segmented_control("Decision status", STATUS_OPTIONS, key="review_status") or STATUS_OPTIONS[0]

    queue = in_list if scope == SCOPES[0] else merged[merged["n_detectors"] >= 1]
    if status == "Not yet reviewed":
        queue = queue[queue["decision"].isna()]
    elif status == "Confirmed":
        queue = queue[queue["decision"] == "Confirm"]
    elif status == "Dismissed":
        queue = queue[queue["decision"] == "Dismiss"]
    queue = queue.sort_values(["n_detectors", "anomaly_score", "latest_balance"], ascending=False).reset_index(drop=True)

    if queue.empty:
        st.info("No accounts in this selection. Change the filters or the decision status.", icon=":material/info:")
    else:
        st.caption(f"{len(queue):,} accounts. Select a row to see why it was flagged and record a decision.")
        event = st.dataframe(
            _review_table(queue), width="stretch", hide_index=True, height=340, on_select="rerun",
            selection_mode="single-row", key=f"review_table_{scope}_{status}",
            column_config={
                "Latest balance": st.column_config.NumberColumn(format="$%,.0f"),
                "Predicted default": st.column_config.NumberColumn(format="%.1f%%"),
                "Why (first reason)": st.column_config.TextColumn(width="medium"),
            },
        )
        rows = event.selection.rows
        wanted = st.session_state.get("review_account")
        if rows:
            current = queue.iloc[rows[0]]
        elif wanted and (queue["account_id"] == wanted).any():
            current = queue[queue["account_id"] == wanted].iloc[0]
        else:
            current = queue.iloc[0]
        st.divider()
        _detail(current, context)

    st.divider()
    _detector_table(report)

    st.divider()
    st.markdown("**Decision log**")
    st.caption("Saved in a database on this app's own disk. Free hosting can reset it on a restart or redeploy, so download a copy to keep it.")
    log = all_decisions()
    if log.empty:
        st.caption("No decisions recorded yet.")
    else:
        display = log.assign(detectors=log["detectors"].map(lambda v: ", ".join(DETECTORS[d]["name"] for d in json.loads(v))))
        st.dataframe(
            display.rename(columns={"account_id": "Account", "decision": "Decision", "note": "Note", "detectors": "Detectors", "decided_at": "When"}),
            width="stretch", hide_index=True,
        )
        st.download_button("Download decisions as CSV", display.to_csv(index=False), file_name="anomaly_decisions.csv", mime="text/csv")
