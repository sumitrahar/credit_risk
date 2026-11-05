"""Turn the one-to-many tables into one row per loan account.

Join logic (the analysis dataset has exactly one row per account):
    loan_accounts  1 : 1    customers             on customer_id
    loan_accounts  1 : 12   repayment_history     aggregated per account
    loan_accounts  1 : n    bureau_credit         aggregated per account, limited by date
    customers      1 : n    bureau_credit         aggregated per customer, current view
    loan_accounts  1 : 0-n  collections_events    aggregated per account, left join
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import BEHAVIOUR_MONTHS

AVERAGE_DAYS_IN_MONTH = 30.44


def repayment_features(repayment: pd.DataFrame, months: int = BEHAVIOUR_MONTHS) -> pd.DataFrame:
    """Behaviour over the first `months` months. Fewer months are used for sensitivity tests."""
    r = repayment[repayment["payment_month"] <= months].sort_values(["account_id", "payment_month"])
    account = r["account_id"]
    missed = (r["payment_status"] == "Missed").astype(int)

    out = pd.DataFrame(
        {
            "rp_missed": missed.groupby(account).sum(),
            "rp_late": (r["payment_status"] == "Late").groupby(account).sum(),
            "rp_max_dpd": r.groupby("account_id")["days_past_due"].max(),
            "rp_mean_dpd": r.groupby("account_id")["days_past_due"].mean(),
        }
    )
    out["rp_pay_ratio"] = (
        r.groupby("account_id")["emi_paid"].sum() / r.groupby("account_id")["emi_due"].sum()
    )

    # Longest run of consecutive missed payments: a cumulative count of non-missed months
    # gives each run its own id, then the run length is a cumulative sum inside that id.
    run_id = (missed == 0).groupby(account).cumsum()
    out["rp_longest_missed_run"] = missed.groupby([account, run_id]).cumsum().groupby(account).max()

    midpoint = months // 2
    first_half = r[r["payment_month"] <= midpoint].groupby("account_id")["days_past_due"].mean()
    second_half = r[r["payment_month"] > midpoint].groupby("account_id")["days_past_due"].mean()
    out["rp_dpd_trend"] = second_half - first_half
    out["rp_recent_mean_dpd"] = (
        r[r["payment_month"] > months - 3].groupby("account_id")["days_past_due"].mean()
    )
    return out


def _aggregate_bureau_lines(lines: pd.DataFrame, key: str, prefix: str) -> pd.DataFrame:
    lines = lines.assign(
        _active=lines["account_status"] == "Active",
        _writeoff=lines["account_status"] == "Written-Off",
        _nonbank=lines["lender_category"].isin(["NBFC", "Fintech"]),
    )
    # Utilisation only counts open lines whose balance is known.
    open_known = lines["_active"] & lines["outstanding_amount"].notna()
    lines = lines.assign(
        _open_balance=lines["outstanding_amount"].where(open_known),
        _open_limit=lines["sanctioned_amount"].where(open_known),
    )
    grouped = lines.groupby(key)
    out = pd.DataFrame(
        {
            f"{prefix}_lines": grouped.size(),
            f"{prefix}_active": grouped["_active"].sum(),
            f"{prefix}_writeoffs": grouped["_writeoff"].sum(),
            f"{prefix}_max_dpd": grouped["days_past_due"].max(),
            f"{prefix}_mean_dpd": grouped["days_past_due"].mean(),
            f"{prefix}_outstanding": grouped["outstanding_amount"].sum(),
            f"{prefix}_nonbank_share": grouped["_nonbank"].mean(),
        }
    )
    limit = grouped["_open_limit"].sum()
    out[f"{prefix}_utilisation"] = (grouped["_open_balance"].sum() / limit).where(limit > 0)
    return out


def bureau_features_point_in_time(
    loans: pd.DataFrame,
    bureau: pd.DataFrame,
    offset_months: int = BEHAVIOUR_MONTHS,
) -> pd.DataFrame:
    """Bureau view for each loan, as it could have been seen at the scoring date.

    A line counts only if it was opened by origination + offset. If it was closed or written
    off after that date, it is treated as still Active then, with unknown balance and DPD.
    """
    accounts = loans[["account_id", "customer_id", "origination_date"]].copy()
    accounts["scoring_date"] = accounts["origination_date"] + pd.DateOffset(months=offset_months)

    lines = accounts.merge(bureau, on="customer_id", how="inner")
    lines = lines[lines["date_opened"] <= lines["scoring_date"]].copy()

    closed_later = lines["date_closed"] > lines["scoring_date"]
    lines.loc[closed_later, "account_status"] = "Active"
    lines.loc[closed_later, ["days_past_due", "outstanding_amount"]] = np.nan

    out = _aggregate_bureau_lines(lines, "account_id", "bu")
    age_days = (lines["scoring_date"] - lines["date_opened"]).dt.days
    out["bu_credit_age_months"] = age_days.groupby(lines["account_id"]).max() / AVERAGE_DAYS_IN_MONTH
    return out.reindex(loans["account_id"])


def bureau_features_all_lines(loans: pd.DataFrame, bureau: pd.DataFrame) -> pd.DataFrame:
    """Every line as reported, ignoring dates. Only used to show the size of the look-ahead effect."""
    per_customer = _aggregate_bureau_lines(bureau, "customer_id", "bu")
    out = per_customer.reindex(loans["customer_id"])
    out.index = loans["account_id"]
    return out


def bureau_current_view(customers: pd.DataFrame, bureau: pd.DataFrame) -> pd.DataFrame:
    """The latest position per customer. Used for monitoring and anomaly detection, not the model."""
    out = _aggregate_bureau_lines(bureau, "customer_id", "cur").reindex(customers["customer_id"])
    out["thin_file"] = out["cur_lines"].isna().astype(int)
    income = customers.set_index("customer_id")["annual_income"]
    out["cur_debt_to_income"] = out["cur_outstanding"] / income.reindex(out.index)
    return out


def collections_features(collections: pd.DataFrame) -> pd.DataFrame:
    grouped = collections.groupby("account_id")
    return pd.DataFrame(
        {
            "co_events": grouped.size(),
            "co_recovered": grouped["amount_recovered"].sum(),
            "co_events_before_origination": grouped["event_before_origination"].sum(),
        }
    )


def finalise_bureau_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Fill what is truly zero, and turn "unknown" into explicit flags.

    No visible line means a true zero for counts and balances. Rates and DPD stay missing,
    because unknown is different from zero. The model imputes them and reads these flags.
    """
    out = df.copy()
    for column in ["bu_lines", "bu_active", "bu_writeoffs", "bu_outstanding"]:
        out[column] = out[column].fillna(0)
    out["bu_no_visible_lines"] = (out["bu_lines"] == 0).astype(int)
    out["bu_dpd_unknown"] = out["bu_max_dpd"].isna().astype(int)
    out["bu_no_open_balance"] = out["bu_utilisation"].isna().astype(int)
    out["bu_debt_to_income"] = out["bu_outstanding"] / out["annual_income"]
    return out


def build_analysis_dataset(clean: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Produce the single clean dataset: one row per loan account."""
    loans, customers = clean["loans"], clean["customers"]
    repayment, bureau = clean["repayment"], clean["bureau"]

    df = loans.merge(customers, on="customer_id", how="left", validate="many_to_one")
    df = df.merge(
        repayment_features(repayment), left_on="account_id", right_index=True,
        how="left", validate="one_to_one",
    )
    df = df.merge(
        bureau_features_point_in_time(loans, bureau), left_on="account_id", right_index=True,
        how="left", validate="one_to_one",
    )
    df = df.merge(
        bureau_current_view(customers, bureau), left_on="customer_id", right_index=True,
        how="left", validate="many_to_one",
    )
    df = df.merge(
        collections_features(clean["collections"]), left_on="account_id", right_index=True,
        how="left", validate="one_to_one",
    )

    latest = repayment[repayment["payment_month"] == BEHAVIOUR_MONTHS]
    df["latest_balance"] = df["account_id"].map(latest.set_index("account_id")["outstanding_balance"])

    df = finalise_bureau_columns(df)
    for column in ["co_events", "co_recovered", "co_events_before_origination"]:
        df[column] = df[column].fillna(0)

    df["loan_to_income"] = df["loan_amount"] / df["annual_income"]
    df["installment_to_income"] = df["emi"] * 12 / df["annual_income"]
    df["defaulted_amount"] = df["loan_amount"] * df["target_default"]

    if len(df) != len(loans) or df["account_id"].duplicated().any():
        raise ValueError("Joins changed the number of accounts")
    return df
