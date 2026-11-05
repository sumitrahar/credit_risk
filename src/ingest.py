"""Load, validate and clean the five raw tables, recording every data-quality decision.

Nothing here is silently dropped: each issue is counted from the data and written to
data/processed/dq_log.csv, which the dashboard and README both read from.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from src.config import (
    AGE_BINS,
    AGE_LABELS,
    BEHAVIOUR_MONTHS,
    INCOME_BINS,
    INCOME_LABELS,
    RAW_DIR,
    SNAPSHOT_DATE,
)


@dataclass
class DQEntry:
    table: str
    column: str
    issue: str
    rows_affected: int
    decision: str
    in_dictionary: bool


@dataclass
class DQLog:
    entries: list[DQEntry] = field(default_factory=list)

    def add(self, table, column, issue, rows_affected, decision, in_dictionary=False):
        self.entries.append(
            DQEntry(table, column, issue, int(rows_affected), decision, in_dictionary)
        )

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([asdict(e) for e in self.entries])


def load_raw(raw_dir=RAW_DIR) -> dict[str, pd.DataFrame]:
    return {
        "customers": pd.read_csv(raw_dir / "customers.csv", parse_dates=["customer_since"]),
        "loans": pd.read_csv(raw_dir / "loan_accounts.csv", parse_dates=["origination_date"]),
        "bureau": pd.read_csv(
            raw_dir / "bureau_credit.csv", parse_dates=["date_opened", "date_closed"]
        ),
        "repayment": pd.read_csv(raw_dir / "repayment_history.csv"),
        "collections": pd.read_csv(raw_dir / "collections_events.csv", parse_dates=["event_date"]),
    }


def check_integrity(raw: dict[str, pd.DataFrame]) -> None:
    """Stop the build if keys are broken, because every later join relies on them."""
    primary_keys = {
        "customers": ["customer_id"],
        "loans": ["account_id"],
        "bureau": ["bureau_id"],
        "collections": ["collection_event_id"],
        "repayment": ["account_id", "payment_month"],
    }
    for table, key in primary_keys.items():
        if raw[table].duplicated(key).any():
            raise ValueError(f"{table}: duplicate key {key}")

    foreign_keys = [
        ("loans", "customer_id", "customers"),
        ("bureau", "customer_id", "customers"),
        ("repayment", "account_id", "loans"),
        ("collections", "account_id", "loans"),
    ]
    for child, column, parent in foreign_keys:
        orphans = ~raw[child][column].isin(raw[parent][column])
        if orphans.any():
            raise ValueError(f"{child}.{column}: {orphans.sum()} rows have no match in {parent}")


def monthly_payment(principal: float, annual_rate_pct: float, months: int) -> float:
    """Standard loan payment (annuity) formula."""
    r = annual_rate_pct / 1200
    return principal * r * (1 + r) ** months / ((1 + r) ** months - 1)


def solve_rate(principal: float, months: int, payment: float) -> float:
    """Invert the annuity formula: find the annual rate that produces this payment."""
    return brentq(lambda rate: monthly_payment(principal, rate, months) - payment, 0.01, 100)


def clean_customers(customers: pd.DataFrame, log: DQLog) -> pd.DataFrame:
    df = customers.copy()

    coded = df["gender"].isin(["M", "F"])
    df["gender"] = df["gender"].replace({"M": "Male", "F": "Female"})
    log.add(
        "customers", "gender", "Gender coded as M/F instead of Male/Female", coded.sum(),
        "Mapped to Male/Female. Gender is kept for profiling only and is not a model input.", True,
    )

    # Cap implausible known values first, so they do not distort the medians used below.
    max_years_employed = (df["age"] - 16).clip(lower=0)
    over_employed = df["years_employed"] > max_years_employed
    df["years_employed_capped"] = over_employed.astype(int)
    df["years_employed"] = df["years_employed"].where(~over_employed, max_years_employed)
    log.add(
        "customers", "years_employed", "Years employed exceeds age minus 16", over_employed.sum(),
        "Capped at age minus 16 and flagged in years_employed_capped.",
    )

    over_residence = df["years_at_residence"] > df["age"]
    df["years_at_residence"] = df["years_at_residence"].where(~over_residence, df["age"])
    log.add(
        "customers", "years_at_residence", "Years at residence exceeds age", over_residence.sum(),
        "Capped at age.",
    )

    unemployed = df["employment_type"] == "Unemployed"
    known_unemployed = df.loc[unemployed, "years_employed"].dropna()
    if not (known_unemployed == 0).all():
        raise ValueError("Expected every known years_employed for Unemployed customers to be 0")

    for column in ["years_employed", "num_dependents", "years_at_residence"]:
        missing = df[column].isna()
        df[f"{column}_missing"] = missing.astype(int)
        if column == "years_employed":
            df.loc[missing & unemployed, column] = 0.0
            decision = (
                "Unemployed customers set to 0, matching every known value. Others use the "
                "median for their employment type, with a missing flag."
            )
        else:
            decision = "Median for the customer's employment type, with a missing flag."
        df[column] = df[column].fillna(df.groupby("employment_type")[column].transform("median"))
        log.add("customers", column, "Missing values with no visible pattern", missing.sum(), decision, True)

    df["num_dependents"] = df["num_dependents"].round()
    df["years_employed"] = df["years_employed"].clip(upper=max_years_employed)
    df["years_at_residence"] = df["years_at_residence"].clip(upper=df["age"])

    log.add(
        "customers", "occupation_sector", "Unemployed customers still have a sector",
        unemployed.sum(), "Kept and read as the customer's last sector.",
    )
    for column, floor in [("age", 21), ("annual_income", 12_000)]:
        log.add(
            "customers", column, f"Values clipped at a floor of {floor:,}",
            (df[column] == floor).sum(), "Left as provided and noted; the true value is unknown.",
        )

    df["income_band"] = pd.cut(
        df["annual_income"], INCOME_BINS, labels=INCOME_LABELS, right=False
    ).astype(str)
    df["age_band"] = pd.cut(df["age"], AGE_BINS, labels=AGE_LABELS).astype(str)
    return df


def clean_repayment(repayment: pd.DataFrame, loans: pd.DataFrame, log: DQLog) -> pd.DataFrame:
    df = repayment.sort_values(["account_id", "payment_month"]).reset_index(drop=True)

    months_per_account = df.groupby("account_id").size()
    if not (months_per_account == BEHAVIOUR_MONTHS).all():
        raise ValueError(f"Expected exactly {BEHAVIOUR_MONTHS} months per account")
    log.add(
        "repayment_history", "payment_month",
        "Every account has exactly 12 months, whatever its term (12 to 360 months), and months "
        "have no calendar date",
        len(months_per_account),
        "Treated as a fixed 12-month behaviour window after origination.",
    )

    previous = df.groupby("account_id")["outstanding_balance"].shift()
    paid = (df["emi_paid"] > 0) & previous.notna()
    share_of_payment = ((previous - df["outstanding_balance"])[paid] / df.loc[paid, "emi_paid"]).median()
    twelve_month = loans.loc[loans["tenure_months"] == 12, ["account_id", "loan_amount"]]
    final = df[df["payment_month"] == 12].merge(twelve_month, on="account_id")
    remaining = (final["outstanding_balance"] / final["loan_amount"]).median()
    log.add(
        "repayment_history", "outstanding_balance",
        f"Balance falls by exactly {share_of_payment:.0%} of each payment regardless of the "
        f"interest rate, so 12-month loans still show a median {remaining:.0%} balance at month 12",
        len(final),
        "Not corrected. Latest balance is shown as approximate, next to the originated amount, "
        "and is not a model input.",
    )

    missed_now = df["payment_status"] == "Missed"
    next_dpd = df.groupby("account_id")["days_past_due"].shift(-1)
    no_roll = (next_dpd[missed_now].dropna() == 0).mean()
    log.add(
        "repayment_history", "days_past_due",
        f"Days past due does not accumulate: {no_roll:.0%} of months following a missed payment "
        f"return to 0, and the highest value is {df['days_past_due'].max()}, so no account reaches 90+",
        missed_now.sum(),
        "Used as reported. It also means target_default cannot be derived from this table.",
    )
    return df


def clean_loans(
    loans: pd.DataFrame, repayment: pd.DataFrame, customers: pd.DataFrame, log: DQLog
) -> pd.DataFrame:
    df = loans.copy()

    # The installment is an exact function of amount, term and rate, so a missing rate can
    # be solved rather than guessed. First prove that on rows where the rate is known.
    payment = df["account_id"].map(
        repayment.loc[repayment["payment_month"] == 1].set_index("account_id")["emi_due"]
    )
    known = df["interest_rate_pct"].notna()
    solved_known = [
        solve_rate(amount, months, emi)
        for amount, months, emi in zip(
            df.loc[known, "loan_amount"], df.loc[known, "tenure_months"], payment[known]
        )
    ]
    worst_error = float(np.abs(np.array(solved_known) - df.loc[known, "interest_rate_pct"]).max())

    missing = ~known
    df["interest_rate_recovered"] = missing.astype(int)
    df.loc[missing, "interest_rate_pct"] = [
        round(solve_rate(amount, months, emi), 2)
        for amount, months, emi in zip(
            df.loc[missing, "loan_amount"], df.loc[missing, "tenure_months"], payment[missing]
        )
    ]
    log.add(
        "loan_accounts", "interest_rate_pct", "Missing interest rate", missing.sum(),
        "Recovered exactly, not imputed: the installment follows the standard loan payment "
        f"formula, so the rate can be solved from it. Tested on {known.sum():,} known rates, the "
        f"largest error was {worst_error:.3f} percentage points.",
        True,
    )

    df["emi"] = payment
    df["origination_year"] = df["origination_date"].dt.year

    since = df["customer_id"].map(customers.set_index("customer_id")["customer_since"])
    df["originated_before_customer_since"] = (df["origination_date"] < since).astype(int)
    log.add(
        "customers", "customer_since", "Loan originated before the customer relationship began",
        df["originated_before_customer_since"].sum(),
        "Field treated as unreliable: kept for display, but no tenure-with-bank feature is built from it.",
    )

    by_purpose = df.groupby("loan_purpose")["target_default"].mean()
    log.add(
        "loan_accounts", "loan_purpose",
        "Purpose is unrelated to loan type (for example Mortgage for Wedding), and default rate "
        f"is flat across purposes ({by_purpose.min():.1%} to {by_purpose.max():.1%})",
        len(df), "Kept for display only; not a model input.",
    )
    log.add(
        "loan_accounts", "disbursal_channel",
        "Dictionary says Partner-DSA, but the data says Partner/DSA",
        (df["disbursal_channel"] == "Partner/DSA").sum(),
        "Used the label as it appears in the data.",
    )
    log.add(
        "loan_accounts", "loan_amount", "Values clipped at a floor of 1,000",
        (df["loan_amount"] == 1000).sum(), "Left as provided and noted.",
    )

    missed = (repayment["payment_status"] == "Missed").groupby(repayment["account_id"]).sum()
    on_time = (repayment["payment_status"] == "On-Time").groupby(repayment["account_id"]).sum()
    defaulted = df.set_index("account_id")["target_default"] == 1
    never_missed = (missed.reindex(defaulted.index) == 0) & defaulted
    always_on_time = (on_time.reindex(defaulted.index) == BEHAVIOUR_MONTHS) & defaulted
    log.add(
        "loan_accounts", "target_default",
        f"The target has no date and does not follow from repayment: {never_missed.sum():,} of "
        f"{defaulted.sum():,} defaulted accounts ({never_missed.sum() / defaulted.sum():.1%}) never "
        f"missed a payment, and {always_on_time.sum():,} were on time in all 12 months",
        never_missed.sum(),
        "Assumed that default is observed after month 12, which makes the model a month-12 "
        "behaviour score. Sensitivity with 3 and 6 months of history is reported.",
    )
    return df


def clean_bureau(
    bureau: pd.DataFrame, loans: pd.DataFrame, customers: pd.DataFrame, log: DQLog
) -> pd.DataFrame:
    df = bureau.copy()

    closed = df["account_status"] == "Closed"
    missing_dpd = df["days_past_due"].isna()
    if not (df.loc[closed & ~missing_dpd, "days_past_due"] == 0).all():
        raise ValueError("Expected every known days_past_due on Closed lines to be 0")
    df.loc[closed & missing_dpd, "days_past_due"] = 0.0
    log.add(
        "bureau_credit", "days_past_due", "Missing days past due on Closed lines",
        (closed & missing_dpd).sum(), "Set to 0, matching every known value on Closed lines.", True,
    )
    df["dpd_missing"] = df["days_past_due"].isna().astype(int)
    log.add(
        "bureau_credit", "days_past_due", "Missing days past due on Active or Written-Off lines",
        df["dpd_missing"].sum(),
        "Left missing and flagged; averages and maximums ignore these rows. The model fills the "
        "gap with a median and also reads an explicit 'unknown' flag.",
        True,
    )

    if not (df["date_closed"].isna() == (df["account_status"] == "Active")).all():
        raise ValueError("Expected date_closed to be null exactly on Active lines")
    log.add(
        "bureau_credit", "date_closed", "Null date_closed", df["date_closed"].isna().sum(),
        "These are exactly the Active lines, so the null is expected. Left as is.", True,
    )

    df["date_closed_after_snapshot"] = (df["date_closed"] > SNAPSHOT_DATE).astype(int)
    log.add(
        "bureau_credit", "date_closed",
        f"Closure dated after the assumed {SNAPSHOT_DATE.date()} snapshot, as late as "
        f"{df['date_closed'].max().date()}",
        df["date_closed_after_snapshot"].sum(),
        "Flagged and the date is not trusted. The point-in-time rule treats these lines as still open.",
    )

    pairs = loans[["customer_id", "origination_date"]].merge(df, on="customer_id")
    after_origination = pairs["date_opened"] > pairs["origination_date"]
    log.add(
        "bureau_credit", "date_opened",
        f"Look-ahead risk: for {after_origination.mean():.1%} of line-and-loan pairs, the bureau "
        "line opened after that loan's origination",
        after_origination.sum(),
        "Model features use only lines opened by origination + 12 months, and lines closed or "
        "written off after that date are rewound to Active with unknown balance.",
    )
    log.add(
        "bureau_credit", "account_status",
        "Status, balance and days past due on Active lines are as at last reporting, with no "
        "reporting date",
        (df["account_status"] == "Active").sum(),
        "Used as reported. This look-ahead cannot be removed with the data provided, and is "
        "listed as an assumption.",
    )
    log.add(
        "bureau_credit", "sanctioned_amount", "Values clipped at a floor of 500",
        (df["sanctioned_amount"] == 500).sum(), "Left as provided and noted.",
    )

    no_record = ~customers["customer_id"].isin(df["customer_id"])
    log.add(
        "customers", "customer_id", "Customer has no bureau record at all", no_record.sum(),
        "Left join with a thin_file flag; not filled with zeros, because no record is different "
        "from a clean record.",
    )
    return df


def clean_collections(collections: pd.DataFrame, loans: pd.DataFrame, log: DQLog) -> pd.DataFrame:
    df = collections.copy()
    info = loans.set_index("account_id")
    df["event_before_origination"] = (
        df["event_date"] < df["account_id"].map(info["origination_date"])
    ).astype(int)
    log.add(
        "collections_events", "event_date", "Collections event dated before the loan was originated",
        df["event_before_origination"].sum(),
        "Flagged and shown in the account view. Collections are never model inputs, because they "
        "are a consequence of delinquency rather than a predictor of it.",
    )

    received = df["outcome"] == "Payment Received"
    log.add(
        "collections_events", "outcome", "'Payment Received' with $0 recovered",
        (received & (df["amount_recovered"] == 0)).sum(), "Reported as provided and noted.",
    )
    no_payment = df["outcome"].isin(["No Response", "Disputed", "Escalated"])
    log.add(
        "collections_events", "amount_recovered",
        "Money recovered on an outcome that implies no payment",
        (no_payment & (df["amount_recovered"] > 0)).sum(), "Reported as provided and noted.",
    )
    referral = df["action_type"] == "Written-Off Referral"
    target = df["account_id"].map(info["target_default"])
    log.add(
        "collections_events", "action_type", "'Written-Off Referral' on a non-defaulted account",
        (referral & (target == 0)).sum(), "Reported as provided and noted.",
    )
    recovered = df.groupby("account_id")["amount_recovered"].sum()
    over_recovered = recovered > info["loan_amount"].reindex(recovered.index)
    log.add(
        "collections_events", "amount_recovered", "Total recovered exceeds the loan amount",
        over_recovered.sum(), "Reported as provided and noted.",
    )
    log.add(
        "collections_events", "action_type", "Dictionary says SMS-Email, but the data says SMS/Email Notice",
        (df["action_type"] == "SMS/Email Notice").sum(), "Used the label as it appears in the data.",
    )
    return df


def run_ingest(raw: dict[str, pd.DataFrame] | None = None) -> tuple[dict[str, pd.DataFrame], DQLog]:
    raw = raw if raw is not None else load_raw()
    check_integrity(raw)
    log = DQLog()
    customers = clean_customers(raw["customers"], log)
    repayment = clean_repayment(raw["repayment"], raw["loans"], log)
    loans = clean_loans(raw["loans"], repayment, customers, log)
    bureau = clean_bureau(raw["bureau"], loans, customers, log)
    collections = clean_collections(raw["collections"], loans, log)
    return (
        {
            "customers": customers,
            "loans": loans,
            "bureau": bureau,
            "repayment": repayment,
            "collections": collections,
        },
        log,
    )
