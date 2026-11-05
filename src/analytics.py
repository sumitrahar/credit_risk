"""Portfolio cuts and the written insights.

Every figure in an insight is computed from the data at build time, so the text cannot
drift away from the numbers on screen.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import DIMENSION_ORDER


def order_dimension(table: pd.DataFrame, dimension: str) -> pd.DataFrame:
    """Sort ordered dimensions by their natural order, everything else by default rate."""
    if dimension in DIMENSION_ORDER:
        order = [v for v in DIMENSION_ORDER[dimension] if v in table.index]
        return table.loc[order]
    if dimension in ("origination_year", "tenure_months"):
        return table.sort_index()
    return table.sort_values("default_rate", ascending=False)


def segment_summary(df: pd.DataFrame, dimension: str) -> pd.DataFrame:
    """Default rate and exposure for each value of one dimension."""
    table = df.groupby(dimension).agg(
        accounts=("account_id", "size"),
        defaults=("target_default", "sum"),
        originated=("loan_amount", "sum"),
        defaulted_amount=("defaulted_amount", "sum"),
        latest_balance=("latest_balance", "sum"),
    )
    table["default_rate"] = table["defaults"] / table["accounts"]
    table["share_of_originated"] = table["originated"] / table["originated"].sum()
    table["share_of_defaulted_amount"] = table["defaulted_amount"] / table["defaulted_amount"].sum()
    return order_dimension(table, dimension)


def cross_tab(df: pd.DataFrame, rows: str, columns: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Two-way default rate, with the account count behind each cell."""
    rates = df.pivot_table(index=rows, columns=columns, values="target_default", aggfunc="mean")
    counts = df.pivot_table(index=rows, columns=columns, values="target_default", aggfunc="size")
    for axis, dimension in ((0, rows), (1, columns)):
        if dimension in DIMENSION_ORDER:
            present = rates.index if axis == 0 else rates.columns
            order = [v for v in DIMENSION_ORDER[dimension] if v in present]
            rates = rates.reindex(order, axis=axis)
            counts = counts.reindex(order, axis=axis)
    return rates, counts.fillna(0).astype(int)


def month_on_book_curve(repayment: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    """Share of accounts late or missed in each month after origination."""
    selected = repayment[repayment["account_id"].isin(df["account_id"])]
    grouped = selected.groupby("payment_month")
    return pd.DataFrame(
        {
            "missed_rate": (selected["payment_status"] == "Missed").groupby(selected["payment_month"]).mean(),
            "late_rate": (selected["payment_status"] == "Late").groupby(selected["payment_month"]).mean(),
            "mean_dpd": grouped["days_past_due"].mean(),
        }
    )


def _rate(df: pd.DataFrame, mask: pd.Series) -> float:
    return float(df.loc[mask, "target_default"].mean())


def compute_insights(df: pd.DataFrame, collections: pd.DataFrame) -> list[dict]:
    """Return the five portfolio insights, each with its evidence and a "so what"."""
    total_defaulted = df["defaulted_amount"].sum()

    # 1. Mortgage concentration
    mortgage = df["loan_type"] == "Mortgage"
    by_type = df.groupby("loan_type")["target_default"].mean()
    others = by_type.drop("Mortgage")
    tenor_240 = _rate(df, mortgage & (df["tenure_months"] == 240))
    insight_1 = {
        "id": "mortgage_concentration",
        "title": "Mortgages are the concentration risk",
        "top3": True,
        "finding": (
            f"Mortgages are {mortgage.mean():.1%} of accounts but "
            f"{df.loc[mortgage, 'loan_amount'].sum() / df['loan_amount'].sum():.1%} of originated exposure and "
            f"{df.loc[mortgage, 'defaulted_amount'].sum() / total_defaulted:.1%} of defaulted dollars. Their "
            f"{by_type['Mortgage']:.1%} default rate is the highest of any product "
            f"(others: {others.min():.1%} to {others.max():.1%}) even though they are secured, and "
            f"240-month mortgages reach {tenor_240:.1%}."
        ),
        "so_what": (
            "Review mortgages first and tighten long-term underwriting. Collateral values are "
            "needed from the bank, because the actual loss cannot be sized without them."
        ),
    }

    # 2. Income x employment cross-cut
    bands = df.groupby("income_band")["target_default"].mean()
    unemployed = df["employment_type"] == "Unemployed"
    contract = df["employment_type"] == "Contract"
    low = df["income_band"] == "<$40k"
    insight_2 = {
        "id": "income_employment",
        "title": "Income and employment together separate risk",
        "top3": True,
        "finding": (
            f"Default falls from {bands['<$40k']:.1%} under $40k to {bands['$100k+']:.1%} at $100k+. "
            f"Unemployed borrowers are {unemployed.mean():.1%} of accounts but "
            f"{df.loc[df['target_default'] == 1, 'employment_type'].eq('Unemployed').mean():.1%} of defaults. "
            f"Unemployed mortgages default at {_rate(df, unemployed & mortgage):.1%} "
            f"({int((unemployed & mortgage).sum())} accounts, "
            f"${df.loc[unemployed & mortgage, 'loan_amount'].sum() / 1e6:.1f}M). Contract borrowers run "
            f"{_rate(df, contract & low):.1%} under $40k but {_rate(df, contract & (df['income_band'] == '$60-80k')):.1%} at $60-80k."
        ),
        "so_what": (
            "Apply a hard policy rule to mortgages for unemployed applicants, and assess contract "
            "workers by income rather than treating them as one group."
        ),
    }

    # 3. Pricing against risk
    rates_by_band = df.groupby("income_band")["interest_rate_pct"].mean()
    personal = df[df["loan_type"] == "Personal Loan"]
    correlation = personal["interest_rate_pct"].corr(personal["annual_income"])
    insight_3 = {
        "id": "pricing",
        "title": "Pricing does not reflect income risk",
        "top3": False,
        "finding": (
            f"The average interest rate is {rates_by_band.min():.1f}% to {rates_by_band.max():.1f}% in every "
            f"income band, while default rate runs from {bands.min():.1%} to {bands.max():.1%}. Within "
            f"personal loans, rate and income are uncorrelated (r = {correlation:.2f})."
        ),
        "so_what": (
            "The strongest risk driver in the portfolio is not priced in, so low-income borrowers "
            "are under-charged for their risk. A risk-based pricing review is worth testing."
        ),
    }

    # 4. Delinquency alerts and collections
    defaulted = df["target_default"] == 1
    never_missed = (df["rp_missed"] == 0) & defaulted
    has_collections = df["co_events"] > 0
    recovered_on_defaults = df.loc[defaulted, "co_recovered"].sum()
    insight_4 = {
        "id": "alerts_collections",
        "title": "Delinquency alerts miss nearly half of defaults, and collections are not targeted",
        "top3": True,
        "finding": (
            f"{never_missed.sum() / defaulted.sum():.1%} of defaulters never missed a payment in 12 months. "
            f"Only {(has_collections & defaulted).sum() / defaulted.sum():.1%} of defaulted accounts had any "
            f"collections action, while {int((has_collections & ~defaulted).sum()):,} non-defaulted accounts did. "
            f"Recoveries equal {recovered_on_defaults / total_defaulted:.2%} of defaulted loan amounts."
        ),
        "so_what": (
            "Prioritise collections by model score rather than by missed payments alone, and audit "
            f"the collections data first: {collections['event_before_origination'].mean():.1%} of events "
            "are dated before the loan began."
        ),
    }

    # 5. Dimensions that do and do not matter
    regions = df.groupby("region")["target_default"].mean()
    ages = df.groupby("age_band")["target_default"].mean()
    flat = {
        label: df.groupby(column)["target_default"].mean()
        for label, column in [
            ("channel", "disbursal_channel"),
            ("purpose", "loan_purpose"),
            ("sector", "occupation_sector"),
        ]
    }
    flat_text = ", ".join(f"{label} {s.min():.1%}-{s.max():.1%}" for label, s in flat.items())
    age_by_product = df.pivot_table(
        index="loan_type", columns="age_band", values="target_default", aggfunc="mean"
    )
    holds_everywhere = bool((age_by_product["21-25"] > age_by_product["56+"]).all())
    insight_5 = {
        "id": "region_age",
        "title": "Region is not a useful lever; age is, with a caveat",
        "top3": False,
        "finding": (
            f"Regions span only {regions.min():.1%} to {regions.max():.1%} ({flat_text}). Age falls from "
            f"{ages['21-25']:.1%} at 21-25 to {ages['56+']:.1%} at 56+"
            + (", and the gradient holds within every product." if holds_everywhere else ".")
        ),
        "so_what": (
            "Drop region-based risk strategy. Age predicts default but raises fair-lending concerns, "
            "so it needs compliance review before any use in decisions."
        ),
    }
    return [insight_1, insight_2, insight_3, insight_4, insight_5]


def portfolio_totals(df: pd.DataFrame) -> dict:
    return {
        "accounts": int(len(df)),
        "customers": int(df["customer_id"].nunique()),
        "defaults": int(df["target_default"].sum()),
        "default_rate": float(df["target_default"].mean()) if len(df) else np.nan,
        "originated": float(df["loan_amount"].sum()),
        "defaulted_amount": float(df["defaulted_amount"].sum()),
        "latest_balance": float(df["latest_balance"].sum()),
    }
