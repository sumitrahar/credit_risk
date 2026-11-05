"""The only place Ask the Portfolio gets numbers from.

A question becomes a QuerySpec: one metric from a fixed list, up to two groupings and some
filters. This code validates the spec and computes the result with pandas. There is no SQL
and no generated code, so an LLM can only choose from what is listed here.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Callable, Literal

import pandas as pd
from pydantic import BaseModel, Field, field_validator

from src.config import DIMENSION_ORDER, MIN_SEGMENT_ACCOUNTS


class QueryError(ValueError):
    """The request cannot be answered from the catalogue."""


@dataclass(frozen=True)
class Metric:
    label: str
    description: str
    kind: Literal["count", "percent", "money", "number", "percent_value"]
    compute: Callable[[pd.DataFrame, pd.DataFrame], float]
    sources: tuple[str, ...]


def _share_of(column: str) -> Callable[[pd.DataFrame, pd.DataFrame], float]:
    def compute(frame: pd.DataFrame, portfolio: pd.DataFrame) -> float:
        return float(frame[column].sum() / portfolio[column].sum() * 100)
    return compute


def _recovery_rate(frame: pd.DataFrame, portfolio: pd.DataFrame) -> float:
    defaulted = frame[frame["target_default"] == 1]
    if defaulted.empty:
        return float("nan")
    return float(defaulted["co_recovered"].sum() / defaulted["loan_amount"].sum() * 100)


METRICS: dict[str, Metric] = {
    "accounts": Metric("number of accounts", "Count of loan accounts.", "count", lambda f, p: len(f), ("loan_accounts.csv",)),
    "customers": Metric("number of customers", "Count of distinct customers.", "count", lambda f, p: f["customer_id"].nunique(), ("loan_accounts.csv",)),
    "defaults": Metric("number of defaults", "Count of defaulted accounts.", "count", lambda f, p: f["target_default"].sum(), ("loan_accounts.csv: target_default",)),
    "default_rate_pct": Metric("default rate", "Defaulted accounts divided by accounts.", "percent", lambda f, p: f["target_default"].mean() * 100, ("loan_accounts.csv: target_default",)),
    "originated_amount": Metric("originated amount", "Sum of original loan amounts.", "money", lambda f, p: f["loan_amount"].sum(), ("loan_accounts.csv: loan_amount",)),
    "defaulted_amount": Metric("defaulted amount", "Sum of original loan amounts on defaulted accounts.", "money", lambda f, p: f["defaulted_amount"].sum(), ("loan_accounts.csv: loan_amount, target_default",)),
    "latest_balance": Metric("latest balance", "Sum of outstanding balances at the end of the repayment window (approximate; see the data-quality log).", "money", lambda f, p: f["latest_balance"].sum(), ("repayment_history.csv: outstanding_balance",)),
    "share_of_accounts_pct": Metric("share of all portfolio accounts", "Accounts as a percentage of all accounts in the portfolio.", "percent", lambda f, p: len(f) / len(p) * 100, ("loan_accounts.csv",)),
    "share_of_originated_pct": Metric("share of all originated exposure", "Originated amount as a percentage of the whole portfolio's.", "percent", _share_of("loan_amount"), ("loan_accounts.csv: loan_amount",)),
    "share_of_defaulted_amount_pct": Metric("share of all defaulted dollars", "Defaulted amount as a percentage of the whole portfolio's.", "percent", _share_of("defaulted_amount"), ("loan_accounts.csv: loan_amount, target_default",)),
    "share_of_defaults_pct": Metric("share of all defaults", "Defaulted accounts as a percentage of all defaulted accounts in the portfolio.", "percent", _share_of("target_default"), ("loan_accounts.csv: target_default",)),
    "avg_loan_amount": Metric("average loan amount", "Mean original loan amount.", "money", lambda f, p: f["loan_amount"].mean(), ("loan_accounts.csv: loan_amount",)),
    "avg_interest_rate_pct": Metric("average interest rate", "Mean annual interest rate.", "percent_value", lambda f, p: f["interest_rate_pct"].mean(), ("loan_accounts.csv: interest_rate_pct",)),
    "avg_annual_income": Metric("average annual income", "Mean annual income of account holders, counted per account.", "money", lambda f, p: f["annual_income"].mean(), ("customers.csv: annual_income",)),
    "avg_age": Metric("average age", "Mean age of account holders, counted per account.", "number", lambda f, p: f["age"].mean(), ("customers.csv: age",)),
    "avg_predicted_pd_pct": Metric("average predicted probability of default", "Mean out-of-fold model score.", "percent", lambda f, p: f["pd_score"].mean() * 100, ("model scores: account_scores.parquet",)),
    "missed_payment_rate_pct": Metric("missed payment rate", "Missed months as a percentage of all repayment months.", "percent", lambda f, p: f["rp_missed"].sum() / (len(f) * 12) * 100, ("repayment_history.csv: payment_status",)),
    "avg_days_past_due": Metric("average days past due", "Mean days past due across the 12 repayment months.", "number", lambda f, p: f["rp_mean_dpd"].mean(), ("repayment_history.csv: days_past_due",)),
    "flagged_accounts": Metric("accounts in the anomaly review list", "Accounts flagged by two or more anomaly detectors.", "count", lambda f, p: f["flagged"].sum(), ("anomaly results: anomaly_flags.parquet",)),
    "collections_events": Metric("number of collections events", "Count of collections actions.", "count", lambda f, p: f["co_events"].sum(), ("collections_events.csv",)),
    "amount_recovered": Metric("amount recovered through collections", "Sum of amounts recovered.", "money", lambda f, p: f["co_recovered"].sum(), ("collections_events.csv: amount_recovered",)),
    "recovery_rate_on_defaults_pct": Metric("recovery rate on defaulted loans", "Amount recovered on defaulted accounts as a percentage of their original loan amounts.", "percent", _recovery_rate, ("collections_events.csv: amount_recovered", "loan_accounts.csv: loan_amount, target_default")),
}

DIMENSIONS: dict[str, tuple[str, str]] = {
    # name: (column in the portfolio table, source)
    "loan_type": ("loan_type", "loan_accounts.csv"),
    "loan_purpose": ("loan_purpose", "loan_accounts.csv"),
    "tenure_months": ("tenure_months", "loan_accounts.csv"),
    "disbursal_channel": ("disbursal_channel", "loan_accounts.csv"),
    "secured_status": ("secured_status", "loan_accounts.csv"),
    "default_status": ("default_status", "loan_accounts.csv"),
    "origination_year": ("origination_year", "loan_accounts.csv"),
    "employment_type": ("employment_type", "customers.csv"),
    "income_band": ("income_band", "customers.csv"),
    "age_band": ("age_band", "customers.csv"),
    "region": ("region", "customers.csv"),
    "city_tier": ("city_tier", "customers.csv"),
    "gender": ("gender", "customers.csv"),
    "marital_status": ("marital_status", "customers.csv"),
    "education_level": ("education_level", "customers.csv"),
    "housing_type": ("housing_type", "customers.csv"),
    "occupation_sector": ("occupation_sector", "customers.csv"),
    "bureau_record": ("bureau_record", "bureau_credit.csv"),
    "risk_band": ("risk_band", "model scores"),
    "review_status": ("review_status", "anomaly results"),
}


class Filter(BaseModel):
    dimension: str
    values: list[str] = Field(min_length=1)

    @field_validator("values", mode="before")
    @classmethod
    def _values_as_text(cls, values):
        # An LLM may send years or terms as numbers. 2023.0 and 2023 both mean "2023".
        if not isinstance(values, list):
            values = [values]
        return [str(int(v)) if isinstance(v, float) and v.is_integer() else str(v) for v in values]


class QuerySpec(BaseModel):
    metric: str
    group_by: list[str] = Field(default_factory=list, max_length=2)
    filters: list[Filter] = Field(default_factory=list)
    order: Literal["desc", "asc"] = "desc"
    limit: int | None = Field(default=None, ge=1, le=100)


@dataclass
class QueryResult:
    spec: QuerySpec
    metric: Metric
    rows: list[dict]               # group values, metric value, accounts and defaults
    scope_accounts: int
    source: str
    scope_text: str
    low_sample_groups: int = 0
    payload: dict = field(default_factory=dict)  # exactly what the LLM may see


def allowed_values(portfolio: pd.DataFrame, dimension: str) -> list[str]:
    column = DIMENSIONS[dimension][0]
    if dimension in DIMENSION_ORDER:
        return list(DIMENSION_ORDER[dimension])
    values = portfolio[column].dropna().unique()
    if pd.api.types.is_numeric_dtype(portfolio[column]):
        return [str(v) for v in sorted(values)]
    return sorted(str(v) for v in values)


def catalogue(portfolio: pd.DataFrame) -> dict:
    """The full list of what can be asked. This is all the LLM is told about the data."""
    return {
        "metrics": {key: metric.description for key, metric in METRICS.items()},
        "dimensions": {name: allowed_values(portfolio, name) for name in DIMENSIONS},
    }


def _resolve_values(portfolio: pd.DataFrame, dimension: str, values: list[str]) -> list[str]:
    allowed = allowed_values(portfolio, dimension)
    by_lower = {v.lower(): v for v in allowed}
    resolved = []
    for value in values:
        match = by_lower.get(str(value).strip().lower())
        if match is None:
            raise QueryError(f"'{value}' is not a known value of {dimension}.")
        resolved.append(match)
    return resolved


def validate_spec(spec: QuerySpec, portfolio: pd.DataFrame) -> QuerySpec:
    if spec.metric not in METRICS:
        raise QueryError(f"'{spec.metric}' is not an available metric.")
    for dimension in spec.group_by:
        if dimension not in DIMENSIONS:
            raise QueryError(f"'{dimension}' is not an available dimension.")
    if len(set(spec.group_by)) != len(spec.group_by):
        raise QueryError("Cannot group by the same dimension twice.")
    filters = []
    for item in spec.filters:
        if item.dimension not in DIMENSIONS:
            raise QueryError(f"'{item.dimension}' is not an available filter.")
        filters.append(Filter(dimension=item.dimension, values=_resolve_values(portfolio, item.dimension, item.values)))
    return spec.model_copy(update={"filters": filters})


def format_value(metric: Metric, value: float) -> str:
    if pd.isna(value):
        return "not available"
    if metric.kind == "count":
        return f"{value:,.0f}"
    if metric.kind == "percent":
        return f"{value:.2f}%"
    if metric.kind == "percent_value":
        return f"{value:.2f}%"
    if metric.kind == "money":
        if abs(value) >= 1e6:
            return f"${value / 1e6:,.1f}M"
        return f"${value:,.0f}"
    return f"{value:,.1f}"


def _scope_text(filters: list[Filter]) -> str:
    if not filters:
        return "whole portfolio"
    return "; ".join(f"{item.dimension} = {', '.join(item.values)}" for item in filters)


def _source_text(spec: QuerySpec, metric: Metric) -> str:
    sources = list(metric.sources)
    for dimension in [*spec.group_by, *(item.dimension for item in spec.filters)]:
        table = DIMENSIONS[dimension][1]
        if not any(table.split(":")[0] in existing for existing in sources):
            sources.append(table)
    return " + ".join(sources)


def run_query(spec: QuerySpec, portfolio: pd.DataFrame) -> QueryResult:
    """Validate the spec and compute the answer. All arithmetic happens here."""
    spec = validate_spec(spec, portfolio)
    metric = METRICS[spec.metric]

    frame = portfolio
    for item in spec.filters:
        column = DIMENSIONS[item.dimension][0]
        frame = frame[frame[column].astype(str).isin(item.values)]
    if frame.empty:
        raise QueryError("No accounts match those filters.")

    def row_for(group: pd.DataFrame, labels: dict) -> dict:
        row = dict(labels)
        row["value"] = float(metric.compute(group, portfolio))
        # A recovery rate only covers defaulted accounts, so that is the count to show with it.
        row["accounts"] = int(group["target_default"].sum()) if spec.metric == "recovery_rate_on_defaults_pct" else int(len(group))
        if spec.metric == "default_rate_pct":
            row["defaults"] = int(group["target_default"].sum())
        row["low_sample"] = len(group) < MIN_SEGMENT_ACCOUNTS
        return row

    if spec.group_by:
        columns = [DIMENSIONS[d][0] for d in spec.group_by]
        rows = [
            row_for(group, {d: str(k) for d, k in zip(spec.group_by, keys if isinstance(keys, tuple) else (keys,))})
            for keys, group in frame.groupby(columns)
        ]
        rows.sort(key=lambda r: r["value"], reverse=spec.order == "desc")
        if spec.limit:
            rows = rows[: spec.limit]
    else:
        rows = [row_for(frame, {})]

    for row in rows:
        row["formatted_value"] = format_value(metric, row["value"])

    result = QueryResult(
        spec=spec,
        metric=metric,
        rows=rows,
        scope_accounts=int(len(frame)),
        source=_source_text(spec, metric),
        scope_text=_scope_text(spec.filters),
        low_sample_groups=sum(r["low_sample"] for r in rows),
    )
    result.payload = {
        "metric": metric.label,
        "metric_definition": metric.description,
        "scope": result.scope_text,
        "order": spec.order if spec.group_by else None,
        "rows": [
            {
                **{d: row[d] for d in spec.group_by},
                "value": row["formatted_value"],
                "accounts": f"{row['accounts']:,}",
                **({"defaults": f"{row['defaults']:,}"} if "defaults" in row else {}),
                **({"low_sample": True} if row["low_sample"] else {}),
            }
            for row in rows[:20]
        ],
    }
    return result


def template_answer(result: QueryResult) -> str:
    """A plain answer written by code. Used when the LLM is not available or fails the check."""
    metric, rows, spec = result.metric, result.rows, result.spec

    def detail(row: dict) -> str:
        if "defaults" in row:
            return f" ({row['defaults']:,} of {row['accounts']:,} accounts)"
        if metric.kind == "count" and spec.metric == "accounts":
            return ""
        return f" ({row['accounts']:,} accounts)"

    scope = "" if not spec.filters else f" for {result.scope_text}"
    if not spec.group_by:
        return f"The {metric.label}{scope} is {rows[0]['formatted_value']}{detail(rows[0])}."

    def name(row: dict) -> str:
        return " / ".join(row[d] for d in spec.group_by)

    word = "highest" if spec.order == "desc" else "lowest"
    text = f"{name(rows[0])} has the {word} {metric.label}{scope}, at {rows[0]['formatted_value']}{detail(rows[0])}."
    if len(rows) > 1:
        others = ", ".join(f"{name(r)} {r['formatted_value']}" for r in rows[1:5])
        more = f", and {len(rows) - 5} more" if len(rows) > 5 else ""
        text += f" Next: {others}{more}."
    if result.low_sample_groups:
        text += f" {result.low_sample_groups} of these groups have fewer than {MIN_SEGMENT_ACCOUNTS} accounts, so treat them with caution."
    return text


def result_table(result: QueryResult) -> pd.DataFrame:
    """The result as a readable table for display."""
    table = pd.DataFrame(result.rows)
    display = pd.DataFrame({d.replace("_", " ").capitalize(): table[d] for d in result.spec.group_by})
    display[result.metric.label.capitalize()] = table["formatted_value"]
    display["Accounts"] = table["accounts"].map("{:,}".format)
    if "defaults" in table:
        display["Defaults"] = table["defaults"].map("{:,}".format)
    return display


@dataclass
class AccountResult:
    account_id: str
    fields: dict
    source: str
    payload: dict


def lookup_account(account_id: str, portfolio: pd.DataFrame) -> AccountResult:
    """Score, band, drivers and flags for one account. No customer details are included."""
    account_id = account_id.strip().upper()
    match = portfolio[portfolio["account_id"] == account_id]
    if match.empty:
        raise QueryError(f"There is no account {account_id} in this portfolio.")
    row = match.iloc[0]
    drivers = json.loads(row["drivers_json"])
    fields = {
        "account_id": account_id,
        "loan_type": row["loan_type"],
        "predicted_pd": f"{row['pd_score'] * 100:.2f}%",
        "risk_band": row["risk_band"],
        "percentile_in_portfolio": f"{row['pd_percentile']:.0f}",
        "outcome": row["default_status"],
        "in_anomaly_review_list": "Yes" if row["flagged"] else "No",
        "anomaly_reasons": json.loads(row["reasons_json"]),
        "top_drivers": [
            {"driver": d["label"], "value": d["value"], "effect_on_pd_points": f"{d['pd_points']:+.2f}"}
            for d in drivers
        ],
    }
    return AccountResult(
        account_id=account_id,
        fields=fields,
        source="model scores: account_scores.parquet + anomaly results: anomaly_flags.parquet",
        payload=fields,
    )


def template_account_answer(result: AccountResult) -> str:
    fields = result.fields
    drivers = "; ".join(f"{d['driver']} ({d['value']}, {d['effect_on_pd_points']} pts)" for d in fields["top_drivers"][:3])
    review = "It is in the anomaly review list." if fields["in_anomaly_review_list"] == "Yes" else "It is not in the anomaly review list."
    return (
        f"{fields['account_id']} is a {fields['loan_type']} with a predicted probability of default of "
        f"{fields['predicted_pd']} ({fields['risk_band']} band). Top drivers: {drivers}. {review}"
    )
