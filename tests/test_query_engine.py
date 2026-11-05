import pandas as pd
import pytest

from src.query_engine import (
    Filter,
    METRICS,
    QueryError,
    QuerySpec,
    catalogue,
    format_value,
    lookup_account,
    run_query,
    template_answer,
)


def test_default_rate_by_loan_type_matches_pandas(portfolio):
    result = run_query(QuerySpec(metric="default_rate_pct", group_by=["loan_type"]), portfolio)
    expected = portfolio.groupby("loan_type")["target_default"].mean().sort_values(ascending=False) * 100
    assert [r["loan_type"] for r in result.rows] == list(expected.index)
    for row in result.rows:
        assert row["value"] == pytest.approx(expected[row["loan_type"]])
    assert result.rows[0]["loan_type"] == "Mortgage"


def test_filters_and_two_way_groups_match_pandas(portfolio):
    spec = QuerySpec(
        metric="default_rate_pct",
        group_by=["loan_type", "employment_type"],
        filters=[Filter(dimension="region", values=["South", "North"])],
    )
    result = run_query(spec, portfolio)
    subset = portfolio[portfolio["region"].isin(["South", "North"])]
    expected = subset.groupby(["loan_type", "employment_type"])["target_default"].mean() * 100
    assert len(result.rows) == len(expected)
    for row in result.rows:
        assert row["value"] == pytest.approx(expected[(row["loan_type"], row["employment_type"])])
    assert result.scope_accounts == len(subset)


@pytest.mark.parametrize(
    "metric, expected",
    [
        ("accounts", lambda p: len(p)),
        ("customers", lambda p: p["customer_id"].nunique()),
        ("defaults", lambda p: p["target_default"].sum()),
        ("originated_amount", lambda p: p["loan_amount"].sum()),
        ("latest_balance", lambda p: p["latest_balance"].sum()),
        ("avg_interest_rate_pct", lambda p: p["interest_rate_pct"].mean()),
        ("flagged_accounts", lambda p: p["flagged"].sum()),
        ("amount_recovered", lambda p: p["co_recovered"].sum()),
        ("share_of_originated_pct", lambda p: 100.0),
    ],
)
def test_whole_portfolio_metrics_match_pandas(portfolio, metric, expected):
    result = run_query(QuerySpec(metric=metric), portfolio)
    assert result.rows[0]["value"] == pytest.approx(expected(portfolio))


def test_shares_use_the_whole_portfolio_as_the_denominator(portfolio):
    spec = QuerySpec(metric="share_of_originated_pct", filters=[Filter(dimension="loan_type", values=["Mortgage"])])
    result = run_query(spec, portfolio)
    mortgage = portfolio[portfolio["loan_type"] == "Mortgage"]
    assert result.rows[0]["value"] == pytest.approx(mortgage["loan_amount"].sum() / portfolio["loan_amount"].sum() * 100)


def test_order_and_limit(portfolio):
    result = run_query(QuerySpec(metric="default_rate_pct", group_by=["region"], order="asc", limit=1), portfolio)
    expected = portfolio.groupby("region")["target_default"].mean().idxmin()
    assert len(result.rows) == 1
    assert result.rows[0]["region"] == expected


def test_filter_values_match_regardless_of_case_or_number_type(portfolio):
    spec = QuerySpec(
        metric="accounts",
        filters=[Filter(dimension="loan_type", values=["mortgage"]), Filter(dimension="origination_year", values=[2023.0])],
    )
    result = run_query(spec, portfolio)
    expected = ((portfolio["loan_type"] == "Mortgage") & (portfolio["origination_year"] == 2023)).sum()
    assert result.rows[0]["value"] == expected


@pytest.mark.parametrize(
    "spec",
    [
        QuerySpec(metric="loss_given_default"),
        QuerySpec(metric="accounts", group_by=["favourite_colour"]),
        QuerySpec(metric="accounts", filters=[Filter(dimension="planet", values=["Mars"])]),
        QuerySpec(metric="accounts", filters=[Filter(dimension="loan_type", values=["Yacht Loan"])]),
        QuerySpec(metric="accounts", group_by=["region", "region"]),
    ],
)
def test_anything_outside_the_catalogue_is_rejected(portfolio, spec):
    with pytest.raises(QueryError):
        run_query(spec, portfolio)


def test_no_matching_accounts_is_an_error_not_a_zero(portfolio):
    spec = QuerySpec(
        metric="default_rate_pct",
        filters=[Filter(dimension="loan_type", values=["Mortgage"]), Filter(dimension="tenure_months", values=["12"])],
    )
    with pytest.raises(QueryError, match="No accounts"):
        run_query(spec, portfolio)


def test_payload_contains_results_but_no_customer_rows(portfolio):
    result = run_query(QuerySpec(metric="default_rate_pct", group_by=["loan_type"]), portfolio)
    text = str(result.payload)
    assert "CUST" not in text and "ACC2" not in text
    assert "Mortgage" in text
    assert len(result.payload["rows"]) == 5


def test_large_results_are_cut_before_reaching_the_llm(portfolio):
    result = run_query(QuerySpec(metric="accounts", group_by=["loan_type", "occupation_sector"]), portfolio)
    assert len(result.rows) == 50
    assert len(result.payload["rows"]) == 20


def test_template_answer_states_the_exact_figures(portfolio):
    result = run_query(QuerySpec(metric="default_rate_pct", group_by=["loan_type"], limit=1), portfolio)
    mortgage = portfolio[portfolio["loan_type"] == "Mortgage"]
    text = template_answer(result)
    assert "Mortgage" in text
    assert f"{mortgage['target_default'].mean() * 100:.2f}%" in text
    assert f"{int(mortgage['target_default'].sum()):,} of {len(mortgage):,}" in text


def test_every_metric_runs_and_formats(portfolio):
    for key, metric in METRICS.items():
        result = run_query(QuerySpec(metric=key, group_by=["region"]), portfolio)
        assert len(result.rows) == 5
        for row in result.rows:
            assert row["formatted_value"] == format_value(metric, row["value"])
            assert not pd.isna(row["value"])


def test_catalogue_lists_every_metric_and_real_values(portfolio):
    listed = catalogue(portfolio)
    assert set(listed["metrics"]) == set(METRICS)
    assert "Mortgage" in listed["dimensions"]["loan_type"]
    assert "2023" in listed["dimensions"]["origination_year"]


def test_account_lookup_returns_scores_and_rejects_unknown_ids(portfolio):
    row = portfolio.iloc[0]
    result = lookup_account(row["account_id"].lower(), portfolio)
    assert result.fields["predicted_pd"] == f"{row['pd_score'] * 100:.2f}%"
    assert len(result.fields["top_drivers"]) == 5
    assert row["customer_id"] not in str(result.payload)
    with pytest.raises(QueryError):
        lookup_account("ACC999999", portfolio)
