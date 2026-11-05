import pandas as pd
import pytest

from src.config import RAW_DIR
from src.features import bureau_features_point_in_time, repayment_features
from src.ingest import check_integrity, load_raw, monthly_payment, solve_rate


def test_analysis_dataset_has_one_row_per_account(analysis):
    raw_accounts = pd.read_csv(RAW_DIR / "loan_accounts.csv")
    assert len(analysis) == len(raw_accounts) == 14_685
    assert analysis["account_id"].is_unique
    assert set(analysis["account_id"]) == set(raw_accounts["account_id"])


def test_headline_figures_match_the_raw_files(analysis):
    raw_accounts = pd.read_csv(RAW_DIR / "loan_accounts.csv")
    assert analysis["target_default"].sum() == raw_accounts["target_default"].sum() == 2_364
    assert analysis["loan_amount"].sum() == pytest.approx(raw_accounts["loan_amount"].sum())
    assert round(analysis["target_default"].mean() * 100, 2) == 16.10
    assert round(analysis["loan_amount"].sum() / 1e6, 1) == 601.1


def test_gender_labels_are_consistent(clean):
    assert set(clean["customers"]["gender"]) == {"Male", "Female"}


def test_no_missing_values_remain_in_imputed_columns(clean):
    customers = clean["customers"]
    for column in ["years_employed", "num_dependents", "years_at_residence"]:
        assert customers[column].notna().all()
        assert customers[f"{column}_missing"].sum() == 500
    assert clean["loans"]["interest_rate_pct"].notna().all()


def test_unemployed_customers_have_zero_years_employed(clean):
    customers = clean["customers"]
    assert (customers.loc[customers["employment_type"] == "Unemployed", "years_employed"] == 0).all()


def test_caps_hold_after_imputation(clean):
    customers = clean["customers"]
    assert (customers["years_employed"] <= (customers["age"] - 16).clip(lower=0)).all()
    assert (customers["years_at_residence"] <= customers["age"]).all()


def test_rate_can_be_recovered_from_the_payment():
    payment = monthly_payment(100_000, 7.25, 60)
    assert solve_rate(100_000, 60, payment) == pytest.approx(7.25, abs=1e-6)


def test_recovered_rates_reproduce_the_payment(clean):
    loans = clean["loans"]
    recovered = loans[loans["interest_rate_recovered"] == 1]
    assert len(recovered) == 294
    for row in recovered.head(50).itertuples():
        # Rates are stored to 2 decimals, so the payment matches to within a small tolerance.
        assert monthly_payment(row.loan_amount, row.interest_rate_pct, row.tenure_months) == pytest.approx(row.emi, rel=1e-2)


def test_structural_nulls_are_left_alone(clean):
    bureau = clean["bureau"]
    assert bureau.loc[bureau["account_status"] == "Active", "date_closed"].isna().all()
    assert bureau.loc[bureau["account_status"] == "Closed", "days_past_due"].notna().all()
    assert bureau["dpd_missing"].sum() == 429


def test_every_issue_in_the_log_has_a_decision(dq_log):
    assert dq_log["decision"].str.len().gt(10).all()
    assert (dq_log["rows_affected"] > 0).all()
    assert len(dq_log) >= 30


def test_broken_keys_stop_the_build():
    raw = load_raw()
    raw["loans"] = pd.concat([raw["loans"], raw["loans"].head(1)])
    with pytest.raises(ValueError, match="duplicate key"):
        check_integrity(raw)

    raw = load_raw()
    raw["repayment"].loc[0, "account_id"] = "ACC999999"
    with pytest.raises(ValueError, match="no match"):
        check_integrity(raw)


def _loan(account_id, customer_id, originated):
    return {"account_id": account_id, "customer_id": customer_id, "origination_date": pd.Timestamp(originated)}


def _line(customer_id, opened, closed=None, status="Active", dpd=10.0, outstanding=1000.0, limit=2000.0):
    return {
        "bureau_id": f"B{opened}{closed}",
        "customer_id": customer_id,
        "lender_category": "Bank",
        "date_opened": pd.Timestamp(opened),
        "date_closed": pd.Timestamp(closed) if closed else pd.NaT,
        "account_status": status,
        "days_past_due": dpd,
        "outstanding_amount": outstanding,
        "sanctioned_amount": limit,
    }


def test_bureau_lines_after_the_scoring_date_are_ignored():
    loans = pd.DataFrame([_loan("A1", "C1", "2020-01-15")])
    bureau = pd.DataFrame(
        [
            _line("C1", "2019-06-01"),                         # visible
            _line("C1", "2020-12-31"),                         # visible: before the 12-month cut
            _line("C1", "2021-02-01", dpd=80, outstanding=9),  # opened after the cut: ignored
        ]
    )
    result = bureau_features_point_in_time(loans, bureau, 12).loc["A1"]
    assert result["bu_lines"] == 2
    assert result["bu_max_dpd"] == 10
    assert result["bu_outstanding"] == 2000


def test_later_write_offs_are_rewound_to_active():
    loans = pd.DataFrame([_loan("A1", "C1", "2020-01-15")])
    bureau = pd.DataFrame(
        [
            _line("C1", "2019-01-01", "2020-06-01", "Written-Off", dpd=50),  # written off before the cut: counts
            _line("C1", "2019-01-01", "2023-06-01", "Written-Off", dpd=70),  # written off later: Active, DPD unknown
            _line("C1", "2019-01-01", "2028-06-01", "Closed", dpd=0),        # impossible future date: still open then
        ]
    )
    result = bureau_features_point_in_time(loans, bureau, 12).loc["A1"]
    assert result["bu_lines"] == 3
    assert result["bu_writeoffs"] == 1
    assert result["bu_active"] == 2
    assert result["bu_max_dpd"] == 50


def test_no_visible_line_reaches_features_past_the_cut_on_real_data(clean):
    loans, bureau = clean["loans"], clean["bureau"]
    features = bureau_features_point_in_time(loans, bureau, 12)
    pairs = loans[["account_id", "customer_id", "origination_date"]].merge(bureau, on="customer_id")
    cut = pairs["origination_date"] + pd.DateOffset(months=12)
    visible = pairs[pairs["date_opened"] <= cut].groupby("account_id").size()
    assert features["bu_lines"].dropna().astype(int).equals(visible.reindex(features["bu_lines"].dropna().index).astype(int))


def test_longest_missed_run():
    rows = []
    pattern = ["On-Time", "Missed", "Missed", "On-Time", "Missed", "Missed", "Missed", "On-Time", "Late", "On-Time", "On-Time", "On-Time"]
    for month, status in enumerate(pattern, start=1):
        rows.append(
            {
                "account_id": "A1", "payment_month": month, "payment_status": status,
                "days_past_due": 30 if status == "Missed" else 0,
                "emi_due": 100.0, "emi_paid": 0.0 if status == "Missed" else 100.0,
            }
        )
    result = repayment_features(pd.DataFrame(rows)).loc["A1"]
    assert result["rp_longest_missed_run"] == 3
    assert result["rp_missed"] == 5
    assert result["rp_late"] == 1
