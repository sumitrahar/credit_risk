import json

import numpy as np
import pandas as pd
import pytest

from src.anomaly import DETECTORS, detect_anomalies
from src.config import RISK_BAND_LABELS
from src.decisions import (
    all_answer_feedback,
    all_decisions,
    detector_outcomes,
    latest_decisions,
    record_answer_feedback,
    record_decision,
)
from src.model import (
    decile_table,
    ks_statistic,
    population_stability_index,
    primary_features,
)


def test_metrics_are_consistent_and_plausible(model_report):
    for name, metrics in model_report["models"].items():
        assert metrics["gini"] == pytest.approx(2 * metrics["auc"] - 1)
        # The signal in this data is modest. A much higher AUC would point to leakage.
        assert 0.60 < metrics["auc"] < 0.75, name
        assert metrics["auc_95_interval"][0] < metrics["auc"] < metrics["auc_95_interval"][1]
        assert 0.60 < metrics["out_of_time"]["auc"] < 0.75, name


def test_model_inputs_exclude_sensitive_and_leaky_fields():
    features = primary_features()
    used = set(features["numeric"] + features["skewed"] + features["categorical"])
    banned = {
        "gender", "marital_status", "region", "age", "age_band", "target_default", "defaulted_amount",
        "co_events", "co_recovered", "customer_since", "loan_purpose", "latest_balance",
    }
    assert not used & banned
    assert not any(name.startswith(("cur_", "co_")) for name in used)


def test_risk_bands_rise_in_order_and_are_calibrated(model_report):
    bands = model_report["risk_bands"]
    assert [b["risk_band"] for b in bands] == RISK_BAND_LABELS
    rates = [b["default_rate"] for b in bands]
    assert rates == sorted(rates)
    for band in bands:
        assert band["default_rate"] == pytest.approx(band["mean_pd"], abs=0.03)
    assert sum(b["accounts"] for b in bands) == model_report["accounts"]


def test_scores_are_probabilities(portfolio):
    assert portfolio["pd_score"].between(0, 1).all()
    assert portfolio["pd_score"].mean() == pytest.approx(portfolio["target_default"].mean(), abs=0.01)


def test_drivers_exist_for_every_account(portfolio):
    sample = portfolio["drivers_json"].sample(200, random_state=1).map(json.loads)
    assert sample.map(len).eq(5).all()
    for drivers in sample:
        effects = [abs(d["log_odds"]) for d in drivers]
        assert effects == sorted(effects, reverse=True)
        assert all(d["label"] and d["value"] for d in drivers)


def test_ks_and_deciles_on_a_known_case():
    y = np.array([0, 0, 0, 0, 1, 1])
    perfect = np.array([0.1, 0.2, 0.3, 0.4, 0.8, 0.9])
    assert ks_statistic(y, perfect) == pytest.approx(1.0)
    table = decile_table(np.array([1] * 10 + [0] * 90), np.linspace(1, 0, 100))
    assert table[0]["capture"] == pytest.approx(1.0)
    assert table[0]["lift"] == pytest.approx(10.0)


def test_psi_detects_drift_and_ignores_noise():
    rng = np.random.default_rng(0)
    baseline = rng.normal(0, 1, 20_000)
    assert population_stability_index(baseline, rng.normal(0, 1, 20_000)) < 0.01
    assert population_stability_index(baseline, rng.normal(1, 1, 20_000)) > 0.25
    # A feature that is nearly always zero still gets a usable result.
    mostly_zero = np.zeros(1000)
    mostly_zero[:5] = 1.0
    assert population_stability_index(mostly_zero, mostly_zero) == pytest.approx(0.0, abs=1e-6)


def test_every_flagged_account_has_two_detectors_and_reasons(portfolio, anomaly_report):
    flagged = portfolio[portfolio["flagged"] == 1]
    assert len(flagged) == anomaly_report["accounts_flagged"]
    assert (flagged["n_detectors"] >= anomaly_report["min_detectors_for_review"]).all()
    for row in flagged.itertuples():
        fired, reasons = json.loads(row.detectors_json), json.loads(row.reasons_json)
        assert len(fired) == row.n_detectors == len(reasons)
        assert set(fired) <= set(DETECTORS)


def test_review_rank_orders_the_list(portfolio):
    flagged = portfolio[portfolio["flagged"] == 1].sort_values("review_rank")
    assert flagged["review_rank"].is_unique
    assert flagged["n_detectors"].is_monotonic_decreasing
    assert portfolio.loc[portfolio["flagged"] == 0, "review_rank"].isna().all()


def test_detectors_never_use_the_target(analysis, clean):
    flags, _ = detect_anomalies(analysis, clean["repayment"])
    shuffled = analysis.assign(target_default=np.random.default_rng(1).permutation(analysis["target_default"].to_numpy()))
    other, _ = detect_anomalies(shuffled, clean["repayment"])
    pd.testing.assert_series_equal(flags["detectors_json"], other["detectors_json"])
    pd.testing.assert_series_equal(flags["flagged"], other["flagged"])


def test_detector_definitions_match_the_data(portfolio):
    missed_run = portfolio[portfolio["rp_longest_missed_run"] >= 2]
    assert missed_run["detectors_json"].map(lambda v: "consecutive_missed" in json.loads(v)).all()
    # Thin-file customers have no bureau data, so bureau detectors cannot fire for them.
    thin = portfolio[portfolio["thin_file"] == 1]
    assert not thin["detectors_json"].map(lambda v: "peer_bureau_debt" in v or "bureau_writeoffs" in v).any()


@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    monkeypatch.setenv("DECISIONS_DB", str(tmp_path / "test.db"))


def test_decisions_persist_and_latest_wins(temp_db):
    record_decision("ACC1", "Confirm", "  looks real  ", ["peer_dpd"])
    record_decision("ACC2", "Dismiss", "", ["late_shock"])
    record_decision("ACC1", "Dismiss", "changed my mind", ["peer_dpd"])
    assert len(all_decisions()) == 3
    latest = latest_decisions()
    assert latest.loc["ACC1", "decision"] == "Dismiss"
    assert latest.loc["ACC1", "note"] == "changed my mind"
    assert latest.loc["ACC2", "decision"] == "Dismiss"


def test_invalid_decisions_are_rejected(temp_db):
    with pytest.raises(ValueError):
        record_decision("ACC1", "Maybe")


def test_detector_outcomes_count_the_latest_decision(temp_db):
    record_decision("ACC1", "Confirm", "", ["peer_dpd", "late_shock"])
    record_decision("ACC2", "Dismiss", "", ["peer_dpd"])
    table = detector_outcomes().set_index("detector")
    assert table.loc["peer_dpd", "Confirm"] == 1
    assert table.loc["peer_dpd", "Dismiss"] == 1
    assert table.loc["peer_dpd", "dismiss_rate"] == pytest.approx(0.5)
    assert table.loc["late_shock", "dismiss_rate"] == 0


def test_empty_decision_log_is_handled(temp_db):
    assert latest_decisions().empty
    assert detector_outcomes().empty


def test_answer_feedback_is_stored(temp_db):
    record_answer_feedback("q", "a", "{}", "wrong")
    feedback = all_answer_feedback()
    assert feedback.iloc[0]["verdict"] == "wrong"
