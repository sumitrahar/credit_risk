"""Run the real Streamlit app headlessly and drive each view. No browser and no API key needed."""
import json
import re
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from src.assistant import Routing
from src.config import MAX_QUESTIONS_PER_SESSION

APP = str(Path(__file__).resolve().parents[1] / "app" / "app.py")
PAGES = ["Portfolio", "Account", "Ask the Portfolio", "Anomaly Review", "Model & Monitoring", "Data Notes"]


@pytest.fixture
def app(tmp_path, monkeypatch):
    monkeypatch.setenv("DECISIONS_DB", str(tmp_path / "app.db"))
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    # Keep any real local key out of these tests.
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: False)
    return AppTest.from_file(APP, default_timeout=120).run()


def open_page(app: AppTest, name: str) -> AppTest:
    app.sidebar.radio[0].set_value(name)
    return app.run()


def all_text(app: AppTest) -> str:
    parts = []
    for group in (app.markdown, app.caption, app.info, app.warning, app.title, app.subheader):
        parts.extend(element.value for element in group)
    # Dollar signs and similar characters are backslash-escaped for display.
    return "\n".join(parts).replace("\\", "")


@pytest.mark.parametrize("page", PAGES)
def test_every_view_renders_without_an_exception(app, page):
    open_page(app, page)
    assert not app.exception, page
    assert app.title[0].value.startswith(page.split(" &")[0])


def test_portfolio_kpis_match_the_data(app, portfolio):
    labels = {metric.label: metric.value for metric in app.metric}
    assert labels["Accounts"] == f"{len(portfolio):,}"
    assert labels["Originated"] == f"${portfolio['loan_amount'].sum() / 1e6:,.1f}M"
    assert labels["Default rate"] == f"{portfolio['target_default'].mean():.1%}"
    assert labels["In review list"] == f"{int(portfolio['flagged'].sum()):,}"


def test_filters_change_the_numbers(app, portfolio):
    mortgages = portfolio[portfolio["loan_type"] == "Mortgage"]
    app.multiselect(key="filter_loan_type").set_value(["Mortgage"]).run()
    labels = {metric.label: metric.value for metric in app.metric}
    assert labels["Accounts"] == f"{len(mortgages):,}"
    assert labels["Default rate"] == f"{mortgages['target_default'].mean():.1%}"
    assert "Showing" in all_text(app) and "Mortgage" in all_text(app)


def test_filters_that_exclude_everything_show_a_message(app):
    app.multiselect(key="filter_loan_type").set_value(["Mortgage"])
    app.multiselect(key="filter_age_band").set_value(["21-25"])
    app.multiselect(key="filter_origination_year").set_value([2018])
    app.multiselect(key="filter_region").set_value(["Central"])
    app.multiselect(key="filter_income_band").set_value(["$100k+"])
    app.multiselect(key="filter_employment_type").set_value(["Unemployed"]).run()
    assert not app.exception
    assert any("No accounts match" in warning.value for warning in app.warning)


def test_account_view_shows_the_requested_account(app, portfolio):
    open_page(app, "Account")
    row = portfolio.iloc[5]
    app.text_input(key="account_box").set_value(row["account_id"]).run()
    assert not app.exception
    labels = {metric.label: metric.value for metric in app.metric}
    assert labels["Predicted default"] == f"{row['pd_score'] * 100:.1f}%"
    assert labels["Recorded outcome"] == row["default_status"]


def test_account_view_rejects_unknown_ids(app):
    open_page(app, "Account")
    app.text_input(key="account_box").set_value("ACC000000").run()
    assert not app.exception
    assert any("no account ACC000000" in warning.value for warning in app.warning)


def test_ask_without_a_key_explains_and_offers_the_guided_query(app):
    open_page(app, "Ask the Portfolio")
    assert any("No Gemini API key is set" in info.value for info in app.info)
    assert "Mortgage has the highest default rate" in all_text(app)
    # The form for typing a question is not offered without a key.
    assert not any(element.label == "Your question" for element in app.text_input)


@pytest.mark.parametrize("page", ["Ask the Portfolio", "Anomaly Review"])
def test_ai_views_state_trigger_false_positive_false_negative_and_human_step(app, page):
    open_page(app, page)
    text = all_text(app)
    for heading in ("What triggers it", "A false positive", "A false negative", "The human step"):
        assert heading in text


class _StubLLM:
    def __init__(self, *args, **kwargs):
        self.calls = 0

    def route(self, question, catalogue_text):
        self.calls += 1
        if "loss given default" in question.lower():
            return Routing(kind="decline", reason="loss given default is not in the data")
        return Routing(kind="query", spec={"metric": "default_rate_pct", "group_by": ["loan_type"], "limit": 1})

    def phrase(self, question, payload):
        return f"{payload['rows'][0]['loan_type']} has the highest default rate, at {payload['rows'][0]['value']}."


@pytest.fixture
def keyed_app(tmp_path, monkeypatch):
    monkeypatch.setenv("DECISIONS_DB", str(tmp_path / "app.db"))
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setattr("src.assistant.GeminiLLM", _StubLLM)  # no test ever calls the real API
    import streamlit as st

    st.cache_resource.clear()
    app = AppTest.from_file(APP, default_timeout=120).run()
    open_page(app, "Ask the Portfolio")
    yield app
    st.cache_resource.clear()


def _submit(app: AppTest, question: str) -> AppTest:
    box = next(element for element in app.text_input if element.label == "Your question")
    box.set_value(question)
    next(button for button in app.button if button.label == "Ask").click()
    return app.run()


def test_a_question_gets_an_answer_with_figures_and_a_source(keyed_app, portfolio):
    _submit(keyed_app, "Which loan type has the highest default rate?")
    assert not keyed_app.exception
    rate = portfolio.loc[portfolio["loan_type"] == "Mortgage", "target_default"].mean() * 100
    text = all_text(keyed_app)
    assert f"Mortgage has the highest default rate, at {rate:.2f}%." in text
    assert "Source: loan_accounts.csv" in text


def test_an_unanswerable_question_is_declined(keyed_app):
    _submit(keyed_app, "What is the loss given default on mortgages?")
    assert not keyed_app.exception
    assert any("I can't answer that from this data" in info.value for info in keyed_app.info)


def test_asking_the_same_question_twice_shows_one_answer_and_no_error(keyed_app):
    _submit(keyed_app, "Which loan type has the highest default rate?")
    _submit(keyed_app, "which loan type  has the highest default rate?")
    assert not keyed_app.exception
    assert len(keyed_app.session_state["ask_history"]) == 1


def test_marking_an_answer_as_wrong_is_logged(keyed_app):
    from src.decisions import all_answer_feedback

    _submit(keyed_app, "Which loan type has the highest default rate?")
    next(button for button in keyed_app.button if button.label == "Mark as wrong").click()
    keyed_app.run()
    assert not keyed_app.exception
    logged = all_answer_feedback()
    assert logged.iloc[0]["question"] == "Which loan type has the highest default rate?"
    assert logged.iloc[0]["verdict"] == "wrong"
    assert "logged as wrong" in all_text(keyed_app)


def test_account_view_links_to_the_anomaly_review(app, portfolio):
    flagged = portfolio[portfolio["flagged"] == 1].iloc[0]["account_id"]
    open_page(app, "Account")
    app.text_input(key="account_box").set_value(flagged).run()
    next(button for button in app.button if button.label == "Open in Anomaly Review").click()
    app.run()
    assert not app.exception
    assert app.title[0].value == "Anomaly Review"
    assert f"Why {flagged} was flagged" in all_text(app)


def test_single_signal_accounts_can_be_opened_in_the_review_view(app, portfolio):
    single = portfolio[portfolio["n_detectors"] == 1].iloc[0]["account_id"]
    open_page(app, "Account")
    app.text_input(key="account_box").set_value(single).run()
    next(button for button in app.button if button.label == "Open in Anomaly Review").click()
    app.run()
    assert not app.exception
    assert f"Why {single} was flagged" in all_text(app)


def test_dollar_amounts_are_escaped_so_they_never_render_as_a_formula(app):
    open_page(app, "Ask the Portfolio")
    app.selectbox(key="guided_metric").set_value("originated_amount").run()
    assert not app.exception
    answers = [m.value for m in app.markdown if "highest originated amount" in m.value]
    assert answers, "the guided query should state the top loan type"
    assert answers[0].count("$") >= 2
    assert re.search(r"(?<!\\)\$", answers[0]) is None, "every $ must be escaped"


def test_several_income_band_filters_do_not_render_as_a_formula(app):
    app.multiselect(key="filter_income_band").set_value(["$40-60k", "$60-80k"]).run()
    assert not app.exception
    caption = next(c.value for c in app.caption if c.value.startswith("Showing"))
    assert re.search(r"(?<!\\)\$", caption) is None


def test_session_cap_stops_further_llm_calls(keyed_app):
    keyed_app.session_state["ask_used"] = MAX_QUESTIONS_PER_SESSION
    _submit(keyed_app, "How many accounts are there, new wording?")
    assert not keyed_app.exception
    assert any(f"used its {MAX_QUESTIONS_PER_SESSION} questions" in info.value for info in keyed_app.info)


def test_a_decision_is_recorded_and_survives_a_new_session(app, tmp_path, monkeypatch, portfolio):
    open_page(app, "Anomaly Review")
    assert not app.exception
    top = portfolio[portfolio["flagged"] == 1].sort_values(
        ["n_detectors", "anomaly_score", "latest_balance"], ascending=False
    ).iloc[0]["account_id"]
    radio = next(element for element in app.radio if element.label == "Your decision")
    radio.set_value("Confirm")
    next(element for element in app.text_input if element.label == "Note").set_value("Real concern")
    next(button for button in app.button if button.label == "Record decision").click()
    app.run()
    assert not app.exception

    # A brand new session, as if the page was reloaded, still sees it.
    fresh = AppTest.from_file(APP, default_timeout=120).run()
    open_page(fresh, "Anomaly Review")
    labels = {metric.label: metric.value for metric in fresh.metric}
    assert labels["Confirmed"] == "1"
    assert labels["Reviewed"] == "1"
    logged = fresh.dataframe
    assert any(top in str(frame.value.get("Account", [])) if hasattr(frame.value, "get") else False for frame in logged)


def test_anomaly_review_lists_the_reasons(app, portfolio):
    open_page(app, "Anomaly Review")
    top = portfolio[portfolio["flagged"] == 1].sort_values(
        ["n_detectors", "anomaly_score", "latest_balance"], ascending=False
    ).iloc[0]
    text = all_text(app)
    assert f"Why {top['account_id']} was flagged" in text
    for reason in json.loads(top["reasons_json"]):
        assert reason in text
