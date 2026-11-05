"""Tests for the assistant pipeline. No API key is needed: the LLM is faked."""
import json
from types import SimpleNamespace

import pytest
from google.genai import errors, types

from src.assistant import (
    GeminiLLM,
    LLMRateLimited,
    LLMUnavailable,
    Routing,
    answer_question,
    check_numbers,
    extract_numbers,
    function_declarations,
)
from src.config import MAX_QUESTION_CHARS


class FakeLLM:
    """Plays back a fixed routing and phrasing, and records what it was sent."""

    def __init__(self, routing=None, phrasing="", route_error=None, phrase_error=None):
        self.routing, self.phrasing = routing, phrasing
        self.route_error, self.phrase_error = route_error, phrase_error
        self.route_calls, self.phrase_calls = [], []

    def route(self, question, catalogue_text):
        self.route_calls.append((question, catalogue_text))
        if self.route_error:
            raise self.route_error
        return self.routing

    def phrase(self, question, payload):
        self.phrase_calls.append((question, payload))
        if self.phrase_error:
            raise self.phrase_error
        return self.phrasing


HIGHEST_DEFAULT_RATE = Routing(kind="query", spec={"metric": "default_rate_pct", "group_by": ["loan_type"], "order": "desc", "limit": 1})


def _mortgage_rate(portfolio):
    return f"{portfolio.loc[portfolio['loan_type'] == 'Mortgage', 'target_default'].mean() * 100:.2f}%"


def test_supported_answer_uses_the_llm_wording(portfolio):
    text = f"Mortgage has the highest default rate, at {_mortgage_rate(portfolio)}."
    answer = answer_question("Which loan type has the highest default rate?", portfolio, FakeLLM(HIGHEST_DEFAULT_RATE, text))
    assert answer.status == "answered"
    assert answer.written_by == "llm"
    assert answer.text == text
    assert "loan_accounts.csv" in answer.source
    assert answer.query["metric"] == "default_rate_pct"
    assert len(answer.table) == 1


def test_invented_number_is_blocked_and_replaced(portfolio):
    answer = answer_question(
        "Which loan type has the highest default rate?",
        portfolio,
        FakeLLM(HIGHEST_DEFAULT_RATE, "Mortgage has the highest default rate, at 31.40%."),
    )
    assert answer.status == "answered"
    assert answer.written_by == "code"
    assert "31.40" not in answer.text
    assert _mortgage_rate(portfolio) in answer.text
    assert "31.4" in answer.notes[0]


def test_rounded_number_is_also_blocked(portfolio):
    rounded = f"{portfolio.loc[portfolio['loan_type'] == 'Mortgage', 'target_default'].mean() * 100:.0f}%"
    answer = answer_question("Which?", portfolio, FakeLLM(HIGHEST_DEFAULT_RATE, f"Mortgage is about {rounded}."))
    assert answer.written_by == "code"


def test_llm_never_sees_data_rows_when_routing(portfolio):
    llm = FakeLLM(HIGHEST_DEFAULT_RATE, "text")
    answer_question("Which loan type has the highest default rate?", portfolio, llm)
    _, catalogue_text = llm.route_calls[0]
    assert "CUST1" not in catalogue_text and "ACC2" not in catalogue_text
    parsed = json.loads(catalogue_text)
    assert set(parsed) == {"metrics", "dimensions"}


def test_llm_decline_is_passed_through_without_calling_phrase(portfolio):
    llm = FakeLLM(Routing(kind="decline", reason="loss given default is not in the data"))
    answer = answer_question("What is the LGD on mortgages?", portfolio, llm)
    assert answer.status == "declined"
    assert "loss given default" in answer.text
    assert llm.phrase_calls == []
    assert answer.source is None


def test_a_decline_cannot_state_invented_figures(portfolio):
    llm = FakeLLM(Routing(kind="decline", reason="industry loss given default is about 42%"))
    answer = answer_question("What is the loss given default on mortgages?", portfolio, llm)
    assert answer.status == "declined"
    assert "42" not in answer.text


def test_a_decline_may_quote_years_from_the_catalogue(portfolio):
    llm = FakeLLM(Routing(kind="decline", reason="2031 is outside the available years, 2018-2023"))
    answer = answer_question("What was the default rate in 2031?", portfolio, llm)
    assert "2018-2023" in answer.text


def test_low_sample_rows_are_marked_for_the_llm(portfolio):
    from src.query_engine import QuerySpec, run_query

    result = run_query(QuerySpec(metric="default_rate_pct", group_by=["employment_type", "income_band"]), portfolio)
    marked = [row for row in result.payload["rows"] if row.get("low_sample")]
    unmarked = [row for row in result.payload["rows"] if not row.get("low_sample")]
    assert unmarked
    assert all(int(row["accounts"].replace(",", "")) < 30 for row in marked)
    assert all(int(row["accounts"].replace(",", "")) >= 30 for row in unmarked)


@pytest.mark.parametrize(
    "spec",
    [
        {"metric": "loss_given_default"},
        {"metric": "accounts", "group_by": ["favourite_colour"]},
        {"metric": "accounts", "filters": [{"dimension": "loan_type", "values": ["Yacht Loan"]}]},
        {"metric": "accounts", "group_by": ["a", "b", "c"]},
        {"not_a_field": True},
    ],
)
def test_router_mistakes_are_declined_not_guessed(portfolio, spec):
    llm = FakeLLM(Routing(kind="query", spec=spec))
    answer = answer_question("Something odd", portfolio, llm)
    assert answer.status == "declined"
    assert llm.phrase_calls == []


def test_rate_limit_while_routing_points_to_the_guided_query(portfolio):
    answer = answer_question("Which?", portfolio, FakeLLM(route_error=LLMRateLimited("429")))
    assert answer.status == "rate_limited"
    assert "guided query" in answer.text


def test_rate_limit_while_phrasing_still_gives_the_figures(portfolio):
    answer = answer_question("Which?", portfolio, FakeLLM(HIGHEST_DEFAULT_RATE, phrase_error=LLMRateLimited("429")))
    assert answer.status == "answered"
    assert answer.written_by == "code"
    assert _mortgage_rate(portfolio) in answer.text


def test_provider_failure_is_reported_plainly(portfolio):
    answer = answer_question("Which?", portfolio, FakeLLM(route_error=LLMUnavailable("Gemini returned an error (403).")))
    assert answer.status == "unavailable"
    assert "403" in answer.text


def test_long_or_empty_questions_never_reach_the_llm(portfolio):
    llm = FakeLLM(HIGHEST_DEFAULT_RATE, "x")
    assert answer_question("x" * (MAX_QUESTION_CHARS + 1), portfolio, llm).status == "declined"
    assert answer_question("   ", portfolio, llm).status == "declined"
    assert llm.route_calls == []


def test_account_questions_use_the_account_lookup(portfolio):
    row = portfolio.iloc[0]
    expected = f"{row['pd_score'] * 100:.2f}%"
    llm = FakeLLM(Routing(kind="account", account_id=row["account_id"]), f"The predicted probability of default is {expected}.")
    answer = answer_question(f"What is the score of {row['account_id']}?", portfolio, llm)
    assert answer.status == "answered"
    assert answer.written_by == "llm"
    assert expected in answer.text
    assert row["customer_id"] not in str(llm.phrase_calls)


def test_unknown_account_is_declined(portfolio):
    answer = answer_question("Score of ACC999999?", portfolio, FakeLLM(Routing(kind="account", account_id="ACC999999")))
    assert answer.status == "declined"


@pytest.mark.parametrize(
    "text, expected",
    [
        ("The rate is 24.65%.", [24.65]),
        ("It totals $314.5M across 1,432 accounts.", [314.5e6, 1432]),
        ("That is $314.5 million.", [314.5e6]),
        ("Average of 3.49x, up from $40k.", [3.49, 40e3]),
        ("Account ACC200123 is fine.", []),
    ],
)
def test_number_extraction(text, expected):
    assert extract_numbers(text) == pytest.approx(expected)


def test_number_check_ignores_labels_and_question_numbers():
    payload = {"rows": [{"income_band": "$40-60k", "city_tier": "Tier 1", "value": "22.42%", "accounts": "2,288"}]}
    assert check_numbers("In Tier 1, the $40-60k band shows 22.42% over 2,288 accounts.", payload) == []
    assert check_numbers("The top 3 start with 22.42%.", payload, "What are the top 3?") == []
    assert check_numbers("The top 3 start with 22.42%.", payload, "What is highest?") == [3]
    assert check_numbers("It is 22.40%.", payload) == [22.40]


def _function_call_response(name, args):
    return types.GenerateContentResponse(
        candidates=[
            types.Candidate(content=types.Content(role="model", parts=[types.Part(function_call=types.FunctionCall(name=name, args=args))]))
        ]
    )


class FakeClient:
    """Stands in for genai.Client, using real SDK request and response types."""

    def __init__(self, response=None, error=None):
        self.response, self.error, self.calls = response, error, []
        self.models = SimpleNamespace(generate_content=self._generate)

    def _generate(self, model, contents, config):
        self.calls.append(SimpleNamespace(model=model, contents=contents, config=config))
        if self.error:
            raise self.error
        return self.response


def test_gemini_query_call_is_parsed():
    client = FakeClient(_function_call_response("query_portfolio", {"metric": "default_rate_pct", "group_by": ["region"]}))
    routing = GeminiLLM("key", model="test-model", client=client).route("q", "{}")
    assert routing.kind == "query"
    assert routing.spec["metric"] == "default_rate_pct"


def test_gemini_is_forced_to_choose_a_function():
    client = FakeClient(_function_call_response("decline", {"reason": "not in the data"}))
    routing = GeminiLLM("key", model="test-model", client=client).route("q", "{}")
    assert routing.kind == "decline" and routing.reason == "not in the data"
    config = client.calls[0].config
    assert config.tool_config.function_calling_config.mode == "ANY"
    names = [d.name for d in config.tools[0].function_declarations]
    assert names == ["query_portfolio", "get_account", "decline"]
    assert config.temperature == 0


def test_function_declarations_build_with_the_pinned_sdk():
    for declaration in function_declarations():
        built = types.FunctionDeclaration(**declaration)
        assert built.name and built.parameters_json_schema


def test_gemini_response_with_no_call_is_treated_as_a_failure():
    response = types.GenerateContentResponse(
        candidates=[types.Candidate(content=types.Content(role="model", parts=[types.Part(text="Hello")]))]
    )
    with pytest.raises(LLMUnavailable):
        GeminiLLM("key", model="test-model", client=FakeClient(response)).route("q", "{}")


def test_gemini_429_becomes_a_rate_limit():
    error = errors.ClientError(429, {"error": {"code": 429, "message": "Quota exceeded", "status": "RESOURCE_EXHAUSTED"}})
    with pytest.raises(LLMRateLimited):
        GeminiLLM("key", model="test-model", client=FakeClient(error=error)).route("q", "{}")


def test_gemini_bad_key_and_outage_become_unavailable():
    for code, error_type in [(403, errors.ClientError), (503, errors.ServerError)]:
        error = error_type(code, {"error": {"code": code, "message": "No", "status": "X"}})
        with pytest.raises(LLMUnavailable):
            GeminiLLM("key", model="test-model", client=FakeClient(error=error)).route("q", "{}")
    with pytest.raises(LLMUnavailable):
        GeminiLLM("key", model="test-model", client=FakeClient(error=TimeoutError())).route("q", "{}")


class ScriptedClient:
    """Each model gets its own scripted result: an error to raise, or a response to return."""

    def __init__(self, script):
        self.script, self.tried = script, []
        self.models = SimpleNamespace(generate_content=self._generate)

    def _generate(self, model, contents, config):
        self.tried.append(model)
        result = self.script[model]
        if isinstance(result, Exception):
            raise result
        return result


def _api_error(code):
    kind = errors.ServerError if code >= 500 else errors.ClientError
    return kind(code, {"error": {"code": code, "message": "No", "status": "X"}})


DECLINE_RESPONSE = _function_call_response("decline", {"reason": "not in the data"})


@pytest.mark.parametrize("code", [404, 429, 500, 503])
def test_an_overloaded_or_retired_model_falls_back_to_the_next(code):
    client = ScriptedClient({"main": _api_error(code), "backup": DECLINE_RESPONSE})
    llm = GeminiLLM("key", model="main", fallbacks=("backup",), client=client)
    assert llm.route("q", "{}").kind == "decline"
    assert client.tried == ["main", "backup"]
    assert llm.last_model == "backup"


def test_a_timeout_falls_back_to_the_next_model():
    client = ScriptedClient({"main": TimeoutError(), "backup": DECLINE_RESPONSE})
    llm = GeminiLLM("key", model="main", fallbacks=("backup",), client=client)
    assert llm.route("q", "{}").kind == "decline"


@pytest.mark.parametrize("code", [400, 401, 403])
def test_a_bad_key_or_request_stops_at_once(code):
    client = ScriptedClient({"main": _api_error(code), "backup": DECLINE_RESPONSE})
    llm = GeminiLLM("key", model="main", fallbacks=("backup",), client=client)
    with pytest.raises(LLMUnavailable):
        llm.route("q", "{}")
    assert client.tried == ["main"]


def test_the_first_healthy_model_is_used_without_trying_others():
    client = ScriptedClient({"main": DECLINE_RESPONSE, "backup": _api_error(503)})
    llm = GeminiLLM("key", model="main", fallbacks=("backup",), client=client)
    llm.route("q", "{}")
    assert client.tried == ["main"]
    assert llm.last_model == "main"


def test_if_every_model_fails_the_error_says_why():
    all_limited = ScriptedClient({"main": _api_error(429), "backup": _api_error(429)})
    with pytest.raises(LLMRateLimited):
        GeminiLLM("key", model="main", fallbacks=("backup",), client=all_limited).route("q", "{}")
    all_down = ScriptedClient({"main": _api_error(503), "backup": _api_error(500)})
    with pytest.raises(LLMUnavailable):
        GeminiLLM("key", model="main", fallbacks=("backup",), client=all_down).route("q", "{}")
    mixed = ScriptedClient({"main": _api_error(503), "backup": _api_error(429)})
    with pytest.raises(LLMRateLimited):
        GeminiLLM("key", model="main", fallbacks=("backup",), client=mixed).route("q", "{}")


def test_the_model_can_be_chosen_by_setting_and_is_tried_first(monkeypatch):
    monkeypatch.setenv("GEMINI_MODEL", "chosen")
    llm = GeminiLLM("key", client=ScriptedClient({}))
    assert llm.models[0] == "chosen"
    assert len(llm.models) == len(set(llm.models))


def test_gemini_phrasing_returns_text():
    response = types.GenerateContentResponse(
        candidates=[types.Candidate(content=types.Content(role="model", parts=[types.Part(text=" Mortgage leads. ")]))]
    )
    assert GeminiLLM("key", model="test-model", client=FakeClient(response)).phrase("q", {"rows": []}) == "Mortgage leads."
