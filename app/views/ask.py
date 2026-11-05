"""Ask the Portfolio view: natural-language questions, answered only from computed data."""
from __future__ import annotations

import html
import json

import streamlit as st

from components import feature_statement, plain
from src import assistant
from src.assistant import Answer, answer_question, find_api_key
from src.config import MAX_QUESTION_CHARS, MAX_QUESTIONS_PER_SESSION, PROCESSED_DIR
from src.decisions import record_answer_feedback
from src.query_engine import (
    DIMENSIONS,
    METRICS,
    Filter,
    QueryError,
    QuerySpec,
    allowed_values,
    result_table,
    run_query,
    template_answer,
)

SAMPLE_QUESTIONS = [
    "Which loan type has the highest default rate?",
    "What share of defaulted dollars comes from mortgages?",
    "What is the default rate for unemployed borrowers by income band?",
    "How many accounts are in the anomaly review list?",
]
DECLINE_EXAMPLES = [
    "What is the loss given default on mortgages?",
    "Will default rates rise next year?",
]


@st.cache_resource(show_spinner=False)
def _llm(api_key: str) -> assistant.LLM:
    return assistant.GeminiLLM(api_key)


@st.cache_resource(show_spinner=False)
def _shared_cache() -> dict:
    """Identical questions are answered once, across all sessions, to save free-tier quota."""
    return {}


def _normalise(question: str) -> str:
    return " ".join(question.lower().split())


def _remember(question: str, answer: Answer) -> None:
    """Show the newest answer first. Asking the same question again moves it back to the top."""
    history = st.session_state.setdefault("ask_history", [])
    history[:] = [item for item in history if _normalise(item[0]) != _normalise(question)]
    history.insert(0, (question, answer))


def _ask(question: str, portfolio) -> None:
    api_key = find_api_key()
    if not api_key:
        return
    cache = _shared_cache()
    key = _normalise(question)
    if key in cache:
        _remember(question, cache[key])
        return
    used = st.session_state.get("ask_used", 0)
    if used >= MAX_QUESTIONS_PER_SESSION:
        _remember(
            question,
            Answer(status="declined", text=f"This session has used its {MAX_QUESTIONS_PER_SESSION} questions. The guided query below still works."),
        )
        return
    st.session_state["ask_used"] = used + 1
    answer = answer_question(question, portfolio, _llm(api_key))
    if answer.status in ("answered", "declined"):
        cache[key] = answer
    _remember(question, answer)


def _show_answer(question: str, answer: Answer) -> None:
    with st.container(border=True):
        st.markdown(f"**{plain(question)}**")
        if answer.status == "answered":
            st.markdown(plain(answer.text))
            st.markdown(f'<p class="source-line">{html.escape(answer.source)}</p>', unsafe_allow_html=True)
        elif answer.status == "declined":
            st.info(plain(answer.text), icon=":material/block:")
        else:
            st.warning(plain(answer.text), icon=":material/warning:")
        for note in answer.notes:
            st.caption(plain(note))

        if answer.status == "answered":
            wording = (
                "Worded by Gemini. Every figure was checked against the computed result."
                if answer.written_by == "llm"
                else "Written by code from the computed result."
            )
            with st.expander("How this was computed"):
                st.caption(wording)
                st.markdown("Query chosen:")
                st.code(json.dumps(answer.query, indent=1), language="json")
                if answer.table is not None:
                    st.dataframe(answer.table, width="stretch", hide_index=True)
            feedback_key = f"feedback_{_normalise(question)}"
            if st.session_state.get(feedback_key):
                st.caption("Thanks: this answer was logged as wrong for review.")
            elif st.button("Mark as wrong", key=f"wrong_{feedback_key}", help="Logs the question, answer and query for review."):
                record_answer_feedback(question, answer.text, json.dumps(answer.query), "wrong")
                st.session_state[feedback_key] = True
                st.rerun()


def _guided_query(portfolio) -> None:
    st.subheader("Guided query")
    st.caption("Uses the same calculations with no LLM involved, so it works without a key and when the free-tier limit is reached.")
    metric_keys = list(METRICS)
    left, middle, right = st.columns(3)
    metric = left.selectbox(
        "Metric", metric_keys, index=metric_keys.index("default_rate_pct"),
        format_func=lambda key: METRICS[key].label.capitalize(), key="guided_metric",
    )
    group_by = middle.multiselect(
        "Group by (up to 2)", list(DIMENSIONS), default=["loan_type"], max_selections=2,
        format_func=lambda d: d.replace("_", " ").capitalize(), key="guided_group",
    )
    filter_dimension = right.selectbox(
        "Filter by", ["None", *DIMENSIONS],
        format_func=lambda d: d.replace("_", " ").capitalize(), key="guided_filter_dimension",
    )
    filters = []
    if filter_dimension != "None":
        values = st.multiselect(
            f"Values of {filter_dimension.replace('_', ' ')}", allowed_values(portfolio, filter_dimension), key=f"guided_filter_values_{filter_dimension}",
        )
        if values:
            filters = [Filter(dimension=filter_dimension, values=values)]
    try:
        result = run_query(QuerySpec(metric=metric, group_by=group_by, filters=filters), portfolio)
    except QueryError as error:
        st.info(plain(error), icon=":material/info:")
        return
    st.markdown(plain(template_answer(result)))
    source = f"Source: {result.source}. Scope: {result.scope_text} ({result.scope_accounts:,} accounts)."
    st.markdown(f'<p class="source-line">{html.escape(source)}</p>', unsafe_allow_html=True)
    st.dataframe(result_table(result), width="stretch", hide_index=True)


def render(context: dict) -> None:
    portfolio = context["portfolio"]
    st.title("Ask the Portfolio")
    st.caption("Ask in plain English. Answers cover the whole portfolio: the sidebar filters do not apply here, so say in the question if you want a subset.")

    feature_statement(
        trigger="An analyst submits a question. Gemini is called only then, never in the background.",
        false_positive="It answers something it should have declined, or reads the question as a different metric. For example, it treats \"exposure\" as originated amount when the analyst meant balance.",
        false_negative="It declines a question the data can answer, usually because the wording does not map to a listed metric.",
        human_step="The analyst checks the query and source under each answer and can mark it as wrong. Answers only inform review: nothing changes and nothing reaches a customer.",
    )
    st.write("")

    evaluation_path = PROCESSED_DIR / "assistant_eval.json"
    if evaluation_path.exists():
        evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
        st.caption(
            f"Live test on {evaluation['model']}: {evaluation['answerable_correct']} of {evaluation['answerable_total']} answerable questions "
            f"matched a direct calculation, and {evaluation['unanswerable_declined']} of {evaluation['unanswerable_total']} unanswerable questions were declined."
        )

    has_key = bool(find_api_key())
    if has_key:
        used = st.session_state.get("ask_used", 0)
        st.caption(f"Your question is sent to Google's Gemini API, so don't type personal details. No customer records are sent. {used} of {MAX_QUESTIONS_PER_SESSION} questions used this session.")
        with st.form("ask_form", clear_on_submit=True):
            question = st.text_input("Your question", max_chars=MAX_QUESTION_CHARS, placeholder="Which loan type has the highest default rate?")
            submitted = st.form_submit_button("Ask", type="primary")
        if submitted and question.strip():
            with st.spinner("Working it out from the data..."):
                _ask(question, portfolio)

        st.markdown('<p class="eyebrow">Try one</p>', unsafe_allow_html=True)
        columns = st.columns(len(SAMPLE_QUESTIONS))
        for column, sample in zip(columns, SAMPLE_QUESTIONS):
            if column.button(sample, key=f"sample_{sample}", width="stretch"):
                with st.spinner("Working it out from the data..."):
                    _ask(sample, portfolio)
        st.markdown('<p class="eyebrow">Questions it should decline</p>', unsafe_allow_html=True)
        columns = st.columns(len(SAMPLE_QUESTIONS))
        for column, sample in zip(columns, DECLINE_EXAMPLES):
            if column.button(sample, key=f"decline_{sample}", width="stretch"):
                with st.spinner("Working it out from the data..."):
                    _ask(sample, portfolio)

        for asked, answer in st.session_state.get("ask_history", []):
            _show_answer(asked, answer)
    else:
        st.info(
            "No Gemini API key is set, so natural-language questions are off. The rest of the dashboard works normally, "
            "and the guided query below gives answers from the same calculations. See the README to add a key.",
            icon=":material/key_off:",
        )

    st.divider()
    _guided_query(portfolio)
