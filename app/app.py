"""Dashboard entry point: `streamlit run app/app.py`."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for path in (str(Path(__file__).resolve().parent), str(ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

import streamlit as st
from dotenv import load_dotenv

from components import PAGE_STYLE
from src.config import DIMENSION_ORDER
from src.data_access import artifacts_ready, load_dq_log, load_json, load_portfolio, load_table

st.set_page_config(page_title="Credit Risk Portfolio Monitor", page_icon=":material/monitoring:", layout="wide")
st.markdown(PAGE_STYLE, unsafe_allow_html=True)

# The Gemini key comes from a local .env file or the host's secrets store, never from the code.
load_dotenv(ROOT / ".env")
try:
    for name in ("GEMINI_API_KEY", "GEMINI_MODEL"):
        if name in st.secrets:
            os.environ.setdefault(name, str(st.secrets[name]))
except Exception:  # no secrets file is normal when running locally
    pass

if not artifacts_ready():
    with st.spinner("Building the data and model for the first time. This takes a few minutes."):
        from src.build import main as build_all

        build_all()


@st.cache_data(show_spinner=False)
def get_data() -> dict:
    return {
        "portfolio": load_portfolio(),
        "repayment": load_table("repayment_clean"),
        "bureau": load_table("bureau_clean"),
        "collections": load_table("collections_clean"),
        "dq_log": load_dq_log(),
        "insights": load_json("insights.json"),
        "model_report": load_json("model_report.json"),
        "anomaly_report": load_json("anomaly_report.json"),
    }


data = get_data()
portfolio = data["portfolio"]

PAGES = ["Portfolio", "Account", "Ask the Portfolio", "Anomaly Review", "Model & Monitoring", "Data Notes"]
if "pending_page" in st.session_state:
    st.session_state["page"] = st.session_state.pop("pending_page")
elif "page" not in st.session_state and st.query_params.get("view") in PAGES:
    # A link such as ?view=Anomaly+Review opens straight on that view.
    st.session_state["page"] = st.query_params["view"]

with st.sidebar:
    st.markdown("### Credit Risk Monitor")
    st.caption("Synthetic portfolio of 14,685 loan accounts")
    page = st.radio("View", PAGES, key="page", label_visibility="collapsed")

    st.divider()
    st.markdown("**Filters**")
    st.caption("Apply to Portfolio and Anomaly Review. Leave empty to include everything.")
    FILTERS = {
        "loan_type": "Loan type",
        "employment_type": "Employment",
        "region": "Region",
        "income_band": "Income band",
        "age_band": "Age band",
        "risk_band": "Risk band",
        "origination_year": "Origination year",
    }
    filtered = portfolio
    active = []
    for column, label in FILTERS.items():
        options = DIMENSION_ORDER.get(column) or sorted(portfolio[column].unique().tolist())
        chosen = st.multiselect(label, options, key=f"filter_{column}", placeholder="All")
        if chosen:
            filtered = filtered[filtered[column].isin(chosen)]
            active.append(f"{label}: {', '.join(str(c) for c in chosen)}")

context = {**data, "filtered": filtered, "active_filters": active}

if page == "Portfolio":
    from views.portfolio import render
elif page == "Account":
    from views.account import render
elif page == "Ask the Portfolio":
    from views.ask import render
elif page == "Anomaly Review":
    from views.anomaly_review import render
elif page == "Model & Monitoring":
    from views.model_view import render
else:
    from views.data_notes import render

render(context)

# Streamlit keeps the scroll position between views, so start each newly opened view at the top.
if st.session_state.get("last_page") != page:
    st.session_state["last_page"] = page
    st.html(
        """<script>document.querySelector('[data-testid="stMain"]').scrollTo(0, 0);</script>""",
        unsafe_allow_javascript=True,
    )
