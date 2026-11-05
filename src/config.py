"""Shared paths and constants.

Every threshold and business assumption lives here, so they can all be read and
challenged in one place.
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
DOCS_DIR = ROOT / "docs"
RUNTIME_DIR = ROOT / "data" / "runtime"  # decision log; created at runtime, never committed

SEED = 42

# The latest date_opened in the bureau table is 2025-12-29, so the extract is assumed to be
# taken at the end of that year. Anything dated later than this cannot have happened yet.
SNAPSHOT_DATE = pd.Timestamp("2025-12-31")

# Every account has exactly 12 months of repayment history. The model scores an account at
# the end of that window, so bureau data is only used if it was visible by then.
BEHAVIOUR_MONTHS = 12

# Bands use left-closed intervals, so "<$40k" means strictly under $40,000.
INCOME_BINS = [0, 40_000, 60_000, 80_000, 100_000, float("inf")]
INCOME_LABELS = ["<$40k", "$40-60k", "$60-80k", "$80-100k", "$100k+"]
AGE_BINS = [0, 25, 35, 45, 55, 200]  # right-closed: 21-25, 26-35, ...
AGE_LABELS = ["21-25", "26-35", "36-45", "46-55", "56+"]

# Predicted probability of default (PD) bands, shown in the dashboard.
RISK_BAND_BINS = [0.0, 0.10, 0.20, 0.35, 1.0]
RISK_BAND_LABELS = ["Low", "Medium", "High", "Very High"]

# Fixed display order for ordered dimensions.
DIMENSION_ORDER = {
    "income_band": INCOME_LABELS,
    "age_band": AGE_LABELS,
    "risk_band": RISK_BAND_LABELS,
    "city_tier": ["Tier 1", "Tier 2", "Tier 3"],
}

# Cells with fewer accounts than this are greyed out or labelled as low sample.
MIN_SEGMENT_ACCOUNTS = 30

# LightGBM must beat logistic regression by this much AUC to justify the added complexity.
CHAMPION_MIN_AUC_GAIN = 0.005

# PSI (population stability index) monitoring thresholds, the usual industry rule of thumb.
PSI_WATCH = 0.10
PSI_ACT = 0.25

# Gemini settings. The key itself is only ever read from the environment or host secrets.
GEMINI_MODEL_DEFAULT = "gemini-3.1-flash-lite"
# Tried in order if the main model is overloaded, rate limited or retired. Free-tier models
# can return 503 at busy times, and each model has its own quota.
GEMINI_FALLBACK_MODELS = ("gemini-3.5-flash-lite", "gemini-3.6-flash", "gemini-3.8-flash")
MAX_QUESTION_CHARS = 300
MAX_QUESTIONS_PER_SESSION = 20
LLM_TIMEOUT_MS = 15_000
