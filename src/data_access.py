"""Read the files produced by `python -m src.build`. Used by the dashboard and the assistant."""
from __future__ import annotations

import json

import pandas as pd

from src.config import PROCESSED_DIR

REQUIRED_FILES = [
    "analysis_dataset.parquet",
    "account_scores.parquet",
    "anomaly_flags.parquet",
    "repayment_clean.parquet",
    "bureau_clean.parquet",
    "collections_clean.parquet",
    "dq_log.csv",
    "insights.json",
    "model_report.json",
    "anomaly_report.json",
]


def artifacts_ready() -> bool:
    return all((PROCESSED_DIR / name).exists() for name in REQUIRED_FILES)


def load_json(name: str):
    return json.loads((PROCESSED_DIR / name).read_text(encoding="utf-8"))


def load_portfolio() -> pd.DataFrame:
    """One row per account: the analysis dataset plus model scores and anomaly results."""
    analysis = pd.read_parquet(PROCESSED_DIR / "analysis_dataset.parquet")
    scores = pd.read_parquet(PROCESSED_DIR / "account_scores.parquet")
    flags = pd.read_parquet(PROCESSED_DIR / "anomaly_flags.parquet")
    df = analysis.merge(scores, on="account_id", validate="one_to_one").merge(
        flags, on="account_id", validate="one_to_one"
    )
    # Readable labels, so these can be used as filters and groupings.
    df["default_status"] = df["target_default"].map({1: "Defaulted", 0: "Performing"})
    df["review_status"] = df["flagged"].map({1: "In review list", 0: "Not in review list"})
    df["bureau_record"] = df["thin_file"].map({1: "No bureau record", 0: "Has bureau record"})
    df["secured_status"] = df["collateral_flag"].map({1: "Secured", 0: "Unsecured"})
    return df


def load_table(name: str) -> pd.DataFrame:
    return pd.read_parquet(PROCESSED_DIR / f"{name}.parquet")


def load_dq_log() -> pd.DataFrame:
    return pd.read_csv(PROCESSED_DIR / "dq_log.csv")
