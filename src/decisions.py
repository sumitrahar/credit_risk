"""Analyst decision log, stored in SQLite.

Every decision is appended, so history is kept and the latest one per account wins.
On free hosting the disk can reset on a reboot or redeploy, which the app tells the analyst.
Production would use a shared database instead.
"""
from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.config import RUNTIME_DIR

VALID_DECISIONS = ("Confirm", "Dismiss")


def _db_path() -> Path:
    # Tests point this at a temporary file through the environment.
    path = Path(os.environ.get("DECISIONS_DB", RUNTIME_DIR / "decisions.db"))
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _connect() -> sqlite3.Connection:
    connection = sqlite3.connect(_db_path())
    connection.execute(
        """CREATE TABLE IF NOT EXISTS decisions (
               id INTEGER PRIMARY KEY AUTOINCREMENT,
               account_id TEXT NOT NULL,
               decision TEXT NOT NULL,
               note TEXT,
               detectors TEXT,
               decided_at TEXT NOT NULL
           )"""
    )
    connection.execute(
        """CREATE TABLE IF NOT EXISTS answer_feedback (
               id INTEGER PRIMARY KEY AUTOINCREMENT,
               question TEXT NOT NULL,
               answer TEXT,
               query TEXT,
               verdict TEXT NOT NULL,
               created_at TEXT NOT NULL
           )"""
    )
    return connection


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def record_decision(account_id: str, decision: str, note: str = "", detectors: list[str] | None = None) -> None:
    if decision not in VALID_DECISIONS:
        raise ValueError(f"Decision must be one of {VALID_DECISIONS}")
    with _connect() as connection:
        connection.execute(
            "INSERT INTO decisions (account_id, decision, note, detectors, decided_at) VALUES (?, ?, ?, ?, ?)",
            (account_id, decision, note.strip(), json.dumps(detectors or []), _now()),
        )


def all_decisions() -> pd.DataFrame:
    with _connect() as connection:
        return pd.read_sql_query(
            "SELECT account_id, decision, note, detectors, decided_at FROM decisions ORDER BY id", connection
        )


def latest_decisions() -> pd.DataFrame:
    """The most recent decision for each account."""
    decisions = all_decisions()
    return decisions.drop_duplicates("account_id", keep="last").set_index("account_id")


def detector_outcomes() -> pd.DataFrame:
    """Confirm and dismiss counts per detector, used to tune thresholds over time."""
    latest = latest_decisions()
    if latest.empty:
        return pd.DataFrame(columns=["detector", "Confirm", "Dismiss", "dismiss_rate"])
    exploded = latest.assign(detector=latest["detectors"].map(json.loads)).explode("detector").dropna(subset=["detector"])
    table = exploded.groupby(["detector", "decision"]).size().unstack(fill_value=0)
    for column in VALID_DECISIONS:
        if column not in table:
            table[column] = 0
    table["dismiss_rate"] = table["Dismiss"] / (table["Confirm"] + table["Dismiss"])
    return table.reset_index()[["detector", "Confirm", "Dismiss", "dismiss_rate"]]


def record_answer_feedback(question: str, answer: str, query: str, verdict: str) -> None:
    with _connect() as connection:
        connection.execute(
            "INSERT INTO answer_feedback (question, answer, query, verdict, created_at) VALUES (?, ?, ?, ?, ?)",
            (question, answer, query, verdict, _now()),
        )


def all_answer_feedback() -> pd.DataFrame:
    with _connect() as connection:
        return pd.read_sql_query(
            "SELECT question, answer, query, verdict, created_at FROM answer_feedback ORDER BY id", connection
        )
