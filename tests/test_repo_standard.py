"""The submission rules in the brief, checked so they cannot quietly break."""
import re
import subprocess

import pandas as pd
import pytest
from pypdf import PdfReader

from src.config import DOCS_DIR, RAW_DIR, ROOT

EXPECTED_ROWS = {
    "customers.csv": 10_000,
    "loan_accounts.csv": 14_685,
    "bureau_credit.csv": 22_979,
    "repayment_history.csv": 176_220,
    "collections_events.csv": 8_635,
}


@pytest.mark.parametrize("name", ["architecture.pdf", "solution_overview.pdf"])
def test_each_document_is_exactly_one_page(name):
    assert len(PdfReader(DOCS_DIR / name).pages) == 1


def test_docs_folder_holds_exactly_the_two_documents():
    assert sorted(p.name for p in DOCS_DIR.iterdir()) == ["architecture.pdf", "solution_overview.pdf"]


def test_architecture_names_the_llm_call_and_the_data_sent():
    text = PdfReader(DOCS_DIR / "architecture.pdf").pages[0].extract_text()
    for phrase in ("Gemini API", "Sent to Gemini", "Never sent", "Ingestion", "Analytics", "Model", "Dashboard", "Anomaly"):
        assert phrase in text


def test_overview_covers_what_the_brief_lists():
    text = PdfReader(DOCS_DIR / "solution_overview.pdf").pages[0].extract_text()
    for phrase in ("AUC", "KS", "Gini", "Trigger", "False positive", "False negative", "Human step", "Top 3 portfolio insights", "Key assumptions", "IT and data teams"):
        assert phrase in text


def test_every_requirement_is_pinned():
    lines = [line.strip() for line in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(lines) > 16
    for line in lines:
        assert re.match(r"^[A-Za-z0-9_.\-\[\]]+==[^=\s]+( ;.*)?$", line), line


def test_env_example_has_names_but_no_values():
    for line in (ROOT / ".env.example").read_text(encoding="utf-8").splitlines():
        name, _, value = line.partition("=")
        assert name and value == "", line


def test_the_five_raw_csvs_are_present_and_complete():
    assert sorted(p.name for p in RAW_DIR.glob("*.csv")) == sorted(EXPECTED_ROWS)
    for name, rows in EXPECTED_ROWS.items():
        assert len(pd.read_csv(RAW_DIR / name)) == rows, name


def test_layout_matches_the_brief():
    for path in ["README.md", "requirements.txt", ".env.example", "app/app.py", "src", "data/raw", "docs/architecture.pdf", "docs/solution_overview.pdf", "tests"]:
        assert (ROOT / path).exists(), path


@pytest.mark.parametrize(
    "phrase",
    [
        "Python 3.12",
        "pip install -r requirements.txt",
        "python -m src.build",
        "streamlit run app/app.py",
        "GEMINI_API_KEY",
        "If no key is set",
        "Where each item in the brief lives",
    ],
)
def test_readme_includes_what_the_brief_requires(phrase):
    assert phrase in (ROOT / "README.md").read_text(encoding="utf-8")


def test_readme_has_exactly_three_run_commands():
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    section = text.split("## Run it")[1].split("## ")[0]
    assert section.count("```bash") == 3


def test_readme_generated_sections_are_filled_in():
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    for name in ["DATA_QUALITY", "INSIGHTS", "MODEL_RESULTS", "SENSITIVITY", "ANOMALY"]:
        content = text.split(f"<!-- {name}:START -->")[1].split(f"<!-- {name}:END -->")[0]
        assert len(content.strip()) > 100, name


def test_the_key_in_the_local_env_file_appears_nowhere_else():
    env_file = ROOT / ".env"
    if not env_file.exists():
        pytest.skip("No local .env file")
    values = [line.partition("=")[2].strip() for line in env_file.read_text(encoding="utf-8").splitlines()]
    values = [value for value in values if len(value) >= 20]
    ignored = {".venv", "__pycache__", ".pytest_cache", "runtime"}
    for path in ROOT.rglob("*"):
        if not path.is_file() or path.name == ".env" or ignored & set(path.relative_to(ROOT).parts):
            continue
        content = path.read_bytes()
        for value in values:
            assert value.encode() not in content, f"The key from .env appears in {path}"


SECRET_PATTERNS = [
    r"AIza[0-9A-Za-z_\-]{30,}",     # Google API key
    r"\bAQ\.[0-9A-Za-z_\-]{30,}",   # Google API key, newer format
    r"sk-[A-Za-z0-9_\-]{20,}",      # other common provider keys
    r"ghp_[A-Za-z0-9]{30,}",        # GitHub token
    r"GEMINI_API_KEY\s*=\s*[\"']?[A-Za-z0-9_\-]{20,}",
]


def test_no_secrets_in_any_file_that_would_be_committed():
    ignored = {".venv", "__pycache__", ".pytest_cache", "runtime"}
    checked = 0
    for path in ROOT.rglob("*"):
        if not path.is_file() or ignored & set(path.relative_to(ROOT).parts):
            continue
        if path.name == ".env" or path.suffix in {".parquet", ".joblib", ".pdf", ".png", ".db"}:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        checked += 1
        for pattern in SECRET_PATTERNS:
            assert not re.search(pattern, text), f"{path}: matches {pattern}"
    assert checked > 20


def test_gitignore_keeps_secrets_and_runtime_files_out():
    ignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
    for entry in [".env", ".venv/", "data/runtime/", ".streamlit/secrets.toml"]:
        assert entry in ignore
    assert subprocess.run(["git", "--version"], capture_output=True).returncode == 0
