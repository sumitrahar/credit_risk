"""Shared fixtures. The build outputs must exist first: `python -m src.build`."""
import pytest

from src.data_access import artifacts_ready, load_json, load_portfolio
from src.features import build_analysis_dataset
from src.ingest import run_ingest


@pytest.fixture(scope="session")
def ingested():
    return run_ingest()


@pytest.fixture(scope="session")
def clean(ingested):
    return ingested[0]


@pytest.fixture(scope="session")
def dq_log(ingested):
    return ingested[1].to_frame()


@pytest.fixture(scope="session")
def analysis(clean):
    return build_analysis_dataset(clean)


@pytest.fixture(scope="session")
def portfolio():
    if not artifacts_ready():
        pytest.skip("Run `python -m src.build` first")
    return load_portfolio()


@pytest.fixture(scope="session")
def model_report():
    return load_json("model_report.json")


@pytest.fixture(scope="session")
def anomaly_report():
    return load_json("anomaly_report.json")
