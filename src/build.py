"""Build everything the dashboard reads: `python -m src.build`.

raw CSVs -> cleaning and data-quality log -> analysis dataset -> insights -> model -> anomalies -> PDFs
"""
from __future__ import annotations

import json
import time

from src.analytics import compute_insights
from src.anomaly import detect_anomalies
from src.config import PROCESSED_DIR
from src.features import build_analysis_dataset
from src.ingest import run_ingest
from src.model import train_and_evaluate


def _save_json(payload, name: str) -> None:
    (PROCESSED_DIR / name).write_text(json.dumps(payload, indent=1, default=float), encoding="utf-8")


def main(with_docs: bool = True) -> None:
    started = time.time()
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print("1/5 Cleaning the raw tables and writing the data-quality log")
    clean, log = run_ingest()
    log.to_frame().to_csv(PROCESSED_DIR / "dq_log.csv", index=False)
    clean["repayment"].to_parquet(PROCESSED_DIR / "repayment_clean.parquet", index=False)
    clean["bureau"].to_parquet(PROCESSED_DIR / "bureau_clean.parquet", index=False)
    clean["collections"].to_parquet(PROCESSED_DIR / "collections_clean.parquet", index=False)

    print("2/5 Building the analysis dataset (one row per account)")
    df = build_analysis_dataset(clean)
    df.to_parquet(PROCESSED_DIR / "analysis_dataset.parquet", index=False)

    print("3/5 Computing portfolio insights")
    _save_json(compute_insights(df, clean["collections"]), "insights.json")

    print("4/5 Training and evaluating the models (a few minutes)")
    scores, report = train_and_evaluate(df, clean)
    scores.to_parquet(PROCESSED_DIR / "account_scores.parquet", index=False)
    _save_json(report, "model_report.json")

    print("5/5 Running anomaly detection")
    flags, anomaly_report = detect_anomalies(df, clean["repayment"])
    flags.to_parquet(PROCESSED_DIR / "anomaly_flags.parquet", index=False)
    _save_json(anomaly_report, "anomaly_report.json")

    if with_docs:
        from src.make_docs import build_docs

        print("Writing the two one-page PDFs")
        build_docs()

    logistic = report["models"]["Logistic regression"]
    print(
        f"\nDone in {time.time() - started:.0f}s. {len(df):,} accounts. Champion: {report['champion']}. "
        f"AUC {logistic['auc']:.3f}, Gini {logistic['gini']:.3f}, KS {logistic['ks']:.3f}. "
        f"{anomaly_report['accounts_flagged']:,} accounts in the review list."
    )


if __name__ == "__main__":
    main()
