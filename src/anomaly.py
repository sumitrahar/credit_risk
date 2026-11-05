"""Anomaly detection: flag accounts unusual against their own history or their peers.

Each account carries plain-language reasons built from its own figures. target_default is
never used to fit a detector. It is only used afterwards, to check whether flags mean anything.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.config import MIN_SEGMENT_ACCOUNTS, SEED

# Thresholds. Percentile thresholds are computed from the data at build time.
CONSECUTIVE_MISSED_MIN = 2
DPD_JUMP_PERCENTILE = 99
CLEAN_MONTHS = 9
PEER_Z_MIN = 3.5
BUREAU_DEBT_PERCENTILE = 99
BUREAU_WRITEOFFS_MIN = 3
ISOLATION_TOP_SHARE = 0.02
MIN_DETECTORS_FOR_REVIEW = 2

ISOLATION_FEATURES = {
    "rp_mean_dpd": ("average days past due", "{:.1f}"),
    "rp_max_dpd": ("worst days past due", "{:.0f}"),
    "rp_missed": ("missed payments", "{:.0f}"),
    "rp_dpd_trend": ("days past due trend", "{:+.1f}"),
    "rp_longest_missed_run": ("longest missed run", "{:.0f}"),
    "rp_recent_mean_dpd": ("recent average days past due", "{:.1f}"),
    "cur_lines": ("bureau lines", "{:.0f}"),
    "cur_writeoffs": ("bureau write-offs", "{:.0f}"),
    "cur_utilisation": ("bureau utilisation", "{:.0%}"),
    "cur_debt_to_income": ("bureau debt-to-income", "{:.2f}x"),
    "cur_max_dpd": ("worst bureau days past due", "{:.0f}"),
}

DETECTORS = {
    "consecutive_missed": {
        "name": "Consecutive missed payments",
        "compared_with": "Own history",
        "rule": f"{CONSECUTIVE_MISSED_MIN}+ missed payments in a row",
    },
    "dpd_jump": {
        "name": "Sudden deterioration",
        "compared_with": "Own history",
        "rule": f"Rise in average days past due, second half vs first, is in the top {100 - DPD_JUMP_PERCENTILE}% of accounts",
    },
    "late_shock": {
        "name": "Clean record, then a miss",
        "compared_with": "Own history",
        "rule": f"No days past due for {CLEAN_MONTHS} months, then a missed payment in the last 3",
    },
    "peer_dpd": {
        "name": "Far behind peers",
        "compared_with": "Peer segment",
        "rule": f"Average days past due is {PEER_Z_MIN}+ robust standard deviations above the same loan type and employment type",
    },
    "peer_bureau_debt": {
        "name": "High bureau debt for income band",
        "compared_with": "Peer segment",
        "rule": f"Bureau debt-to-income is above the {BUREAU_DEBT_PERCENTILE}th percentile of the same income band",
    },
    "bureau_writeoffs": {
        "name": "Multiple bureau write-offs",
        "compared_with": "Bureau history",
        "rule": f"{BUREAU_WRITEOFFS_MIN}+ written-off lines at other lenders",
    },
    "isolation_forest": {
        "name": "Unusual overall pattern",
        "compared_with": "Whole portfolio",
        "rule": f"Isolation Forest places the account in the most unusual {ISOLATION_TOP_SHARE:.0%} across all signals",
    },
}


def _longest_missed_run(missed_row: np.ndarray) -> tuple[int, int, int]:
    """Longest run of missed months, returned as (length, first month, last month)."""
    best = current = best_end = 0
    for month, value in enumerate(missed_row, start=1):
        current = current + 1 if value else 0
        if current > best:
            best, best_end = current, month
    return best, best_end - best + 1, best_end


def detect_anomalies(df: pd.DataFrame, repayment: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Run all detectors and return one row per account plus a summary report."""
    from sklearn.ensemble import IsolationForest  # imported here so the dashboard itself stays light

    data = df.set_index("account_id", drop=False)
    months = repayment.pivot(index="account_id", columns="payment_month", values="days_past_due").reindex(data.index)
    missed = repayment.assign(missed=repayment["payment_status"] == "Missed").pivot(
        index="account_id", columns="payment_month", values="missed"
    ).reindex(data.index)
    first_half = months[[1, 2, 3, 4, 5, 6]].mean(axis=1)
    second_half = months[[7, 8, 9, 10, 11, 12]].mean(axis=1)

    flags = pd.DataFrame(index=data.index)
    thresholds = {}

    # Own history
    flags["consecutive_missed"] = data["rp_longest_missed_run"] >= CONSECUTIVE_MISSED_MIN
    thresholds["dpd_jump_days"] = float(np.percentile(data["rp_dpd_trend"], DPD_JUMP_PERCENTILE))
    flags["dpd_jump"] = data["rp_dpd_trend"] > thresholds["dpd_jump_days"]
    clean_months = list(range(1, CLEAN_MONTHS + 1))
    last_months = list(range(CLEAN_MONTHS + 1, 13))
    flags["late_shock"] = (months[clean_months].max(axis=1) == 0) & missed[last_months].any(axis=1)

    # Peer segment: robust z-score, which is not distorted by the outliers it is looking for.
    peer_group = data["loan_type"] + " / " + data["employment_type"]
    group_sizes = peer_group.map(peer_group.value_counts())
    small_group = group_sizes < MIN_SEGMENT_ACCOUNTS
    peer_group = peer_group.where(~small_group, data["loan_type"])
    grouped = data["rp_mean_dpd"].groupby(peer_group)
    peer_median = grouped.transform("median")
    peer_mad = (data["rp_mean_dpd"] - peer_median).abs().groupby(peer_group).transform("median")
    peer_z = (data["rp_mean_dpd"] - peer_median) / (1.4826 * peer_mad.clip(lower=0.1))
    flags["peer_dpd"] = peer_z >= PEER_Z_MIN

    # Bureau: thin-file customers have no bureau data, so these detectors cannot fire for them.
    band_cutoff = data.groupby("income_band")["cur_debt_to_income"].transform(
        lambda s: s.quantile(BUREAU_DEBT_PERCENTILE / 100)
    )
    flags["peer_bureau_debt"] = data["cur_debt_to_income"] > band_cutoff
    flags["bureau_writeoffs"] = data["cur_writeoffs"] >= BUREAU_WRITEOFFS_MIN

    # Multivariate: finds unusual combinations no single rule is looking for.
    iso_matrix = data[list(ISOLATION_FEATURES)].fillna(0.0)
    forest = IsolationForest(n_estimators=200, contamination=ISOLATION_TOP_SHARE, random_state=SEED, n_jobs=-1)
    forest.fit(iso_matrix)
    iso_raw = -forest.score_samples(iso_matrix)
    iso_cutoff = np.quantile(iso_raw, 1 - ISOLATION_TOP_SHARE)
    flags["isolation_forest"] = iso_raw > iso_cutoff
    feature_percentile = iso_matrix.rank(pct=True)

    n_detectors = flags.sum(axis=1)
    flagged = n_detectors >= MIN_DETECTORS_FOR_REVIEW

    medians = iso_matrix.median()
    missed_values = missed.to_numpy()
    records = []
    for position, account in enumerate(data.index):
        row = data.iloc[position]
        fired = [name for name in DETECTORS if flags.iloc[position][name]]
        reasons = []
        if "consecutive_missed" in fired:
            length, start, end = _longest_missed_run(missed_values[position])
            reasons.append(f"{length} missed payments in a row (months {start}-{end}).")
        if "dpd_jump" in fired:
            reasons.append(
                f"Average days past due rose from {first_half.iloc[position]:.1f} in months 1-6 to "
                f"{second_half.iloc[position]:.1f} in months 7-12 (top {100 - DPD_JUMP_PERCENTILE}% of accounts: "
                f"over +{thresholds['dpd_jump_days']:.1f} days)."
            )
        if "late_shock" in fired:
            worst = months.iloc[position][last_months].max()
            reasons.append(f"No days past due for {CLEAN_MONTHS} months, then a missed payment at {worst:.0f} days past due.")
        if "peer_dpd" in fired:
            reasons.append(
                f"Average days past due of {row['rp_mean_dpd']:.1f} vs a peer median of {peer_median.iloc[position]:.1f} "
                f"({peer_group.iloc[position]}, z = {peer_z.iloc[position]:.1f})."
            )
        if "peer_bureau_debt" in fired:
            reasons.append(
                f"Bureau debt-to-income of {row['cur_debt_to_income']:.2f}x vs a {BUREAU_DEBT_PERCENTILE}th percentile "
                f"of {band_cutoff.iloc[position]:.2f}x for the {row['income_band']} income band."
            )
        if "bureau_writeoffs" in fired:
            reasons.append(f"{row['cur_writeoffs']:.0f} written-off lines at other lenders.")
        if "isolation_forest" in fired:
            top = feature_percentile.iloc[position].nlargest(2).index
            parts = [
                f"{ISOLATION_FEATURES[f][0]} {ISOLATION_FEATURES[f][1].format(iso_matrix.iloc[position][f])} "
                f"(portfolio median {ISOLATION_FEATURES[f][1].format(medians[f])})"
                for f in top
            ]
            reasons.append("Led by " + " and ".join(parts) + ".")
        records.append(
            {
                "account_id": account,
                "detectors_json": json.dumps(fired),
                "reasons_json": json.dumps(reasons),
            }
        )

    out = pd.DataFrame(records)
    out["n_detectors"] = n_detectors.to_numpy()
    out["flagged"] = flagged.to_numpy().astype(int)
    out["anomaly_score"] = (pd.Series(iso_raw).rank(pct=True) * 100).to_numpy()
    out["peer_group"] = peer_group.to_numpy()
    out["peer_median_dpd"] = peer_median.to_numpy()
    out["peer_z"] = peer_z.to_numpy()
    out["first_half_mean_dpd"] = first_half.to_numpy()
    out["second_half_mean_dpd"] = second_half.to_numpy()

    # Rank by detectors fired, then how unusual the account is, then money at stake.
    review = out.assign(balance=data["latest_balance"].to_numpy()).sort_values(
        ["flagged", "n_detectors", "anomaly_score", "balance"], ascending=False
    )
    out["review_rank"] = pd.Series(np.arange(1, len(out) + 1), index=review.index).reindex(out.index).to_numpy()
    out.loc[out["flagged"] == 0, "review_rank"] = pd.NA
    out["review_rank"] = out["review_rank"].astype("Int64")

    target = data["target_default"]
    base_rate = float(target.mean())
    detector_rows = []
    for name, info in DETECTORS.items():
        mask = flags[name]
        rate = float(target[mask].mean())
        detector_rows.append(
            {
                "detector": name,
                **info,
                "accounts": int(mask.sum()),
                "in_review_list": int((mask & flagged).sum()),
                "default_rate": rate,
                "lift": rate / base_rate,
            }
        )
    flagged_rate = float(target[flagged].mean())
    report = {
        "accounts_flagged": int(flagged.sum()),
        "share_flagged": float(flagged.mean()),
        "min_detectors_for_review": MIN_DETECTORS_FOR_REVIEW,
        "flagged_default_rate": flagged_rate,
        "unflagged_default_rate": float(target[~flagged].mean()),
        "base_default_rate": base_rate,
        "lift": flagged_rate / base_rate,
        "defaults_in_list": int(target[flagged].sum()),
        "share_of_defaults_in_list": float(target[flagged].sum() / target.sum()),
        "accounts_any_detector": int((n_detectors >= 1).sum()),
        "detectors": detector_rows,
        "thresholds": thresholds,
    }
    return out, report
