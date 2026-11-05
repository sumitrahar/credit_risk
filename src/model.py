"""Default model: logistic regression baseline against LightGBM.

Validation rules:
- Folds are grouped by customer, because customers hold several loans and one default
  predicts another. A random split would leak that information and inflate the result.
- Scores shown in the dashboard are out-of-fold, so no account is scored by a model that
  saw its own customer.

Left out of the model on purpose:
- Gender, marital status, region and age: fair-lending risk (see the sensitivity table).
- Collections: a consequence of delinquency, and they add nothing.
- rp_pay_ratio: exactly 1 - rp_missed / 12, because each month is paid in full or not at all.
- Loan purpose and customer_since: unreliable fields (see the data-quality log).
"""
from __future__ import annotations

import json
import warnings

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import KFold, StratifiedGroupKFold
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from src.config import (
    CHAMPION_MIN_AUC_GAIN,
    MODELS_DIR,
    PSI_ACT,
    PSI_WATCH,
    RISK_BAND_BINS,
    RISK_BAND_LABELS,
    SEED,
    SNAPSHOT_DATE,
)
from src.features import (
    AVERAGE_DAYS_IN_MONTH,
    bureau_features_all_lines,
    bureau_features_point_in_time,
    finalise_bureau_columns,
    repayment_features,
)

APPLICATION_NUMERIC = ["annual_income", "years_employed", "years_at_residence", "interest_rate_pct"]
APPLICATION_SKEWED = ["loan_to_income", "installment_to_income"]
REPAYMENT_NUMERIC = ["rp_missed", "rp_late", "rp_mean_dpd", "rp_longest_missed_run", "rp_dpd_trend"]
BUREAU_NUMERIC = [
    "bu_writeoffs", "bu_max_dpd", "bu_utilisation", "bu_credit_age_months",
    "thin_file", "bu_no_visible_lines", "bu_dpd_unknown", "bu_no_open_balance",
]
BUREAU_SKEWED = ["bu_debt_to_income"]
CATEGORICAL = ["employment_type", "education_level", "housing_type", "loan_type"]
SENSITIVE_CATEGORICAL = ["gender", "marital_status", "region"]

# Extra engineered features tested and removed because they added nothing. Kept here so the
# sensitivity table can show that removing them did not cost any AUC.
REMOVED_NUMERIC = [
    "num_dependents", "tenure_months", "rp_max_dpd", "rp_recent_mean_dpd",
    "bu_lines", "bu_active", "bu_mean_dpd", "bu_nonbank_share",
]
REMOVED_SKEWED = ["loan_amount", "bu_outstanding"]
REMOVED_CATEGORICAL = ["city_tier", "disbursal_channel"]

# Readable names and display formats for key drivers and importance charts.
FEATURE_INFO = {
    "annual_income": ("Annual income", "money"),
    "years_employed": ("Years employed", "float"),
    "years_at_residence": ("Years at residence", "float"),
    "interest_rate_pct": ("Interest rate", "pct_points"),
    "loan_to_income": ("Loan-to-income", "ratio"),
    "installment_to_income": ("Installments as a share of income", "share"),
    "rp_missed": ("Missed payments (12m)", "int"),
    "rp_late": ("Late payments (12m)", "int"),
    "rp_mean_dpd": ("Average days past due (12m)", "float"),
    "rp_longest_missed_run": ("Longest run of missed payments", "int"),
    "rp_dpd_trend": ("Days past due trend, second half vs first", "float"),
    "bu_writeoffs": ("Bureau write-offs", "int"),
    "bu_max_dpd": ("Worst bureau days past due", "int"),
    "bu_utilisation": ("Bureau credit utilisation", "share"),
    "bu_credit_age_months": ("Bureau credit age (months)", "int"),
    "thin_file": ("No bureau record", "flag"),
    "bu_no_visible_lines": ("No bureau lines at scoring date", "flag"),
    "bu_dpd_unknown": ("Bureau days past due unknown", "flag"),
    "bu_no_open_balance": ("No open bureau line with a known balance", "flag"),
    "bu_debt_to_income": ("Bureau debt-to-income", "ratio"),
    "employment_type": ("Employment type", "text"),
    "education_level": ("Education", "text"),
    "housing_type": ("Housing", "text"),
    "loan_type": ("Loan type", "text"),
}


def format_feature_value(feature: str, value) -> str:
    kind = FEATURE_INFO[feature][1]
    if kind == "text":
        return str(value)
    if pd.isna(value):
        return "not known"
    if kind == "money":
        return f"${value:,.0f}"
    if kind == "int":
        return f"{value:,.0f}"
    if kind == "float":
        return f"{value:,.1f}"
    if kind == "ratio":
        return f"{value:.2f}x"
    if kind == "share":
        return f"{value:.0%}"
    if kind == "pct_points":
        return f"{value:.2f}%"
    return "Yes" if value else "No"  # flag


def primary_features() -> dict[str, list[str]]:
    return {
        "numeric": APPLICATION_NUMERIC + REPAYMENT_NUMERIC + BUREAU_NUMERIC,
        "skewed": APPLICATION_SKEWED + BUREAU_SKEWED,
        "categorical": list(CATEGORICAL),
    }


def _columns(features: dict[str, list[str]]) -> list[str]:
    return features["numeric"] + features["skewed"] + features["categorical"]


def make_logistic(features: dict[str, list[str]]) -> Pipeline:
    prep = ColumnTransformer(
        [
            (
                "log",
                make_pipeline(
                    SimpleImputer(strategy="median"),
                    FunctionTransformer(np.log1p, feature_names_out="one-to-one"),
                    StandardScaler(),
                ),
                features["skewed"],
            ),
            ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), features["numeric"]),
            ("cat", OneHotEncoder(handle_unknown="ignore"), features["categorical"]),
        ],
        sparse_threshold=0,
    )
    return Pipeline([("prep", prep), ("clf", LogisticRegression(C=0.1, max_iter=2000, random_state=SEED))])


# A deliberately small search: the data has modest signal, so heavier tuning would only fit noise.
LIGHTGBM_CANDIDATES = [
    {"n_estimators": 300, "learning_rate": 0.02, "num_leaves": 4, "min_child_samples": 80},
    {"n_estimators": 300, "learning_rate": 0.02, "num_leaves": 8, "min_child_samples": 80},
    {"n_estimators": 500, "learning_rate": 0.01, "num_leaves": 8, "min_child_samples": 120},
]


def make_lightgbm(features: dict[str, list[str]], params: dict) -> Pipeline:
    from lightgbm import LGBMClassifier

    # LightGBM handles missing values natively, so numeric columns pass through untouched.
    prep = ColumnTransformer(
        [
            ("num", "passthrough", features["numeric"] + features["skewed"]),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), features["categorical"]),
        ],
        sparse_threshold=0,
    )
    classifier = LGBMClassifier(
        subsample=0.8, subsample_freq=1, colsample_bytree=0.6, reg_lambda=5.0,
        random_state=SEED, n_jobs=-1, verbose=-1, deterministic=True, force_row_wise=True, **params,
    )
    return Pipeline([("prep", prep), ("clf", classifier)])


def ks_statistic(y, scores) -> float:
    fpr, tpr, _ = roc_curve(y, scores)
    return float((tpr - fpr).max())


def score_metrics(y, scores) -> dict:
    auc = float(roc_auc_score(y, scores))
    return {"auc": auc, "gini": 2 * auc - 1, "ks": ks_statistic(y, scores)}


def cross_validate(make_model, features, df, grouped=True) -> dict:
    """Fit across 5 folds and return out-of-fold scores, fold metrics and the fitted models."""
    X = df[_columns(features)]
    y = df["target_default"].to_numpy()
    if grouped:
        splits = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED).split(X, y, df["customer_id"])
    else:
        splits = KFold(n_splits=5, shuffle=True, random_state=SEED).split(X, y)

    oof = np.zeros(len(df))
    fold_of = np.zeros(len(df), dtype=int)
    fold_aucs, models = [], []
    for fold, (train, test) in enumerate(splits):
        model = make_model()
        model.fit(X.iloc[train], y[train])
        oof[test] = model.predict_proba(X.iloc[test])[:, 1]
        fold_of[test] = fold
        fold_aucs.append(roc_auc_score(y[test], oof[test]))
        models.append(model)

    summary = score_metrics(y, oof)
    summary["auc_fold_sd"] = float(np.std(fold_aucs))
    return {"oof": oof, "fold_of": fold_of, "metrics": summary, "models": models}


def bootstrap_auc_interval(df: pd.DataFrame, scores: np.ndarray, n_resamples: int = 300) -> list[float]:
    """95% interval for AUC, resampling whole customers so related loans stay together."""
    rng = np.random.default_rng(SEED)
    y = df["target_default"].to_numpy()
    index_by_customer = list(df.groupby("customer_id").indices.values())
    aucs = []
    for _ in range(n_resamples):
        picks = rng.integers(0, len(index_by_customer), len(index_by_customer))
        rows = np.concatenate([index_by_customer[i] for i in picks])
        aucs.append(roc_auc_score(y[rows], scores[rows]))
    return [float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))]


def out_of_time_test(make_model, features, df) -> dict:
    """Train on older vintages and test on the newest, excluding customers seen in training."""
    train = df[df["origination_year"] <= 2021]
    test = df[(df["origination_year"] == 2023) & ~df["customer_id"].isin(train["customer_id"])]
    columns = _columns(features)
    model = make_model().fit(train[columns], train["target_default"])
    scores = model.predict_proba(test[columns])[:, 1]
    result = score_metrics(test["target_default"], scores)
    result.update(train_accounts=int(len(train)), test_accounts=int(len(test)))
    return result


def _with_bureau(df: pd.DataFrame, bureau_frame: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    aligned = bureau_frame.reindex(out["account_id"])
    for column in aligned.columns:
        out[column] = aligned[column].to_numpy()
    return finalise_bureau_columns(out)


def _with_repayment(df: pd.DataFrame, repayment: pd.DataFrame, months: int) -> pd.DataFrame:
    out = df.copy()
    aligned = repayment_features(repayment, months).reindex(out["account_id"])
    for column in aligned.columns:
        out[column] = aligned[column].to_numpy()
    return out


def sensitivity_table(df: pd.DataFrame, clean: dict[str, pd.DataFrame]) -> list[dict]:
    """Show how each modelling choice moves AUC, using logistic regression throughout."""
    primary = primary_features()
    application_only = {
        "numeric": list(APPLICATION_NUMERIC),
        "skewed": list(APPLICATION_SKEWED),
        "categorical": list(CATEGORICAL),
    }
    application_and_bureau = {
        "numeric": APPLICATION_NUMERIC + BUREAU_NUMERIC,
        "skewed": APPLICATION_SKEWED + BUREAU_SKEWED,
        "categorical": list(CATEGORICAL),
    }
    all_engineered = {
        "numeric": primary["numeric"] + REMOVED_NUMERIC,
        "skewed": primary["skewed"] + REMOVED_SKEWED,
        "categorical": primary["categorical"] + REMOVED_CATEGORICAL,
    }
    with_age = {**primary, "numeric": primary["numeric"] + ["age"]}
    with_demographics = {
        **primary,
        "numeric": primary["numeric"] + ["age"],
        "categorical": primary["categorical"] + SENSITIVE_CATEGORICAL,
    }
    with_collections = {**primary, "numeric": primary["numeric"] + ["co_events", "co_recovered"]}

    all_lines = bureau_features_all_lines(clean["loans"], clean["bureau"])
    first_opened = clean["bureau"].groupby("customer_id")["date_opened"].min()
    opened = pd.Series(clean["loans"]["customer_id"].map(first_opened).to_numpy(), index=all_lines.index)
    all_lines["bu_credit_age_months"] = (SNAPSHOT_DATE - opened).dt.days / AVERAGE_DAYS_IN_MONTH

    def windowed(months: int) -> pd.DataFrame:
        with_bureau = _with_bureau(df, bureau_features_point_in_time(clean["loans"], clean["bureau"], months))
        return _with_repayment(with_bureau, clean["repayment"], months)

    at_origination = _with_bureau(df, bureau_features_point_in_time(clean["loans"], clean["bureau"], 0))

    variants = [
        ("Final model: 12 months of repayment, date-limited bureau", primary, df, True, "The reported model."),
        ("All 36 engineered features", all_engineered, df, True, "The final 24 were chosen by removing redundant features; this shows nothing was lost."),
        ("Customer and loan data only", application_only, df, True, "Floor: no behaviour or bureau data."),
        ("Customer, loan and bureau at origination", application_and_bureau, at_origination, True, "A true at-application score."),
        ("3 months of repayment", primary, windowed(3), True, "An earlier scoring point, in case default can occur before month 12."),
        ("6 months of repayment", primary, windowed(6), True, "An earlier scoring point, in case default can occur before month 12."),
        ("Final model + age", with_age, df, True, "Age is left out pending compliance review; this is what it would add."),
        ("Final model + age, gender, marital status, region", with_demographics, df, True, "Shows what the fair-lending exclusions cost."),
        ("Bureau lines ignoring dates (look-ahead)", primary, _with_bureau(df, all_lines), True, "Leak: uses lines opened after the scoring date."),
        ("Final model + collections (leak check)", with_collections, df, True, "Collections follow delinquency, so they are excluded."),
        ("Random folds, not grouped by customer", primary, df, False, "Leak: the same customer appears in training and test."),
    ]
    rows = []
    for name, features, frame, grouped, note in variants:
        result = cross_validate(lambda f=features: make_logistic(f), features, frame, grouped=grouped)
        rows.append({"variant": name, **result["metrics"], "feature_count": len(_columns(features)), "note": note})
    return rows


def parent_feature(transformed_name: str, features: dict[str, list[str]]) -> str:
    """Map a transformed column name back to the original feature."""
    _, _, name = transformed_name.partition("__")
    if name in features["numeric"] or name in features["skewed"]:
        return name
    for column in features["categorical"]:
        if name.startswith(f"{column}_"):
            return column
    raise KeyError(transformed_name)


def _group_by_parent(values: np.ndarray, names, features, index) -> pd.DataFrame:
    """Add one-hot columns back together under their original feature."""
    parents = [parent_feature(name, features) for name in names]
    return pd.DataFrame(values, columns=parents, index=index).T.groupby(level=0).sum().T


def _sigmoid(x):
    return 1 / (1 + np.exp(-x))


def logistic_contributions(model: Pipeline, X_train: pd.DataFrame, X: pd.DataFrame, features) -> tuple[pd.DataFrame, float]:
    """Each feature's contribution to the log-odds, relative to the average training account.

    log-odds = base + sum(coefficient * (value - training mean)), so the parts add up exactly.
    """
    prep, classifier = model.named_steps["prep"], model.named_steps["clf"]
    means = prep.transform(X_train).mean(axis=0)
    coefficients = classifier.coef_[0]
    raw = coefficients * (prep.transform(X) - means)
    base = float(classifier.intercept_[0] + coefficients @ means)
    return _group_by_parent(raw, prep.get_feature_names_out(), features, X.index), base


def out_of_fold_contributions(result: dict, df: pd.DataFrame, features) -> tuple[pd.DataFrame, np.ndarray]:
    """Contributions for each account, from the same fold model that produced its score."""
    columns = _columns(features)
    X = df[columns]
    contributions = pd.DataFrame(0.0, index=df.index, columns=columns)
    bases = np.zeros(len(df))
    for fold, model in enumerate(result["models"]):
        test = result["fold_of"] == fold
        fold_values, base = logistic_contributions(model, X[~test], X[test], features)
        contributions.loc[X.index[test], fold_values.columns] = fold_values.to_numpy()
        bases[test] = base
    return contributions, bases


def driver_records(contributions: pd.DataFrame, df: pd.DataFrame, oof: np.ndarray, top_n: int = 5) -> list[str]:
    """Top drivers per account as JSON, biggest effect first."""
    logits = np.log(oof / (1 - oof))
    matrix = contributions.to_numpy()
    names = list(contributions.columns)
    records = []
    for row in range(len(df)):
        items = []
        for position in np.argsort(-np.abs(matrix[row]))[:top_n]:
            feature = names[position]
            effect = float(matrix[row, position])
            # PD points: how far this feature alone moves the account's score.
            points = (_sigmoid(logits[row]) - _sigmoid(logits[row] - effect)) * 100
            items.append(
                {
                    "feature": feature,
                    "label": FEATURE_INFO[feature][0],
                    "value": format_feature_value(feature, df[feature].iloc[row]),
                    "log_odds": round(effect, 4),
                    "pd_points": round(float(points), 2),
                }
            )
        records.append(json.dumps(items))
    return records


def shap_importance(model: Pipeline, df: pd.DataFrame, features, sample_size: int = 4000) -> pd.Series:
    """Mean absolute SHAP value per original feature, for the LightGBM model."""
    import shap

    X = df[_columns(features)].sample(min(sample_size, len(df)), random_state=SEED)
    prep, classifier = model.named_steps["prep"], model.named_steps["clf"]
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*TreeExplainer shap values output has changed.*")
        values = shap.TreeExplainer(classifier).shap_values(prep.transform(X))
    if isinstance(values, list):  # one array per class: take the default class
        values = values[1]
    grouped = _group_by_parent(values, prep.get_feature_names_out(), features, X.index)
    return grouped.abs().mean()


def decile_table(y: np.ndarray, scores: np.ndarray) -> list[dict]:
    """Accounts sorted by score: decile 1 is the riskiest 10%."""
    frame = pd.DataFrame({"y": y, "score": scores}).sort_values("score", ascending=False, kind="stable").reset_index(drop=True)
    frame["decile"] = (np.arange(len(frame)) * 10 // len(frame)) + 1
    base_rate = frame["y"].mean()
    table = frame.groupby("decile").agg(accounts=("y", "size"), defaults=("y", "sum"), mean_pd=("score", "mean"))
    table["default_rate"] = table["defaults"] / table["accounts"]
    table["capture"] = table["defaults"] / frame["y"].sum()
    table["cumulative_capture"] = table["capture"].cumsum()
    table["lift"] = table["default_rate"] / base_rate
    return table.reset_index().to_dict(orient="records")


def population_stability_index(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """PSI: how far a distribution has moved from its baseline, using baseline quantile bins."""
    expected = expected[~np.isnan(expected)]
    actual = actual[~np.isnan(actual)]
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:  # mostly constant feature, such as a count that is usually zero
        edges = np.array([expected.min() - 1, expected.mean(), expected.max() + 1])
    edges[0], edges[-1] = -np.inf, np.inf
    expected_share = np.clip(np.histogram(expected, edges)[0] / len(expected), 1e-4, None)
    actual_share = np.clip(np.histogram(actual, edges)[0] / len(actual), 1e-4, None)
    return float(((actual_share - expected_share) * np.log(actual_share / expected_share)).sum())


def stability_report(df: pd.DataFrame, top_features: list[str]) -> dict:
    baseline = (df["origination_year"] <= 2021).to_numpy()
    recent = (df["origination_year"] >= 2022).to_numpy()
    scores = df["pd_score"].to_numpy()
    by_vintage = []
    for year in sorted(df["origination_year"].unique()):
        in_year = (df["origination_year"] == year).to_numpy()
        by_vintage.append(
            {
                "origination_year": int(year),
                "accounts": int(in_year.sum()),
                "psi_vs_baseline": population_stability_index(scores[baseline], scores[in_year]),
                **score_metrics(df.loc[in_year, "target_default"], scores[in_year]),
                "observed_default_rate": float(df.loc[in_year, "target_default"].mean()),
                "predicted_default_rate": float(scores[in_year].mean()),
                "mean_visible_bureau_lines": float(df.loc[in_year, "bu_lines"].mean()),
            }
        )
    by_feature = [
        {
            "feature": feature,
            "label": FEATURE_INFO[feature][0],
            "psi": population_stability_index(
                df.loc[baseline, feature].to_numpy(dtype=float), df.loc[recent, feature].to_numpy(dtype=float)
            ),
        }
        for feature in top_features
        if pd.api.types.is_numeric_dtype(df[feature])
    ]
    return {
        "baseline": "2018-2021 originations",
        "by_vintage": by_vintage,
        "by_feature": by_feature,
        "watch_threshold": PSI_WATCH,
        "act_threshold": PSI_ACT,
    }


def fairness_table(df: pd.DataFrame) -> list[dict]:
    """Observed and predicted default by excluded attribute, to check for systematic error."""
    rows = []
    for attribute in ["gender", "marital_status", "region", "age_band"]:
        grouped = df.groupby(attribute).agg(
            accounts=("account_id", "size"),
            observed_default_rate=("target_default", "mean"),
            mean_predicted_pd=("pd_score", "mean"),
        )
        for value, row in grouped.iterrows():
            rows.append({"attribute": attribute, "value": value, **row.to_dict()})
    return rows


def assign_risk_band(pd_scores: np.ndarray) -> np.ndarray:
    return pd.cut(pd_scores, RISK_BAND_BINS, labels=RISK_BAND_LABELS, right=False).astype(str)


def _roc_points(y, scores, n_points: int = 120) -> dict:
    fpr, tpr, _ = roc_curve(y, scores)
    picks = np.unique(np.linspace(0, len(fpr) - 1, n_points).astype(int))
    return {"fpr": fpr[picks].round(4).tolist(), "tpr": tpr[picks].round(4).tolist()}


def _ks_curve(y, scores, n_points: int = 100) -> dict:
    thresholds = np.quantile(scores, np.linspace(0, 1, n_points))
    defaults, others = scores[y == 1], scores[y == 0]
    return {
        "score": thresholds.round(4).tolist(),
        "defaults_cdf": [float((defaults <= t).mean()) for t in thresholds],
        "non_defaults_cdf": [float((others <= t).mean()) for t in thresholds],
    }


def train_and_evaluate(df: pd.DataFrame, clean: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, dict]:
    """Train both models, choose the champion, and return account scores and the full report."""
    import joblib

    df = df.reset_index(drop=True)
    features = primary_features()
    y = df["target_default"].to_numpy()

    logistic = cross_validate(lambda: make_logistic(features), features, df)
    logistic["metrics"]["auc_95_interval"] = bootstrap_auc_interval(df, logistic["oof"])

    candidates = [
        (params, cross_validate(lambda p=params: make_lightgbm(features, p), features, df))
        for params in LIGHTGBM_CANDIDATES
    ]
    best_params, lightgbm = max(candidates, key=lambda item: item[1]["metrics"]["auc"])
    lightgbm["metrics"]["auc_95_interval"] = bootstrap_auc_interval(df, lightgbm["oof"])

    gain = lightgbm["metrics"]["auc"] - logistic["metrics"]["auc"]
    if gain >= CHAMPION_MIN_AUC_GAIN:
        # Key drivers rely on logistic contributions adding up exactly. If LightGBM ever wins,
        # the drivers must move to per-account SHAP, so stop rather than mislabel them.
        raise RuntimeError("LightGBM won: add per-account SHAP drivers before using it as champion")
    champion = logistic
    champion_name = "Logistic regression"

    contributions, bases = out_of_fold_contributions(logistic, df, features)
    rebuilt = _sigmoid(bases + contributions.sum(axis=1).to_numpy())
    if not np.allclose(rebuilt, logistic["oof"], atol=1e-6):
        raise ValueError("Driver contributions do not add up to the model score")

    scores = pd.DataFrame(
        {
            "account_id": df["account_id"].to_numpy(),
            "pd_score": champion["oof"],
            "lightgbm_pd_score": lightgbm["oof"],
            "risk_band": assign_risk_band(champion["oof"]),
            "drivers_json": driver_records(contributions, df, champion["oof"]),
        }
    )
    scores["pd_percentile"] = scores["pd_score"].rank(pct=True) * 100
    scores["pd_percentile_in_loan_type"] = scores.groupby(df["loan_type"].to_numpy())["pd_score"].rank(pct=True) * 100

    # Fit the final models on every account, ready for scoring new data.
    columns = _columns(features)
    final_logistic = make_logistic(features).fit(df[columns], y)
    final_lightgbm = make_lightgbm(features, best_params).fit(df[columns], y)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_logistic, MODELS_DIR / "logistic_regression.joblib")
    joblib.dump(final_lightgbm, MODELS_DIR / "lightgbm.joblib")

    shap_values = shap_importance(final_lightgbm, df, features)
    importance = (
        pd.DataFrame({"feature": contributions.columns, "logistic_mean_abs_log_odds": contributions.abs().mean().to_numpy()})
        .assign(
            label=lambda t: t["feature"].map(lambda f: FEATURE_INFO[f][0]),
            lightgbm_mean_abs_shap=lambda t: t["feature"].map(shap_values).fillna(0.0),
        )
        .sort_values("logistic_mean_abs_log_odds", ascending=False)
    )

    enriched = df.assign(pd_score=champion["oof"], risk_band=scores["risk_band"].to_numpy())
    bands = (
        enriched.groupby("risk_band")
        .agg(
            accounts=("account_id", "size"),
            defaults=("target_default", "sum"),
            mean_pd=("pd_score", "mean"),
            originated=("loan_amount", "sum"),
            latest_balance=("latest_balance", "sum"),
        )
        .reindex(RISK_BAND_LABELS)
        .fillna(0)
        .reset_index()
    )
    bands["default_rate"] = bands["defaults"] / bands["accounts"].where(bands["accounts"] > 0)

    report = {
        "champion": champion_name,
        "lightgbm_gain_over_logistic": gain,
        "champion_rule": f"LightGBM replaces logistic regression only if it wins by at least {CHAMPION_MIN_AUC_GAIN} AUC.",
        "validation": "5-fold cross-validation grouped by customer; all figures are out-of-fold.",
        "models": {
            "Logistic regression": {**logistic["metrics"], "out_of_time": out_of_time_test(lambda: make_logistic(features), features, df)},
            "LightGBM": {
                **lightgbm["metrics"],
                "out_of_time": out_of_time_test(lambda: make_lightgbm(features, best_params), features, df),
                "params": best_params,
            },
        },
        "base_default_rate": float(y.mean()),
        "accounts": int(len(df)),
        "feature_count": len(columns),
        "feature_groups": {
            "Customer and loan": APPLICATION_NUMERIC + APPLICATION_SKEWED + CATEGORICAL,
            "Repayment history": REPAYMENT_NUMERIC,
            "Bureau": BUREAU_NUMERIC + BUREAU_SKEWED,
        },
        "sensitivity": sensitivity_table(df, clean),
        "deciles": decile_table(y, champion["oof"]),
        "risk_bands": bands.to_dict(orient="records"),
        "importance": importance.to_dict(orient="records"),
        "stability": stability_report(enriched, importance["feature"].head(10).tolist()),
        "fairness": fairness_table(enriched),
        "age_proxy": {
            "years_employed_vs_age": float(df["years_employed"].corr(df["age"])),
            "score_vs_age": float(enriched["pd_score"].corr(enriched["age"])),
        },
        "curves": {
            "roc": {
                "Logistic regression": _roc_points(y, logistic["oof"]),
                "LightGBM": _roc_points(y, lightgbm["oof"]),
            },
            "ks": _ks_curve(y, champion["oof"]),
        },
    }
    return scores, report
