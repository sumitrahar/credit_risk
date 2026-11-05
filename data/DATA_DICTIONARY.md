# Credit Risk Portfolio — Synthetic Dataset

5 interlinked tables covering **10,000 customers** and their loan accounts,
external credit history, repayment behavior, and collections activity.
All data is synthetically generated but designed to mimic realistic
distributions, correlations, and messiness found in real banking data.

## Table Overview

| File | Grain | Rows (approx.) | Links to |
|---|---|---|---|
| `customers.csv` | 1 row per customer | 10,000 | primary key: `customer_id` |
| `loan_accounts.csv` | 1 row per loan account | ~14,700 | `customer_id` → customers |
| `bureau_credit.csv` | 1 row per external credit line | ~23,000 | `customer_id` → customers |
| `repayment_history.csv` | 1 row per account per month | ~176,000 | `account_id` → loan_accounts |
| `collections_events.csv` | 1 row per collections action | ~8,600 | `account_id` → loan_accounts |

## `customers.csv` — demographic / profile dimension
| Column | Description |
|---|---|
| customer_id | Unique customer identifier |
| age | Age in years |
| gender | Male/Female (note: ~6% coded as "M"/"F" — inconsistent labeling, intentional) |
| marital_status | Married / Single / Divorced / Widowed |
| education_level | Highest education attained |
| employment_type | Salaried / Self-Employed / Business Owner / Contract / Unemployed |
| occupation_sector | Industry sector |
| years_employed | Years in current employment (has missing values) |
| annual_income | Annual income in USD |
| region | North / South / East / West / Central |
| city_tier | Tier 1 / 2 / 3 city classification |
| num_dependents | Number of dependents (has missing values) |
| housing_type | Owned / Mortgaged / Rented / Living with Family |
| years_at_residence | Years at current residence (has missing values) |
| customer_since | Date customer relationship began |

## `loan_accounts.csv` — main fact table (has the modeling target)
| Column | Description |
|---|---|
| account_id | Unique loan account identifier |
| customer_id | Links to customers.csv |
| loan_type | Personal Loan / Auto Loan / Credit Card / Mortgage / Business Loan |
| loan_purpose | Stated purpose of the loan |
| loan_amount | Original loan/credit amount |
| interest_rate_pct | Annual interest rate (has missing values) |
| tenure_months | Loan tenure in months |
| origination_date | Date loan was originated |
| disbursal_channel | Branch / Online / Partner-DSA / Mobile App |
| collateral_flag | 1 if secured (Mortgage/Auto), 0 otherwise |
| **target_default** | **1 = defaulted, 0 = performing — this is the prediction target** |

## `bureau_credit.csv` — external credit history (many-to-one with customers)
| Column | Description |
|---|---|
| bureau_id | Unique bureau record identifier |
| customer_id | Links to customers.csv |
| credit_type | Type of external credit line |
| lender_category | Bank / NBFC / Fintech / Cooperative |
| sanctioned_amount | Approved credit limit/amount at that institution |
| outstanding_amount | Current outstanding balance |
| account_status | Active / Closed / Written-Off |
| days_past_due | Days past due at last reporting (has missing values) |
| date_opened | Date the external credit line was opened |
| date_closed | Date closed (null if still active) |

## `repayment_history.csv` — monthly repayment behavior (many-to-one with loan_accounts)
| Column | Description |
|---|---|
| account_id | Links to loan_accounts.csv |
| payment_month | Month index since origination (1, 2, 3...) |
| emi_due | Installment amount due |
| emi_paid | Amount actually paid (0 if missed) |
| days_past_due | Days past due for that month |
| payment_status | On-Time / Late / Missed |
| outstanding_balance | Remaining balance after that month |

## `collections_events.csv` — collections/recovery actions (sparse — only delinquent accounts)
| Column | Description |
|---|---|
| collection_event_id | Unique event identifier |
| account_id | Links to loan_accounts.csv |
| event_date | Date of the collections action |
| action_type | Reminder Call / SMS-Email / Field Visit / Settlement Offer / Legal Notice / Written-Off Referral |
| outcome | Promise to Pay / Payment Received / No Response / Disputed / Escalated |
| amount_recovered | Amount recovered through this action (often 0) |

## Notes for whoever is using this dataset
- **Known data quality issues are intentional**: missing values in several columns,
  inconsistent gender labels, missing interest rates/DPD values. A strong candidate
  should surface and handle these explicitly, not silently drop rows.
- Relationships are genuinely one-to-many (a customer can have multiple loan
  accounts; an account has many months of repayment history) — this requires
  real aggregation/feature-engineering before modeling, not just a single join.
- `customers.csv` supports **profiling/segmentation** independent of the
  credit risk model (e.g., income x region x employment cross-tabs), so it's
  usable for both the analytics and modeling stages of the exercise.
- Overall portfolio default rate is ~16%, with clear (but not trivial)
  signal from income, employment type, and prior bureau write-offs.
