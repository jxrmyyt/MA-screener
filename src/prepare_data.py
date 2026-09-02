"""
Build a single labeled dataset from the two raw sources:
  - data/ma_targets.csv   : companies that WERE acquired (label = 1)
  - data/ma_controls.csv  : companies that were NOT acquired, still independent (label = 0)

Key methodological note (see README for full discussion):
Target financials are the last full fiscal year reported BEFORE the deal was
announced (so the model can't "see the future"). Control financials are each
company's most recently completed fiscal year as of the research date. This is
a simplification -- ideally every control would be sampled at a fiscal year
matched to the targets' 2015-2022 window -- but point-in-time historical
financials for a matched control panel aren't freely available without a paid
database (WRDS/Capital IQ). This is called out explicitly as a limitation.
"""
import pandas as pd
import numpy as np

NUMERIC_COLS = [
    "annual_revenue_usd_millions",
    "revenue_growth_yoy_pct",
    "gross_margin_pct",
    "operating_margin_pct",
    "rd_expense_pct_of_revenue",
    "total_debt_usd_millions",
    "cash_and_equivalents_usd_millions",
]


def _clean_numeric(series):
    return pd.to_numeric(series, errors="coerce")


def load():
    targets = pd.read_csv("data/ma_targets.csv")
    controls = pd.read_csv("data/ma_controls.csv")

    targets = targets.rename(columns={"ticker_at_time": "ticker"})
    targets["label"] = 1
    controls["label"] = 0

    keep_cols = ["company", "ticker", "label"] + NUMERIC_COLS
    targets = targets[keep_cols].copy()
    controls = controls[keep_cols].copy()

    df = pd.concat([targets, controls], ignore_index=True)

    for col in NUMERIC_COLS:
        df[col] = _clean_numeric(df[col])

    return df


def engineer_features(df):
    df = df.copy()
    # Debt burden relative to size
    df["debt_to_revenue"] = df["total_debt_usd_millions"] / df["annual_revenue_usd_millions"]
    # Cash cushion relative to size
    df["cash_to_revenue"] = df["cash_and_equivalents_usd_millions"] / df["annual_revenue_usd_millions"]
    # Net leverage (debt minus cash, relative to revenue) -- a classic "ripe for a deal" signal
    df["net_debt_to_revenue"] = (
        df["total_debt_usd_millions"] - df["cash_and_equivalents_usd_millions"]
    ) / df["annual_revenue_usd_millions"]
    # log revenue as a size control (deal activity is very size-dependent)
    df["log_revenue"] = np.log(df["annual_revenue_usd_millions"].clip(lower=1))
    return df


FEATURE_COLS = [
    "log_revenue",
    "revenue_growth_yoy_pct",
    "gross_margin_pct",
    "operating_margin_pct",
    "rd_expense_pct_of_revenue",
    "debt_to_revenue",
    "cash_to_revenue",
    "net_debt_to_revenue",
]


def build_dataset():
    df = load()
    df = engineer_features(df)

    # Median-impute missing values per feature (documented, not hidden -- see README)
    missing_report = df[FEATURE_COLS].isna().sum()
    for col in FEATURE_COLS:
        median = df[col].median()
        df[col] = df[col].fillna(median)

    return df, missing_report


if __name__ == "__main__":
    df, missing = build_dataset()
    df.to_csv("data/dataset.csv", index=False)
    print(f"Built dataset: {len(df)} companies ({df['label'].sum()} targets, {(df['label']==0).sum()} controls)")
    print("\nMissing values before imputation (filled with column median):")
    print(missing.to_string())
    print(f"\nSaved to data/dataset.csv")
