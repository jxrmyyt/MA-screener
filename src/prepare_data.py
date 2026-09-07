"""
Build a single labeled dataset from the two raw sources:
  - data/ma_targets.csv   : companies that WERE acquired (label = 1)
  - data/ma_controls.csv  : companies that were NOT acquired, still independent (label = 0)

Now cross-industry (Semiconductor, Software, Healthcare, Industrials, Consumer/Retail,
Telecom) rather than semiconductor-only. That changes the modeling problem: a 25% R&D-
to-revenue ratio is unremarkable for a software company and extreme for an industrial
one, so comparing raw financial ratios across sectors would mostly teach a model "this
is a software company" rather than anything about acquisition likelihood. Every ratio
feature is therefore converted to a SECTOR-RELATIVE z-score (how many sector-standard-
deviations this company's ratio is from its own sector's median) before modeling.
Company size (log revenue) is left as a global (non-sector-relative) feature, since
absolute deal capacity plausibly matters regardless of industry.

Key methodological note (see README for full discussion):
Target financials are the last full fiscal year reported BEFORE the deal was
announced (so the model can't see the future). Control financials are each
company's most recently completed fiscal year as of the research date. This is
a simplification -- ideally every control would be sampled at a fiscal year
matched to the targets' distribution -- but point-in-time historical financials
for a matched control panel aren't freely available without a paid database
(WRDS/Capital IQ). Called out explicitly as a limitation.
"""
import pandas as pd
import numpy as np

RAW_NUMERIC_COLS = [
    "annual_revenue_usd_millions",
    "revenue_growth_yoy_pct",
    "gross_margin_pct",
    "operating_margin_pct",
    "rd_expense_pct_of_revenue",
    "total_debt_usd_millions",
    "cash_and_equivalents_usd_millions",
]

# Ratio columns get winsorized (clipped to these bounds) before sector z-scoring, so a
# handful of extreme pre-commercial biotech values (e.g. R&D at 600%+ of revenue, or
# operating margin below -1000%, both real and expected for clinical-stage names with
# near-zero revenue) don't single-handedly blow up a sector's mean/std and swamp every
# other company's z-score in that sector.
WINSOR_BOUNDS = {
    "revenue_growth_yoy_pct": (-80, 150),
    "gross_margin_pct": (-50, 100),
    "operating_margin_pct": (-150, 60),
    "rd_expense_pct_of_revenue": (0, 120),
    "debt_to_revenue": (0, 3),
    "cash_to_revenue": (0, 3),
    "net_debt_to_revenue": (-3, 3),
}

MIN_SECTOR_SIZE = 4  # below this, fall back to whole-dataset stats for that sector's z-scores


def _clean_numeric(series):
    return pd.to_numeric(series, errors="coerce")


def load():
    targets = pd.read_csv("data/ma_targets.csv")
    controls = pd.read_csv("data/ma_controls.csv")

    targets = targets.rename(columns={"ticker_at_time": "ticker"})
    targets["label"] = 1
    controls["label"] = 0

    keep_cols = ["company", "ticker", "sector", "label"] + RAW_NUMERIC_COLS
    targets = targets[keep_cols].copy()
    controls = controls[keep_cols].copy()

    df = pd.concat([targets, controls], ignore_index=True)

    for col in RAW_NUMERIC_COLS:
        df[col] = _clean_numeric(df[col])

    return df


def engineer_features(df):
    df = df.copy()
    # Floor revenue when used as a ratio denominator so near-zero-revenue, pre-commercial
    # companies (a handful of clinical-stage biotech targets) don't produce absurd ratios
    # from dividing by a number close to zero.
    safe_revenue = df["annual_revenue_usd_millions"].clip(lower=5)

    df["debt_to_revenue"] = df["total_debt_usd_millions"] / safe_revenue
    df["cash_to_revenue"] = df["cash_and_equivalents_usd_millions"] / safe_revenue
    df["net_debt_to_revenue"] = (
        df["total_debt_usd_millions"] - df["cash_and_equivalents_usd_millions"]
    ) / safe_revenue
    df["log_revenue"] = np.log(df["annual_revenue_usd_millions"].clip(lower=1))
    return df


RATIO_FEATURES = [
    "revenue_growth_yoy_pct",
    "gross_margin_pct",
    "operating_margin_pct",
    "rd_expense_pct_of_revenue",
    "debt_to_revenue",
    "cash_to_revenue",
    "net_debt_to_revenue",
]

FEATURE_COLS = ["log_revenue"] + [f"sector_z_{c}" for c in RATIO_FEATURES]


def compute_sector_baselines(df):
    """Return {feature: {sector: [median, std], "__global__": [median, std]}}, using
    winsorized values. A sector with fewer than MIN_SECTOR_SIZE companies has no entry
    of its own -- callers should fall back to "__global__" for it, same rule used when
    building the training features below, so a company scored later in an unseen or
    tiny sector is treated consistently with how training data was."""
    baselines = {}
    sector_counts = df["sector"].value_counts()

    for col in RATIO_FEATURES:
        lo, hi = WINSOR_BOUNDS[col]
        winsorized = df[col].clip(lower=lo, upper=hi)

        global_median = float(winsorized.median())
        global_std = float(winsorized.std()) if winsorized.std() and winsorized.std() > 1e-6 else 1.0

        feature_baselines = {"__global__": [global_median, global_std]}
        for sector, group_idx in df.groupby("sector").groups.items():
            group = winsorized.loc[group_idx]
            if sector_counts[sector] >= MIN_SECTOR_SIZE and group.std(skipna=True) > 1e-6:
                feature_baselines[sector] = [float(group.median()), float(group.std())]

        baselines[col] = feature_baselines

    return baselines


def zscore_row(raw_row, sector, baselines):
    """Given a dict of engineered ratio values for ONE company (already run through
    engineer_features) and its sector, return {sector_z_<feature>: value} using
    pre-computed baselines -- so scoring a new company never needs the full training
    dataframe, only the baselines learned from it. Falls back to the "__global__"
    baseline for a sector the training data didn't have enough of (or a sector never
    seen at all)."""
    out = {}
    for col in RATIO_FEATURES:
        lo, hi = WINSOR_BOUNDS[col]
        value = raw_row[col]
        if value is None or (isinstance(value, float) and np.isnan(value)):
            out[f"sector_z_{col}"] = 0.0
            continue
        value = min(max(value, lo), hi)
        median, std = baselines[col].get(sector, baselines[col]["__global__"])
        out[f"sector_z_{col}"] = (value - median) / std
    return out


def add_sector_relative_features(df, baselines=None):
    """Add sector_z_<feature> columns to df. If baselines is None, computes them from df
    itself (the training-time path); otherwise uses the given pre-computed baselines
    (the scoring-time path, e.g. for a single new company)."""
    df = df.copy()
    if baselines is None:
        baselines = compute_sector_baselines(df)

    for col in RATIO_FEATURES:
        z_col = f"sector_z_{col}"
        df[z_col] = [
            zscore_row({c: df.at[i, c] for c in RATIO_FEATURES}, df.at[i, "sector"], baselines)[z_col]
            for i in df.index
        ]

    return df, baselines


def build_dataset():
    df = load()
    df = engineer_features(df)
    df, baselines = add_sector_relative_features(df)

    missing_report = df[RATIO_FEATURES + ["annual_revenue_usd_millions"]].isna().sum()

    # Median-impute missing sector-z values with 0 (= "sector-typical") after normalization,
    # since 0 is the natural "no information" value on a z-score scale.
    for col in FEATURE_COLS:
        if col == "log_revenue":
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna(0.0)

    return df, missing_report, baselines


if __name__ == "__main__":
    df, missing, baselines = build_dataset()
    df.to_csv("data/dataset.csv", index=False)
    print(f"Built dataset: {len(df)} companies ({df['label'].sum()} targets, {(df['label']==0).sum()} controls)")
    print("\nBy sector:")
    print(df.groupby(["sector", "label"]).size().unstack(fill_value=0).to_string())
    print("\nMissing raw values before imputation:")
    print(missing.to_string())
    print(f"\nSaved to data/dataset.csv")
