"""
M&A Target Screener -- Streamlit app.

Three views:
  1. Screener: ranks today's independent companies (across sectors) by predicted
     acquisition-target probability (out-of-fold LOOCV scores, so no company
     ever saw its own label during scoring). Switchable between all 4 models,
     filterable by sector.
  2. Custom scorer: enter any company's financial profile (sector included --
     ratios are compared against that sector's own baseline, not the whole
     cross-industry dataset) or load a real one from the dataset, and get a
     live prediction compared against the typical acquired-target and typical
     independent-company profile in the SAME sector.
  3. Model details: LOOCV metrics, feature importance, known limitations.

Run with: streamlit run app/app.py   (from the project root)
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import json
import joblib
import pandas as pd
import streamlit as st

from prepare_data import build_dataset, engineer_features, zscore_row, RATIO_FEATURES

st.set_page_config(page_title="M&A Target Screener", page_icon="\U0001F50E", layout="wide")

ROOT = os.path.join(os.path.dirname(__file__), "..")

DISPLAY_NAMES = {
    "logistic_regression": "Logistic Regression",
    "decision_tree": "Decision Tree",
    "svm_rbf": "SVM (RBF)",
    "random_forest": "Random Forest",
}

RAW_FIELDS = [
    ("annual_revenue_usd_millions", "Annual revenue ($M)", 800.0),
    ("revenue_growth_yoy_pct", "Revenue growth YoY (%)", 8.0),
    ("gross_margin_pct", "Gross margin (%)", 50.0),
    ("operating_margin_pct", "Operating margin (%)", 10.0),
    ("rd_expense_pct_of_revenue", "R&D expense (% of revenue)", 20.0),
    ("total_debt_usd_millions", "Total debt ($M)", 100.0),
    ("cash_and_equivalents_usd_millions", "Cash & equivalents ($M)", 200.0),
]


@st.cache_data
def load_predictions():
    return pd.read_csv(os.path.join(ROOT, "outputs", "loocv_predictions.csv"))


@st.cache_data
def load_metrics():
    with open(os.path.join(ROOT, "outputs", "model_metrics.json")) as f:
        return json.load(f)


@st.cache_data
def load_importance():
    with open(os.path.join(ROOT, "outputs", "feature_importance.json")) as f:
        return json.load(f)


@st.cache_data
def load_baselines():
    with open(os.path.join(ROOT, "outputs", "sector_baselines.json")) as f:
        return json.load(f)


@st.cache_resource
def load_model():
    return joblib.load(os.path.join(ROOT, "outputs", "final_model.joblib"))


@st.cache_data
def load_full_dataset():
    df, _, _ = build_dataset()
    return df


# ---- Load everything up front, fail with a helpful message instead of a raw traceback ----
missing = [
    p for p in ["loocv_predictions.csv", "model_metrics.json", "feature_importance.json",
                "final_model.joblib", "sector_baselines.json"]
    if not os.path.exists(os.path.join(ROOT, "outputs", p))
]
if missing:
    st.error(
        "This app reads pre-computed results from `outputs/`, but the following files "
        f"are missing: {', '.join(missing)}.\n\n"
        "Run the pipeline first, from the project root:\n\n"
        "```bash\npython3 src/prepare_data.py\npython3 src/train_models.py\n```"
    )
    st.stop()

try:
    preds = load_predictions()
    metrics = load_metrics()
    importance = load_importance()
    baselines = load_baselines()
    dataset = load_full_dataset()
except Exception as e:
    st.error(f"Found the output files but couldn't load them cleanly: {e}\n\n"
             "Try re-running `python3 src/train_models.py` to regenerate `outputs/`.")
    st.stop()

BEST_MODEL = importance["model"]
MODEL_PROB_COLS = [c for c in preds.columns if c.endswith("_predicted_prob")]
AVAILABLE_MODELS = [c.replace("_predicted_prob", "") for c in MODEL_PROB_COLS]
N_COMPANIES = len(dataset)
N_TARGETS = int(dataset["label"].sum())
N_CONTROLS = N_COMPANIES - N_TARGETS
SECTORS = sorted(dataset["sector"].unique().tolist())

st.title("Cross-Industry M&A Target Screener")
st.caption(
    f"A classifier trained on {N_TARGETS} real, completed acquisitions and {N_CONTROLS} comparable "
    f"companies that remained independent, spanning {len(SECTORS)} sectors ({', '.join(SECTORS)}). "
    "Every financial ratio is compared against its own sector's baseline before scoring, so a software "
    "company's R&D spending isn't judged against an industrial company's norms."
)

with st.expander("⚠️ Known limitations (read before trusting any number below)"):
    st.markdown(f"""
- **Modest sample ({N_COMPANIES} companies across {len(SECTORS)} sectors).** LOOCV squeezes maximum
  signal out of a limited labeled set, but metrics still carry real uncertainty — a handful of different
  labels would shift the numbers. Some sectors (e.g. Telecom, 3 targets + 3 controls) are too small to
  have their own normalization baseline and fall back to the whole-dataset baseline instead.
- **Point-in-time mismatch.** Target financials are the last full fiscal year before each deal's
  announcement; control financials are each company's most recent fiscal year today. Ideally every
  company would be sampled at a matched point in time, but that requires a paid historical database
  (WRDS/Capital IQ) rather than free public sources.
- **Cross-industry comparability.** Ratios are converted to sector-relative z-scores specifically to
  avoid the model just learning "high R&D% = software company", but this only controls for sector
  *median and spread* — it can't fully correct for how differently "typical" itself varies across very
  different business models.
- **Missing data imputed** — sector-z values default to 0 (sector-typical) when the raw figure wasn't
  disclosed, most often debt/cash for older or private-at-acquisition targets.
- **Survivorship framing.** "Not yet acquired" is treated as the negative class, but several control
  companies could plausibly be acquired in the future — this is a limitation of any target-screening
  model, not just this one.
- **This is a pattern-match, not a forecast.** A high score means "financially resembles past
  acquisition targets in its sector," not "will be acquired."
""")

tab1, tab2, tab3 = st.tabs(["\U0001F4CA Screener", "\U0001F9EE Score a company", "\U0001F4D0 Model details"])

with tab1:
    st.subheader("Ranked by predicted acquisition-target probability")

    col_a, col_b = st.columns([2, 1])
    with col_a:
        model_choice = st.selectbox(
            "Rank by model",
            options=AVAILABLE_MODELS,
            index=AVAILABLE_MODELS.index(BEST_MODEL),
            format_func=lambda m: DISPLAY_NAMES.get(m, m) + ("  (best by ROC-AUC)" if m == BEST_MODEL else ""),
        )
    with col_b:
        sector_filter = st.selectbox("Filter by sector", options=["All sectors"] + SECTORS)

    st.caption(
        "Companies below are all currently independent (never acquired). Scores are out-of-fold "
        "leave-one-out predictions, so no company's own outcome leaked into its score."
    )

    prob_col = f"{model_choice}_predicted_prob"
    controls_only = preds[preds["actual_label"] == 0][["company", "sector", prob_col]].copy()
    if sector_filter != "All sectors":
        controls_only = controls_only[controls_only["sector"] == sector_filter]
    controls_only = controls_only.rename(columns={prob_col: "predicted target probability"})
    # ProgressColumn's format string doesn't auto-scale a 0-1 fraction into a percent --
    # it just prints the raw value -- so convert to a 0-100 scale before display, or
    # "63.7%" silently becomes "0.6%".
    controls_only["predicted target probability"] = (
        controls_only["predicted target probability"].clip(0, 1) * 100
    )
    controls_only = controls_only.sort_values("predicted target probability", ascending=False)
    controls_only.insert(0, "rank", range(1, len(controls_only) + 1))

    st.dataframe(
        controls_only,
        use_container_width=True,
        hide_index=True,
        column_config={
            "predicted target probability": st.column_config.ProgressColumn(
                "predicted target probability",
                format="%.1f%%",
                min_value=0.0,
                max_value=100.0,
            )
        },
    )

    st.download_button(
        "Download full ranked screener (CSV)",
        data=controls_only.to_csv(index=False).encode("utf-8"),
        file_name=f"ma_screener_{model_choice}.csv",
        mime="text/csv",
    )

with tab2:
    st.subheader("Score any company by its financial profile")

    company_options = ["-- manual entry --"] + sorted(dataset["company"].tolist())

    def _load_selected_company():
        selected = st.session_state.get("company_selector")
        if not selected or selected == "-- manual entry --":
            return
        row = dataset[dataset["company"] == selected].iloc[0]
        for field, _, _ in RAW_FIELDS:
            st.session_state[f"input_{field}"] = float(row[field])
        st.session_state["input_sector"] = row["sector"]

    st.selectbox(
        "Load a real company from the dataset (optional)",
        options=company_options,
        key="company_selector",
        on_change=_load_selected_company,
        help="Prefills the fields below (including sector) with that company's actual financials, "
             "so you can sanity-check the model against a company you already know the outcome for.",
    )

    st.session_state.setdefault("input_sector", SECTORS[0])
    sector_choice = st.selectbox(
        "Sector",
        options=SECTORS,
        key="input_sector",
        help="Ratios are compared against this sector's own baseline before scoring.",
    )

    for field, _, default in RAW_FIELDS:
        st.session_state.setdefault(f"input_{field}", default)

    col1, col2 = st.columns(2)
    columns = [col1, col1, col1, col1, col2, col2, col2]
    values = {}
    for (field, label, _), col in zip(RAW_FIELDS, columns):
        step = 0.5 if "pct" in field or "margin" in field else 10.0
        min_val = None if "growth" in field or "margin" in field else 0.0
        values[field] = col.number_input(label, key=f"input_{field}", step=step, min_value=min_val)

    if st.button("Score this company", type="primary"):
        bundle = load_model()
        model, scaler, features = bundle["model"], bundle["scaler"], bundle["features"]

        raw_row = pd.DataFrame([{
            "annual_revenue_usd_millions": values["annual_revenue_usd_millions"],
            "revenue_growth_yoy_pct": values["revenue_growth_yoy_pct"],
            "gross_margin_pct": values["gross_margin_pct"],
            "operating_margin_pct": values["operating_margin_pct"],
            "rd_expense_pct_of_revenue": values["rd_expense_pct_of_revenue"],
            "total_debt_usd_millions": values["total_debt_usd_millions"],
            "cash_and_equivalents_usd_millions": values["cash_and_equivalents_usd_millions"],
        }])
        raw_row = engineer_features(raw_row)

        # Convert this one company's ratios to sector-relative z-scores using the
        # baselines learned at training time -- the same transform the model was
        # trained on, applied to a company the training data never saw.
        engineered = raw_row.iloc[0].to_dict()
        z_features = zscore_row(engineered, sector_choice, baselines)
        engineered.update(z_features)

        X = pd.DataFrame([engineered])[features].values
        X_scaled = scaler.transform(X)
        prob = model.predict_proba(X_scaled)[0][1]

        st.metric("Predicted acquisition-target probability", f"{prob*100:.1f}%",
                   help=f"Using {DISPLAY_NAMES.get(BEST_MODEL, BEST_MODEL)}, the best model by LOOCV ROC-AUC.")

        selected = st.session_state.get("company_selector")
        if selected and selected != "-- manual entry --":
            screener_row = preds[preds["company"] == selected]
            best_prob_col = f"{BEST_MODEL}_predicted_prob"
            if not screener_row.empty:
                screener_prob = screener_row.iloc[0][best_prob_col] * 100
                st.caption(
                    f"Note: the Screener tab shows **{screener_prob:.1f}%** for {selected} — that's an "
                    f"out-of-fold LOOCV score (the model never saw {selected}'s own label while scoring it). "
                    f"This **{prob*100:.1f}%** comes from the final model fit on *all* {N_COMPANIES} companies, "
                    f"including {selected}. The gap between the two is itself informative: a big gap means this "
                    "company's own data point meaningfully shifted the model — a small gap means the model's "
                    "view of it is stable regardless of whether it was held out."
                )

        if prob > 0.6:
            st.warning("High likelihood profile — resembles historical acquisition targets in this sector.")
        elif prob > 0.4:
            st.info("Moderate likelihood — mixed signal.")
        else:
            st.success("Low likelihood — resembles companies that stayed independent in this sector.")

        st.markdown(
            f"**Why this score:** how this company's inputs compare to the typical acquired target and "
            f"typical independent company **within {sector_choice}** — not the whole cross-industry dataset, "
            "since comparing a software company's margins to an industrial company's would be misleading."
        )
        sector_data = dataset[dataset["sector"] == sector_choice]
        target_pool = sector_data[sector_data["label"] == 1]
        control_pool = sector_data[sector_data["label"] == 0]
        raw_cols = [f for f, _, _ in RAW_FIELDS]
        target_median = target_pool[raw_cols].median() if len(target_pool) else dataset[dataset["label"] == 1][raw_cols].median()
        control_median = control_pool[raw_cols].median() if len(control_pool) else dataset[dataset["label"] == 0][raw_cols].median()
        compare_df = pd.DataFrame({
            "This company": [values[f] for f in raw_cols],
            f"Typical {sector_choice} target": [round(target_median[f], 1) for f in raw_cols],
            f"Typical independent {sector_choice} company": [round(control_median[f], 1) for f in raw_cols],
        }, index=[label for _, label, _ in RAW_FIELDS])
        st.dataframe(compare_df, use_container_width=True)
        if len(target_pool) < 4 or len(control_pool) < 4:
            st.caption(
                f"({sector_choice} has only {len(target_pool)} target(s) and {len(control_pool)} control(s) "
                "in the dataset — this comparison is thin, treat it as indicative rather than robust.)"
            )

with tab3:
    st.subheader(f"Model comparison (Leave-One-Out Cross-Validation, N={N_COMPANIES})")
    metrics_df = pd.DataFrame(metrics).T
    metrics_df.index = [DISPLAY_NAMES.get(i, i) for i in metrics_df.index]
    st.dataframe(metrics_df, use_container_width=True)
    st.caption(
        f"**{DISPLAY_NAMES.get(BEST_MODEL, BEST_MODEL)}** is selected as the production model because it has "
        "the best ROC-AUC — the right metric for a *ranked* screener, since it rewards correctly ordering "
        "all companies rather than just getting the 0.5-cutoff call right on each one."
    )

    st.subheader("Feature importance (best model)")
    st.caption(f"Model: {DISPLAY_NAMES.get(BEST_MODEL, BEST_MODEL)}")
    imp_df = pd.DataFrame(
        list(importance["importance"].items()), columns=["feature", "importance"]
    ).sort_values("importance", key=abs, ascending=False)
    st.bar_chart(imp_df.set_index("feature"))

    st.subheader("Companies by sector")
    st.dataframe(
        dataset.groupby(["sector", "label"]).size().unstack(fill_value=0).rename(
            columns={0: "independent (control)", 1: "acquired (target)"}
        ),
        use_container_width=True,
    )
