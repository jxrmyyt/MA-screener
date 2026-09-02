"""
M&A Target Screener -- Streamlit app.

Three views:
  1. Screener: ranks today's independent semiconductor companies by predicted
     acquisition-target probability (out-of-fold LOOCV scores, so no company
     ever saw its own label during scoring). Switchable between all 4 models.
  2. Custom scorer: enter any company's financial profile (or load a real one
     from the dataset) and get a live prediction, compared against the typical
     acquired-target and typical independent-company profile.
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

from prepare_data import build_dataset, engineer_features

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


@st.cache_resource
def load_model():
    return joblib.load(os.path.join(ROOT, "outputs", "final_model.joblib"))


@st.cache_data
def load_full_dataset():
    df, _ = build_dataset()
    return df


# ---- Load everything up front, fail with a helpful message instead of a raw traceback ----
missing = [
    p for p in ["loocv_predictions.csv", "model_metrics.json", "feature_importance.json", "final_model.joblib"]
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
    dataset = load_full_dataset()
except Exception as e:
    st.error(f"Found the output files but couldn't load them cleanly: {e}\n\n"
             "Try re-running `python3 src/train_models.py` to regenerate `outputs/`.")
    st.stop()

BEST_MODEL = importance["model"]
MODEL_PROB_COLS = [c for c in preds.columns if c.endswith("_predicted_prob")]
AVAILABLE_MODELS = [c.replace("_predicted_prob", "") for c in MODEL_PROB_COLS]

st.title("Semiconductor M&A Target Screener")
st.caption(
    "A classifier trained on 16 real, completed semiconductor-sector acquisitions (2015-2022) "
    "and 20 comparable companies that remained independent, predicting acquisition-target likelihood "
    "from a company's financial profile."
)

with st.expander("⚠️ Known limitations (read before trusting any number below)"):
    st.markdown("""
- **Small sample (N=36).** LOOCV squeezes maximum signal out of a small labeled set, but
  metrics still carry wide uncertainty — a handful of different labels would shift the numbers.
- **Point-in-time mismatch.** Target financials are the last full fiscal year before each deal's
  announcement; control financials are each company's most recent fiscal year today. Ideally every
  company would be sampled at a matched point in time, but that requires a paid historical database
  (WRDS/Capital IQ) rather than free public sources.
- **Missing data imputed with column medians** for ~10-30% of cells per feature (mostly debt/cash,
  which weren't always disclosed in the press releases used as sources for older, delisted targets).
- **Survivorship framing.** "Not yet acquired" is treated as the negative class, but several control
  companies could plausibly be acquired in the future — this is a limitation of any target-screening
  model, not just this one.
- **This is a pattern-match, not a forecast.** A high score means "financially resembles past
  acquisition targets," not "will be acquired."
""")

tab1, tab2, tab3 = st.tabs(["\U0001F4CA Screener", "\U0001F9EE Score a company", "\U0001F4D0 Model details"])

with tab1:
    st.subheader("Ranked by predicted acquisition-target probability")

    model_choice = st.selectbox(
        "Rank by model",
        options=AVAILABLE_MODELS,
        index=AVAILABLE_MODELS.index(BEST_MODEL),
        format_func=lambda m: DISPLAY_NAMES.get(m, m) + ("  (best by ROC-AUC)" if m == BEST_MODEL else ""),
    )
    st.caption(
        "Companies below are all currently independent (never acquired). Scores are out-of-fold "
        "leave-one-out predictions, so no company's own outcome leaked into its score."
    )

        prob_col = f"{model_choice}_predicted_prob"
    controls_only = preds[preds["actual_label"] == 0][["company", prob_col]].copy()
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

    st.selectbox(
        "Load a real company from the dataset (optional)",
        options=company_options,
        key="company_selector",
        on_change=_load_selected_company,
        help="Prefills the fields below with that company's actual financials, so you can "
             "sanity-check the model against a company you already know the outcome for.",
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

        row = pd.DataFrame([{
            "annual_revenue_usd_millions": values["annual_revenue_usd_millions"],
            "revenue_growth_yoy_pct": values["revenue_growth_yoy_pct"],
            "gross_margin_pct": values["gross_margin_pct"],
            "operating_margin_pct": values["operating_margin_pct"],
            "rd_expense_pct_of_revenue": values["rd_expense_pct_of_revenue"],
            "total_debt_usd_millions": values["total_debt_usd_millions"],
            "cash_and_equivalents_usd_millions": values["cash_and_equivalents_usd_millions"],
        }])
        row = engineer_features(row)
        X = row[features].values
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
                    f"This **{prob*100:.1f}%** comes from the final model fit on *all* 36 companies, including "
                    f"{selected}. The gap between the two is itself informative: a big gap means this "
                    "company's own data point meaningfully shifted the model — a small gap means the model's "
                    "view of it is stable regardless of whether it was held out."
                )

        if prob > 0.6:
            st.warning("High likelihood profile — resembles historical acquisition targets.")
        elif prob > 0.4:
            st.info("Moderate likelihood — mixed signal.")
        else:
            st.success("Low likelihood — resembles companies that stayed independent.")

        st.markdown("**Why this score:** how this company's inputs compare to the typical acquired target and the typical independent company in the dataset.")
        target_median = dataset[dataset["label"] == 1][[f for f, _, _ in RAW_FIELDS]].median()
        control_median = dataset[dataset["label"] == 0][[f for f, _, _ in RAW_FIELDS]].median()
        compare_df = pd.DataFrame({
            "This company": [values[f] for f, _, _ in RAW_FIELDS],
            "Typical acquired target": [round(target_median[f], 1) for f, _, _ in RAW_FIELDS],
            "Typical independent company": [round(control_median[f], 1) for f, _, _ in RAW_FIELDS],
        }, index=[label for _, label, _ in RAW_FIELDS])
        st.dataframe(compare_df, use_container_width=True)

with tab3:
    st.subheader("Model comparison (Leave-One-Out Cross-Validation, N=36)")
    metrics_df = pd.DataFrame(metrics).T
    metrics_df.index = [DISPLAY_NAMES.get(i, i) for i in metrics_df.index]
    st.dataframe(metrics_df, use_container_width=True)
    st.caption(
        f"**{DISPLAY_NAMES.get(BEST_MODEL, BEST_MODEL)}** is selected as the production model because it has "
        "the best ROC-AUC — the right metric for a *ranked* screener, since it rewards correctly ordering "
        "all 36 companies rather than just getting the 0.5-cutoff call right on each one."
    )

    st.subheader("Feature importance (best model)")
    st.caption(f"Model: {DISPLAY_NAMES.get(BEST_MODEL, BEST_MODEL)}")
    imp_df = pd.DataFrame(
        list(importance["importance"].items()), columns=["feature", "importance"]
    ).sort_values("importance", key=abs, ascending=False)
    st.bar_chart(imp_df.set_index("feature"))