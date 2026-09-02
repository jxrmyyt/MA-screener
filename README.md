# Semiconductor M&A Target Screener

A machine learning project that predicts which semiconductor / analog / RF companies
look like acquisition targets, based on their financial profile — and, as a stretch
extension, starts on predicting whether an announced deal completes or falls through.

I've been increasingly drawn to M&A and corporate strategy — the mix of financial
analysis and pattern-recognition in figuring out why a company becomes a target feels
like a natural place to point machine learning. This project is me testing that: can a
model trained on real, completed semiconductor deals actually pick up a signal in the
kind of financial profile that tends to attract an acquirer, using nothing but public
data and standard classifiers?

## The question

Given a company's financials — revenue, growth, margins, leverage, R&D intensity —
can we predict whether it looks like the kind of company that gets acquired?

## Data

Two hand-curated, sourced datasets (`data/ma_targets.csv`, `data/ma_controls.csv`),
36 companies total:

- **16 targets** — real, completed semiconductor-sector acquisitions from 2015-2022
  (Xilinx→AMD, Maxim→Analog Devices, Cypress→Infineon, Linear Tech→Analog Devices,
  Atmel→Microchip, Freescale→NXP, Fairchild→ON Semi, and 9 others). Financials are
  each company's **last full fiscal year reported before the deal was announced** —
  deliberately excluding anything disclosed after announcement, so the model can't
  cheat by seeing the future.
- **20 controls** — comparable semiconductor/analog companies that were **not**
  acquired and remain independent today (Skyworks, Qorvo, Microchip, Monolithic Power,
  Lattice, onsemi, and 14 others), using each company's most recent fiscal year.

All figures were sourced via public financial data sites (stockanalysis.com) and, for
the delisted targets, SEC earnings press releases and deal-announcement news coverage —
no paid database was used. Every figure is a real, sourced number; anything that
genuinely couldn't be found is left blank and median-imputed (never fabricated).

**This is the project's main honest limitation**, and worth stating up front: target
financials are point-in-time (pre-2015-2022), control financials are current-day. A
rigorous version of this would sample every company at a matched fiscal year, which
needs a paid historical database (WRDS / Capital IQ) rather than free public sources.
That gap, and how you'd close it with real data access, is exactly the kind of thing
worth raising in an interview.

## Features

Engineered from the raw financials: `log_revenue` (size), `revenue_growth_yoy_pct`,
`gross_margin_pct`, `operating_margin_pct`, `rd_expense_pct_of_revenue`,
`debt_to_revenue`, `cash_to_revenue`, `net_debt_to_revenue`.

## Models & evaluation

Four classifiers, spanning a linear baseline, two non-linear approaches, and an
ensemble for comparison: **Logistic Regression**, **Decision Tree**, **SVM (RBF)**,
**Random Forest**.

With only 36 labeled companies, a held-out test split would leave test folds too small
to trust. Instead every model is evaluated with **Leave-One-Out Cross-Validation**:
each company is scored by a model trained on the other 35, so every prediction is
genuinely out-of-sample.

**A reproducibility bug worth noting.** The first version of this evaluation scored SVM
using `SVC(probability=True).predict_proba()`, whose internal 5-fold Platt-scaling
cross-validation isn't fully deterministic — identical code on identical data produced
ROC-AUC values of 0.075, 0.450, and 0.325 across separate runs, even with `random_state`
set. Wrapping the SVM in `CalibratedClassifierCV(ensemble=False)` instead, with
sklearn's `clone()` for generating fold-level copies, fixed it — repeated runs now
produce byte-identical output.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.472 | 0.385 | 0.312 | 0.345 | 0.419 |
| Decision Tree | 0.833 | 0.812 | 0.812 | 0.812 | 0.780 |
| SVM (RBF) | 0.361 | 0.000 | 0.000 | 0.000 | 0.075 |
| **Random Forest** | **0.722** | 0.714 | 0.625 | 0.667 | **0.803** |

**Finding worth flagging.** Random forest and decision tree — the two tree-based
models — clearly outperform both linear/kernel methods, consistent with
`net_debt_to_revenue` (the top feature by importance) having a non-monotonic
relationship to takeover likelihood: too little debt looks unremarkable, too much looks
distressed, and only a tree can carve out the moderate band in between. What's less
expected is that SVM (RBF) does *worse* than plain logistic regression (0.075 vs. 0.419
ROC-AUC), despite its kernel being able to model non-linear boundaries in theory. The
likely explanation is sample size: under LOOCV each training fold has ~35 companies, and
`CalibratedClassifierCV`'s internal calibration split leaves it fitting on roughly 28 of
those and calibrating on ~7 — too thin to reliably tune a kernel using default
`C=1.0`/`gamma='scale'`. A proper hyperparameter search over `C` and `gamma`, nested
inside the outer LOOCV, would likely close some of this gap, but wasn't run here — the
dataset is too small to support a second layer of cross-validation on top of LOOCV
without folds collapsing to a handful of companies each. Left as an open limitation
rather than tuned away.

**Why random forest is the production model despite lower accuracy/F1 than decision
tree.** Decision tree actually beats random forest on accuracy (0.833 vs. 0.722) and F1
(0.812 vs. 0.667) — but those are both computed from hard 0/1 calls at a single 0.5
threshold. ROC-AUC measures something different: how well the model *ranks* all 36
companies relative to each other, across every possible threshold. With `max_depth=3`,
the decision tree only has a handful of leaves, so it can only output a few distinct
probability values — coarse buckets that happen to land on the right side of 0.5 often,
but that can't finely distinguish "probably not a target" from "definitely not." Random
forest averages 200 trees into a much smoother probability estimate, which ranks
borderline companies better even though its single-threshold hit rate is slightly worse.
Since this project's actual deliverable is a *ranked* screener, not a single yes/no call
on one company, ROC-AUC is the right selection criterion — and the code picks the
production model by it (`max(results, key=lambda k: results[k]["roc_auc"])`) rather than
by accuracy.

## Screener output

Applying the (out-of-fold) model to the 20 currently-independent control companies
ranks them by predicted acquisition-target probability. Top of the list: **Lattice
Semiconductor (63.7%)**, Diodes Incorporated (55.4%), MACOM Technology Solutions
(55.3%), Vishay Intertechnology (51.9%), Microchip Technology (48.6%) — smaller/mid-cap
analog and specialty-chip names with leverage or margin profiles that echo the
historical targets. `outputs/loocv_predictions.csv` now stores all four models' scores
(not just the winner's), and the app's Screener tab lets you switch which model ranks
the list — a direct, interactive view of the SVM-vs-tree disagreement documented above.

This is *not* a forecast ("Lattice will be acquired") — it's a pattern-match against
what acquired companies have historically looked like financially, which is exactly the
kind of first-pass screening signal a corporate strategy team would use to prioritize
which companies to look into more deeply, not to make a call on its own.

**Out-of-fold vs. production-model scores can disagree — on purpose.** The Screener tab
shows *out-of-fold* LOOCV probabilities (each company scored by a model that never saw
its label). The "Score a company" tab, when you load a real company from the dataset,
scores it with the *final* model fit on all 36 companies — including that one. These can
differ meaningfully: Lattice Semiconductor is 63.7% out-of-fold but 32.8% from the
full-fit model. Rather than pick one number and hide the discrepancy, the app shows both
and treats the gap itself as a signal — a company whose score swings a lot depending on
whether it was held out is one the model is less confident about, which matters for how
much weight to put on any single prediction.

## Repo structure

```
ma-target-screener/
├── data/
│   ├── ma_targets.csv        # 16 acquired companies, pre-announcement financials
│   ├── ma_controls.csv       # 20 independent companies, current financials
│   └── dataset.csv           # combined, cleaned, feature-engineered (generated)
├── src/
│   ├── prepare_data.py       # cleaning, feature engineering, imputation
│   └── train_models.py       # LOOCV training/evaluation for all 4 models
├── outputs/
│   ├── model_metrics.json        # accuracy/precision/recall/F1/ROC-AUC per model
│   ├── loocv_predictions.csv     # out-of-fold probability, ALL 4 models, per company
│   ├── feature_importance.json   # coefficients/importances for the best model
│   └── final_model.joblib        # best model + scaler, fit on all 36 companies
└── app/
    └── app.py                # Streamlit: screener, custom scorer, model details
```

## App features

Three tabs, `streamlit run app/app.py`:

- **Screener** — ranked list of the 20 currently-independent companies by predicted
  target probability, colored progress bars, switchable between all 4 models, and a CSV
  download.
- **Score a company** — enter any financial profile by hand, or load one straight from
  the dataset via a dropdown (useful for sanity-checking the model against a company
  whose real outcome you already know). Shows the prediction plus a "why this score"
  table comparing the inputs to the typical acquired-target and typical
  independent-company profile, and — when a real company was loaded — the out-of-fold
  vs. full-fit discrepancy explained above.
- **Model details** — the full metrics table, why random forest was selected over
  decision tree, and feature importance.

If `outputs/` hasn't been generated yet, the app shows a clear message telling you to
run `prepare_data.py` and `train_models.py` first, instead of a raw traceback.

## Running it

```bash
pip install -r requirements.txt
python3 src/prepare_data.py     # builds data/dataset.csv
python3 src/train_models.py     # trains all 4 models, writes outputs/
streamlit run app/app.py        # launches the interactive screener
```

## Extending with real data access

The pipeline is deliberately decoupled from the data source — `prepare_data.py` just
needs a CSV with the same columns. With access to a proper historical database
(WRDS, Capital IQ, or similar), the natural next steps are: (1) pull a properly
point-in-time-matched panel across hundreds of companies instead of 36, closing the
project's main current limitation, (2) add deal-level features (premium paid, stock
vs. cash, strategic vs. financial buyer) to build the **deal-outcome predictor**
(completed vs. terminated) as a second model, and (3) try narrowing or widening the
universe — a different sector, or a specific region — to see how much the pattern
generalizes beyond semiconductors.