# Cross-Industry M&A Target Screener

A machine learning project that predicts which companies look like acquisition targets, based on their financial profile — now spanning six sectors rather than just one.
I have been increasingly drawn to M&A and corporate strategy — the mix of financial analysis and pattern-recognition in figuring out why a company becomes a target feels like a natural place to point machine learning. This project is me testing that: can a model trained on real, completed deals actually pick up a signal in the kind of financial profile that tends to attract an acquirer, using nothing but public data and standard classifiers? This started as a semiconductor-only screener; this version expands it across semiconductors, software, healthcare, industrials, consumer/retail, and telecom, to see whether the pattern holds beyond one industry.
## The question

Given a company's financials — revenue, growth, margins, leverage, R&D intensity, can we predict whether it looks like the kind of company that gets acquired? And does that answer generalize once "financial profile" has to mean something comparable across very different industries?
## Data

Two hand-curated, sourced datasets (data/ma_targets.csv, data/ma_controls.csv), 116 companies total across six sectors:
| Sector | Targets (acquired) | Controls (independent) |
| --- | ---: | ---: |
| Semiconductor | 16 | 20 |
| Software | 11 | 9 |
| Healthcare | 10 | 10 |
| Industrials | 10 | 10 |
| Consumer/Retail | 7 | 7 |
| Telecom | 3 | 3 |
| **Total** | **57** | **59** |
- **Targets:** real, completed acquisitions (Xilinx→AMD, Maxim→Analog Devices, Magento→Adobe, Marketo→Adobe, MyoKardia→Bristol Myers Squibb, Rofin-Sinar→Coherent, Smartsheet→Blackstone/Vista, and dozens of others). Financials are each company's last full fiscal year reported before the deal was announced — deliberately excluding anything disclosed after announcement, so the model can't cheat by seeing the future.
- **Controls:** comparable companies in the same sector that were not acquired and remain independent today (Skyworks, Qorvo, HubSpot, Atlassian, Workday, Ingersoll Rand, and many others), using each company's most recent fiscal year.
All figures were sourced via public financial data sites (stockanalysis.com) and, for delisted targets, SEC earnings press releases and deal-announcement news coverage — no paid database was used. Every figure is a real, sourced number; anything that genuinely couldn't be found is left blank and imputed (never fabricated). One company originally researched as a control (Smartsheet) turned out to have been taken private mid-project — it was reclassified as a target rather than dropped or left mislabeled.
The project's main honest limitation worth stating up front: target financials are point-in-time (spanning roughly 2015-2025 depending on the deal), control financials are current-day. A rigorous version of this would sample every company at a matched fiscal year, which needs a paid historical database (WRDS / Capital IQ) rather than free public sources. 
## How the data was researched

The original 36-company semiconductor dataset was hand-sourced one company at a time. For the cross-industry expansion, gathering ~80 more companies' financials by hand would have taken far longer than the modeling work itself, so the research step was split across four parallel AI research agents, one per new sector (software, healthcare, industrials, and consumer/retail + telecom combined). Each agent was given the same brief: find real, completed acquisitions and comparable still-independent companies in its sector, pull each one's pre-announcement (for targets) or most recent (for controls) fiscal-year financials from public sources, and use N/A rather than invent a number it couldn't verify. Running the four sectors in parallel rather than sequentially was purely a time-saving choice — each agent's output is a plain CSV with the same schema as the original hand-built one, and every row was spot-checked before being merged into data/ma_targets.csv / data/ma_controls.csv.
This division of labor caught a few things a single pass likely would have missed: the agents flagged and corrected several announcement dates I'd supplied incorrectly (I'd mixed up a deal's announcement date with its later closing date in a few cases — IXYS, IDT, Quantenna, MyoKardia, Esterline, and KLX), and one agent flagged that a company I'd listed as a still-independent control (Smartsheet) had actually been taken private since — which is why it's now a target instead. The raw per-sector files each agent produced are kept in data/extended/ for anyone who wants to see the sourcing before it was merged.
## Handling missing data

Not every company discloses every field cleanly — rd_expense_pct_of_revenue in particular is missing for roughly 40% of the 116 companies, since a lot of industrials, consumer/retail, and older filings don't break R&D out as its own line item the way tech companies do. Two different imputation rules apply, depending on the feature:
- The seven ratio features (growth, margins, R&D intensity, the three debt/cash ratios) are never median-filled on their raw scale. Instead, a missing value skips the winsorize-and-z-score step in `zscore_row()` entirely and is assigned `sector_z_<feature> = 0.0` directly — and since 0.0 on a z-score scale means "exactly at the sector median," this is equivalent to treating a company with an undisclosed R&D figure as "R&D-typical for its sector" rather than guessing a number.
- `log_revenue` (from `annual_revenue_usd_millions`, missing for exactly 1 company) is filled with the plain global median of `log_revenue` across all 116 companies, since company size was deliberately left un-normalized by sector.
Both choices are reasonable given the constraint of free public data, but both are worth being upfront about: a stricter version of this project would either drop rows with too many missing fields or use a proper multivariate imputation method (e.g. IterativeImputer) instead of a single fallback value per feature.
## Features and cross-industry normalization

Engineered from the raw financials: log_revenue (size, kept on a global/absolute scale), plus seven ratio features — revenue_growth_yoy_pct, gross_margin_pct, operating_margin_pct, rd_expense_pct_of_revenue, debt_to_revenue, cash_to_revenue, net_debt_to_revenue.
Going cross-industry changes what "features" has to mean. A 25% R&D-to-revenue ratio is unremarkable for a software company and extreme for an industrial one — feeding raw ratios into a model trained across six sectors would mostly teach it "this is a software company," not anything about acquisition likelihood. So, every ratio feature is converted to a sector-relative z-score: how many sector-standard-deviations this company's ratio sits from its own sector's median, computed after winsorizing (clipping) extreme values so a handful of pre-commercial biotech names with R&D at 600%+ of revenue don't single-handedly blow up a sector's baseline. A sector with fewer than 4 companies (currently none but built to be robust if the dataset gets sparser) falls back to a whole-dataset baseline instead of an unreliable sector-specific one. Company size (log_revenue) is left un-normalized, since absolute deal capacity plausibly matters regardless of industry.
This also had to work at scoring time, not just training time: prepare_data.py separates compute_sector_baselines() (run once over the full training set, saved to outputs/sector_baselines.json) from zscore_row() (applied to one company using the saved baselines), so the app can score a brand-new company against the sectors the model already learned without needing the full training data at inference time or leaking that company into its own baseline.
## Models & evaluation

Four classifiers, spanning a linear baseline, two non-linear approaches, and an ensemble for comparison: Logistic Regression, Decision Tree, SVM (RBF), Random Forest.
With only 116 labeled companies, a held-out test split would leave test folds too small to trust. Instead, every model is evaluated with Leave-One-Out Cross-Validation: each company is scored by a model trained on the other 115, so every prediction is genuinely out-of-sample.
A reproducibility bug worth noting. The first version of this evaluation scored SVM using SVC(probability=True).predict_proba(), where the internal 5-fold Platt-scaling cross-validation isn't fully deterministic — identical code on identical data produced different ROC-AUC values across separate runs, even with random_state set. Wrapping the SVM in CalibratedClassifierCV(ensemble=False) instead, with sklearn's clone() for generating fold-level copies, fixed it — repeated runs now produce identical output.
| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | 0.690 | 0.684 | 0.684 | 0.684 | 0.769 |
| Decision Tree | 0.629 | 0.635 | 0.579 | 0.606 | 0.655 |
| SVM (RBF) | 0.672 | 0.656 | 0.702 | 0.678 | 0.720 |
| **Random Forest** | **0.750** | **0.804** | **0.649** | **0.718** | **0.814** |
How the finding changed with more, more varied data. In the original 36-company, semiconductor-only version, the linear/kernel models badly underperformed the tree-based ones (logistic regression 0.419 ROC-AUC, SVM 0.075) — a gap traced to net_debt_to_revenue having a non-monotonic relationship with takeover likelihood that only a tree could carve out, combined with too little data for the SVM's kernel to calibrate reliably.
With 116 companies across six sectors, that gap has mostly closed: logistic regression now reaches 0.769 and SVM 0.720, both respectable. Two things plausibly changed. First, more data lets every model — especially the SVM's internal calibration split — fit and calibrate on enough companies to stop being noise-dominated. Second, the sector-relative z-scoring may have straightened out some of the non-monotonic pattern the original model was working around: a ratio that reads as "too much debt" in one sector and "unremarkable leverage" in another was previously mixed into one raw scale, which a non-linear model handles more gracefully than a linear one. Once normalized within-sector, some of what looked like non-linearity may really have been sector-mismatch. Feature importance in the current model still puts sector_z_operating_margin_pct (+0.211) and sector_z_debt_to_revenue (+0.162) at the top, followed by log_revenue (+0.145) and sector_z_net_debt_to_revenue (+0.126) — leverage and profitability, now compared within-sector, remain the strongest signals.
Why random forest is still the production model. Random forest wins clearly on ROC-AUC (0.814, vs. 0.769 for the next-best logistic regression) despite logistic regression edging it slightly on raw accuracy/precision/recall trade-offs in places. ROC-AUC measures how well a model ranks all 116 companies relative to each other across every possible threshold, rather than how it performs at one arbitrary 0.5 cutoff — and since this project's actual deliverable is a ranked screener, not a single yes/no call on one company, ROC-AUC is the right selection criterion. The code picks the production model by it (max(results, key=lambda k: results[k]["roc_auc"])) rather than by accuracy.
## Screener output

Applying the (out-of-fold) model to the 59 currently-independent control companies ranks them by predicted acquisition-target probability. Top of the list: SiTime Corporation (67.9%, Semiconductor), Asana (66.3%, Software), Synaptics (66.2%, Semiconductor), CEVA Inc (62.4%, Semiconductor), Ingersoll Rand (60.2%, Industrials) — notably, a mix of sectors rather than a semiconductor-only list, which is the direct payoff of going cross-industry. outputs/loocv_predictions.csv stores all four models' scores (not just the winner's) plus each company's sector, and the app's Screener tab lets you switch which model ranks the list and filter by sector.
This is not a forecast ("SiTime will be acquired") — it's a pattern-match against what acquired companies have historically looked like financially, which is exactly the kind of first-pass screening signal a corporate strategy team would use to prioritize which companies to look into more deeply, not to make a call on its own.
Out-of-fold vs. production-model scores can disagree — on purpose. The Screener tab shows out-of-fold LOOCV probabilities (each company scored by a model that never saw its label). The "Score a company" tab, when you load a real company from the dataset, scores it with the final model fit on all 116 companies — including that one. These can differ meaningfully. Rather than pick one number and hide the discrepancy, the app shows both and treats the gap itself as a signal — a company whose score swings a lot depending on whether it was held out is one the model is less confident about, which matters for how much weight to put on any single prediction.
## Forward validation (checked September 2026)

A proper backtest isn't feasible here — see "Handling missing data" and the point-in-time limitation above; control companies only have current-day financials. Thus, there's no fair way to reconstruct what they looked like at some earlier cutoff date. What is checkable is the reverse: since the top of the screener's ranking is a list of real, named companies, it's possible to look up whether any of them have had real M&A activity since their financials were pulled. This isn't a rigorous evaluation — 15 companies is a small, non-random sample and the lookback window is only a few months — but it's a genuine out-of-sample check rather than anything reconstructed from the training data, so it's worth recording honestly, hit and miss both.
Checking the top 15 currently independent companies by predicted probability:

| Company | Predicted probability | Outcome since |
| --- | ---: | --- |
| Synaptics (rank 3) | 66.2% | Agreed to be acquired by onsemi (~$7B all-stock, announced ~June 2026) |
| Allegro MicroSystems (rank 6) | 57.8% | Received an unsolicited ~$6.9B takeover offer from onsemi; board rejected it as "inadequate" |
| Ambarella (rank 8) | 53.8% | Reported acquisition talks with NXP Semiconductors; talks appear to have lapsed without a signed deal |
| Alnylam Pharmaceuticals (rank 10) | 49.0% | Named in analyst speculation as a possible target for Novartis; no reported approach or talks |
| Remaining 11 companies | 48–68% | No M&A-target news found |
The headline result: the company the model ranked third out of 59 independent companies is now the subject of an actual signed acquisition agreement, and the company ranked sixth received a real, reported nine-figure buyout offer. Two more of the top 10 had credible-to-speculative M&A chatter. That's a small sample, and "credible rumor" and "signed deal" are being lumped together loosely in this table — this is a directional signal to mention in conversation, not a claim that the model is validated in any statistical sense. It's also worth being honest about the miss rate: 11 of the top 15 have no matching news at all, which is exactly what you'd expect from a model that's pattern-matching a historical profile rather than forecasting — most companies that look like a target on paper still don't get acquired in any given few-month window. Re-checking this list periodically (the app's Screener tab makes it easy to re-pull the current top N) is the closest thing this project has to an ongoing backtest.

## Repo structure

```text
ma-target-screener/
├── data/
│   ├── ma_targets.csv          # 57 acquired companies across 6 sectors, pre-announcement financials
│   ├── ma_controls.csv         # 59 independent companies across 6 sectors, current financials
│   ├── extended/                # raw per-sector research CSVs used to build the expansion
│   └── dataset.csv             # combined, cleaned, feature-engineered (generated)
├── src/
│   ├── prepare_data.py         # cleaning, feature engineering, sector-relative z-scoring, imputation
│   └── train_models.py         # LOOCV training/evaluation for all 4 models
├── outputs/
│   ├── model_metrics.json        # accuracy/precision/recall/F1/ROC-AUC per model
│   ├── loocv_predictions.csv     # out-of-fold probability, ALL 4 models, per company + sector
│   ├── feature_importance.json   # coefficients/importances for the best model
│   ├── sector_baselines.json     # per-sector median/std used for z-scoring (+ global fallback)
│   └── final_model.joblib         # best model + scaler, fit on all 116 companies
└── app/
    └── app.py                  # Streamlit: screener, custom scorer, model details
```


## App features

Three tabs, launched with `streamlit run app/app.py`:

- **Screener** — ranked list of the 59 currently-independent companies by predicted target probability, colored progress bars, switchable between all 4 models, filterable by sector, and a CSV download.
- **Score a company** — enter any financial profile by hand, or load one straight from the dataset via a dropdown. Shows the prediction plus a "why this score" table comparing the inputs to typical acquired-target and independent-company profiles within that sector.
- **Model details** — the full metrics table, why random forest was selected, feature importance, and a breakdown of companies by sector.
If outputs/ hasn't been generated yet, the app shows a clear message telling you to run prepare_data.py and train_models.py first, instead of a raw traceback.
## Running it

```bash
pip install -r requirements.txt
python3 src/prepare_data.py     # builds data/dataset.csv
python3 src/train_models.py     # trains all 4 models, writes outputs/
streamlit run app/app.py        # launches the interactive screener
```
## Extending with real data access

The pipeline is deliberately decoupled from the data source — prepare_data.py just needs a CSV with a sector column and the same numeric columns. With access to a proper historical database (WRDS, Capital IQ, or similar), the natural next steps are: (1) pull a properly point-in-time-matched panel across hundreds of companies per sector instead of a few dozen, closing the project's main current limitation, (2) add deal-level features (premium paid, stock vs. cash, strategic vs. financial buyer) to build the deal-outcome predictor (completed vs. terminated) as a second model, and (3) push further into sectors that are currently thin (Telecom has only 6 companies total) or add entirely new ones, to keep testing how far the sector-relative approach generalizes.
