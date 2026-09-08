"""FastAPI backend for the M&A Target Screener.

Wraps the existing pipeline (src/prepare_data.py, plus the artifacts
src/train_models.py already writes to outputs/) in REST endpoints, so a
separate frontend can drive the same sector-relative scoring logic that
powers the Streamlit app. Nothing about the modeling changes here -- this
file is HTTP plumbing around prepare_data.py and outputs/, not a rewrite
of either.
"""
from __future__ import annotations

import json
import logging
import os
import sys
from functools import lru_cache

import joblib
import pandas as pd
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

# prepare_data.py reads data/ma_targets.csv etc. as paths relative to the
# working directory -- true today because the Streamlit app is always run
# with `streamlit run app/app.py` from the project root. uvicorn has no such
# convention, so pin the working directory here rather than editing the
# already-working data-loading code.
os.chdir(ROOT)

from prepare_data import build_dataset, engineer_features, zscore_row  # noqa: E402

logger = logging.getLogger("ma_screener_api")
logging.basicConfig(level=logging.INFO)

app = FastAPI(title="M&A Target Screener API", version="1.0.0")

# Always allow local dev; add the deployed frontend's real origin(s) via an
# env var so the allow-list doesn't need a code change (and a redeploy) every
# time the frontend's URL changes. Set CORS_ORIGINS on Render, e.g.
# "https://ma-target-screener.vercel.app" or a comma-separated list.
_extra_origins = [o.strip() for o in os.environ.get("CORS_ORIGINS", "").split(",") if o.strip()]
ALLOWED_ORIGINS = ["http://localhost:3000"] + _extra_origins

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

DISPLAY_NAMES = {
    "logistic_regression": "Logistic Regression",
    "decision_tree": "Decision Tree",
    "svm_rbf": "SVM (RBF)",
    "random_forest": "Random Forest",
}

RAW_FIELDS = [
    "annual_revenue_usd_millions",
    "revenue_growth_yoy_pct",
    "gross_margin_pct",
    "operating_margin_pct",
    "rd_expense_pct_of_revenue",
    "total_debt_usd_millions",
    "cash_and_equivalents_usd_millions",
]

FIELD_LABELS = {
    "annual_revenue_usd_millions": "Annual revenue ($M)",
    "revenue_growth_yoy_pct": "Revenue growth YoY (%)",
    "gross_margin_pct": "Gross margin (%)",
    "operating_margin_pct": "Operating margin (%)",
    "rd_expense_pct_of_revenue": "R&D expense (% of revenue)",
    "total_debt_usd_millions": "Total debt ($M)",
    "cash_and_equivalents_usd_millions": "Cash & equivalents ($M)",
}

REQUIRED_OUTPUTS = [
    "loocv_predictions.csv", "model_metrics.json", "feature_importance.json",
    "final_model.joblib", "sector_baselines.json",
]


# ---------------------------------------------------------------------------
# Domain errors -- each gets its own error_code so the frontend can render a
# specific screen instead of one generic failure banner for everything.
# ---------------------------------------------------------------------------

class PipelineNotReadyError(Exception):
    def __init__(self, missing: list[str]):
        self.missing = missing
        super().__init__(f"missing outputs: {missing}")


@app.exception_handler(PipelineNotReadyError)
async def handle_not_ready(_, exc: PipelineNotReadyError):
    return JSONResponse(
        status_code=503,
        content={
            "error_code": "pipeline_not_trained",
            "message": (
                "The model hasn't been trained yet -- outputs/ is missing "
                f"{', '.join(exc.missing)}. From the project root, run "
                "`python3 src/prepare_data.py` then `python3 src/train_models.py`."
            ),
        },
    )


class UnknownSectorError(Exception):
    def __init__(self, sector: str, valid: list[str]):
        self.sector, self.valid = sector, valid
        super().__init__(sector)


@app.exception_handler(UnknownSectorError)
async def handle_unknown_sector(_, exc: UnknownSectorError):
    return JSONResponse(
        status_code=400,
        content={
            "error_code": "unknown_sector",
            "message": f"'{exc.sector}' isn't one of the trained sectors: {', '.join(exc.valid)}.",
        },
    )


class UnknownModelError(Exception):
    def __init__(self, model: str, valid: list[str]):
        self.model, self.valid = model, valid
        super().__init__(model)


@app.exception_handler(UnknownModelError)
async def handle_unknown_model(_, exc: UnknownModelError):
    return JSONResponse(
        status_code=400,
        content={
            "error_code": "unknown_model",
            "message": f"'{exc.model}' isn't a trained model. Choose from: {', '.join(exc.valid)}.",
        },
    )


class CompanyNotFoundError(Exception):
    def __init__(self, name: str):
        self.name = name
        super().__init__(name)


@app.exception_handler(CompanyNotFoundError)
async def handle_company_not_found(_, exc: CompanyNotFoundError):
    return JSONResponse(
        status_code=404,
        content={
            "error_code": "company_not_found",
            "message": f"No company named '{exc.name}' in the training dataset.",
        },
    )


# ---------------------------------------------------------------------------
# Cached loaders -- same artifacts the Streamlit app reads from outputs/,
# loaded once per process instead of once per request.
# ---------------------------------------------------------------------------

def _outputs_dir() -> str:
    return os.path.join(ROOT, "outputs")


def _require_outputs():
    missing = [p for p in REQUIRED_OUTPUTS if not os.path.exists(os.path.join(_outputs_dir(), p))]
    if missing:
        raise PipelineNotReadyError(missing)


@lru_cache
def get_predictions() -> pd.DataFrame:
    _require_outputs()
    return pd.read_csv(os.path.join(_outputs_dir(), "loocv_predictions.csv"))


@lru_cache
def get_metrics() -> dict:
    _require_outputs()
    with open(os.path.join(_outputs_dir(), "model_metrics.json")) as f:
        return json.load(f)


@lru_cache
def get_importance() -> dict:
    _require_outputs()
    with open(os.path.join(_outputs_dir(), "feature_importance.json")) as f:
        return json.load(f)


@lru_cache
def get_baselines() -> dict:
    _require_outputs()
    with open(os.path.join(_outputs_dir(), "sector_baselines.json")) as f:
        return json.load(f)


@lru_cache
def get_model_bundle():
    _require_outputs()
    return joblib.load(os.path.join(_outputs_dir(), "final_model.joblib"))


@lru_cache
def get_dataset() -> pd.DataFrame:
    _require_outputs()
    df, _, _ = build_dataset()
    return df


def best_model_name() -> str:
    return get_importance()["model"]


def available_models() -> list[str]:
    preds = get_predictions()
    return [c.replace("_predicted_prob", "") for c in preds.columns if c.endswith("_predicted_prob")]


# ---------------------------------------------------------------------------
# Response schemas
# ---------------------------------------------------------------------------

class ScreenerRow(BaseModel):
    rank: int
    company: str
    sector: str
    predicted_probability_pct: float


class ScreenerResponse(BaseModel):
    model: str
    model_display_name: str
    is_best_model: bool
    sector_filter: str | None
    rows: list[ScreenerRow]


class ModelMetrics(BaseModel):
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float


class ModelsResponse(BaseModel):
    best_model: str
    metrics: dict[str, ModelMetrics]
    n_companies: int


class FeatureImportanceResponse(BaseModel):
    model: str
    importance: dict[str, float]


class SectorBreakdown(BaseModel):
    sector: str
    targets: int
    controls: int


class MetaResponse(BaseModel):
    n_companies: int
    n_targets: int
    n_controls: int
    sectors: list[str]
    sector_breakdown: list[SectorBreakdown]


class CompanySummary(BaseModel):
    company: str
    sector: str


class CompanyDetail(BaseModel):
    company: str
    sector: str
    annual_revenue_usd_millions: float
    revenue_growth_yoy_pct: float
    gross_margin_pct: float
    operating_margin_pct: float
    rd_expense_pct_of_revenue: float
    total_debt_usd_millions: float
    cash_and_equivalents_usd_millions: float


class ScoreRequest(BaseModel):
    sector: str
    annual_revenue_usd_millions: float
    revenue_growth_yoy_pct: float
    gross_margin_pct: float
    operating_margin_pct: float
    rd_expense_pct_of_revenue: float
    total_debt_usd_millions: float
    cash_and_equivalents_usd_millions: float
    compare_to_company: str | None = None


class ComparisonRow(BaseModel):
    field: str
    label: str
    this_company: float
    sector_target_median: float
    sector_control_median: float


class ScoreResponse(BaseModel):
    predicted_probability_pct: float
    model_used: str
    out_of_fold_probability_pct: float | None
    out_of_fold_company: str | None
    comparison: list[ComparisonRow]
    thin_sector_data: bool
    sector_target_count: int
    sector_control_count: int


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/meta", response_model=MetaResponse)
def get_meta():
    dataset = get_dataset()
    n_targets = int(dataset["label"].sum())
    breakdown = dataset.groupby(["sector", "label"]).size().unstack(fill_value=0)
    sector_breakdown = [
        SectorBreakdown(sector=sector, targets=int(row.get(1, 0)), controls=int(row.get(0, 0)))
        for sector, row in breakdown.iterrows()
    ]
    return MetaResponse(
        n_companies=len(dataset),
        n_targets=n_targets,
        n_controls=len(dataset) - n_targets,
        sectors=sorted(dataset["sector"].unique().tolist()),
        sector_breakdown=sector_breakdown,
    )


@app.get("/api/screener", response_model=ScreenerResponse)
def get_screener(
    model: str | None = Query(default=None, description="Defaults to the best model by ROC-AUC"),
    sector: str | None = Query(default=None, description="Filter to one sector"),
):
    preds = get_predictions()
    models = available_models()
    best = best_model_name()
    chosen = model or best

    if chosen not in models:
        raise UnknownModelError(chosen, models)

    dataset = get_dataset()
    valid_sectors = sorted(dataset["sector"].unique().tolist())
    if sector and sector not in valid_sectors:
        raise UnknownSectorError(sector, valid_sectors)

    prob_col = f"{chosen}_predicted_prob"
    controls = preds[preds["actual_label"] == 0][["company", "sector", prob_col]].copy()
    if sector:
        controls = controls[controls["sector"] == sector]

    controls[prob_col] = (controls[prob_col].clip(0, 1) * 100).round(1)
    controls = controls.sort_values(prob_col, ascending=False).reset_index(drop=True)

    rows = [
        ScreenerRow(
            rank=i + 1,
            company=row["company"],
            sector=row["sector"],
            predicted_probability_pct=row[prob_col],
        )
        for i, row in controls.iterrows()
    ]

    return ScreenerResponse(
        model=chosen,
        model_display_name=DISPLAY_NAMES.get(chosen, chosen),
        is_best_model=chosen == best,
        sector_filter=sector,
        rows=rows,
    )


@app.get("/api/models", response_model=ModelsResponse)
def get_models():
    metrics = get_metrics()
    dataset = get_dataset()
    return ModelsResponse(
        best_model=best_model_name(),
        metrics={k: ModelMetrics(**v) for k, v in metrics.items()},
        n_companies=len(dataset),
    )


@app.get("/api/feature-importance", response_model=FeatureImportanceResponse)
def get_feature_importance():
    importance = get_importance()
    return FeatureImportanceResponse(model=importance["model"], importance=importance["importance"])


@app.get("/api/companies", response_model=list[CompanySummary])
def list_companies():
    dataset = get_dataset()
    return [
        CompanySummary(company=row["company"], sector=row["sector"])
        for _, row in dataset.sort_values("company").iterrows()
    ]


@app.get("/api/companies/{name}", response_model=CompanyDetail)
def get_company(name: str):
    dataset = get_dataset()
    match = dataset[dataset["company"] == name]
    if match.empty:
        raise CompanyNotFoundError(name)
    row = match.iloc[0]
    return CompanyDetail(
        company=row["company"],
        sector=row["sector"],
        **{f: float(row[f]) for f in RAW_FIELDS},
    )


@app.post("/api/score", response_model=ScoreResponse)
def score_company(body: ScoreRequest):
    dataset = get_dataset()
    valid_sectors = sorted(dataset["sector"].unique().tolist())
    if body.sector not in valid_sectors:
        raise UnknownSectorError(body.sector, valid_sectors)

    bundle = get_model_bundle()
    model, scaler, features = bundle["model"], bundle["scaler"], bundle["features"]
    baselines = get_baselines()

    raw_row = pd.DataFrame([{f: getattr(body, f) for f in RAW_FIELDS}])
    raw_row = engineer_features(raw_row)
    engineered = raw_row.iloc[0].to_dict()
    engineered.update(zscore_row(engineered, body.sector, baselines))

    X = pd.DataFrame([engineered])[features].values
    X_scaled = scaler.transform(X)
    prob = float(model.predict_proba(X_scaled)[0][1])

    out_of_fold_prob = None
    out_of_fold_company = None
    if body.compare_to_company:
        preds = get_predictions()
        best = best_model_name()
        screener_row = preds[preds["company"] == body.compare_to_company]
        if not screener_row.empty:
            out_of_fold_prob = round(float(screener_row.iloc[0][f"{best}_predicted_prob"]) * 100, 1)
            out_of_fold_company = body.compare_to_company

    sector_data = dataset[dataset["sector"] == body.sector]
    target_pool = sector_data[sector_data["label"] == 1]
    control_pool = sector_data[sector_data["label"] == 0]
    target_source = target_pool if len(target_pool) else dataset[dataset["label"] == 1]
    control_source = control_pool if len(control_pool) else dataset[dataset["label"] == 0]

    comparison = [
        ComparisonRow(
            field=f,
            label=FIELD_LABELS[f],
            this_company=getattr(body, f),
            sector_target_median=round(float(target_source[f].median()), 1),
            sector_control_median=round(float(control_source[f].median()), 1),
        )
        for f in RAW_FIELDS
    ]

    return ScoreResponse(
        predicted_probability_pct=round(prob * 100, 1),
        model_used=best_model_name(),
        out_of_fold_probability_pct=out_of_fold_prob,
        out_of_fold_company=out_of_fold_company,
        comparison=comparison,
        thin_sector_data=len(target_pool) < 4 or len(control_pool) < 4,
        sector_target_count=len(target_pool),
        sector_control_count=len(control_pool),
    )
