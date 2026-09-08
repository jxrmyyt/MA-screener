const API_BASE = process.env.NEXT_PUBLIC_API_BASE ?? "http://localhost:8000";

export class ScreenerApiError extends Error {
  errorCode: string;

  constructor(errorCode: string, message: string) {
    super(message);
    this.errorCode = errorCode;
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    ...init,
    headers: { "Content-Type": "application/json", ...init?.headers },
  });
  const body = await res.json();

  if (!res.ok) {
    throw new ScreenerApiError(
      body.error_code ?? "unknown_error",
      body.message ?? "Something went wrong talking to the screener API."
    );
  }

  return body as T;
}

export interface SectorBreakdown {
  sector: string;
  targets: number;
  controls: number;
}

export interface Meta {
  n_companies: number;
  n_targets: number;
  n_controls: number;
  sectors: string[];
  sector_breakdown: SectorBreakdown[];
}

export function fetchMeta() {
  return request<Meta>("/api/meta");
}

export interface ScreenerRow {
  rank: number;
  company: string;
  sector: string;
  predicted_probability_pct: number;
}

export interface ScreenerResponse {
  model: string;
  model_display_name: string;
  is_best_model: boolean;
  sector_filter: string | null;
  rows: ScreenerRow[];
}

export function fetchScreener(model?: string, sector?: string) {
  const params = new URLSearchParams();
  if (model) params.set("model", model);
  if (sector) params.set("sector", sector);
  const query = params.toString();
  return request<ScreenerResponse>(`/api/screener${query ? `?${query}` : ""}`);
}

export interface ModelMetrics {
  accuracy: number;
  precision: number;
  recall: number;
  f1: number;
  roc_auc: number;
}

export interface ModelsResponse {
  best_model: string;
  metrics: Record<string, ModelMetrics>;
  n_companies: number;
}

export function fetchModels() {
  return request<ModelsResponse>("/api/models");
}

export interface FeatureImportanceResponse {
  model: string;
  importance: Record<string, number>;
}

export function fetchFeatureImportance() {
  return request<FeatureImportanceResponse>("/api/feature-importance");
}

export interface CompanySummary {
  company: string;
  sector: string;
}

export function fetchCompanies() {
  return request<CompanySummary[]>("/api/companies");
}

export interface CompanyDetail {
  company: string;
  sector: string;
  annual_revenue_usd_millions: number;
  revenue_growth_yoy_pct: number;
  gross_margin_pct: number;
  operating_margin_pct: number;
  rd_expense_pct_of_revenue: number;
  total_debt_usd_millions: number;
  cash_and_equivalents_usd_millions: number;
}

export function fetchCompany(name: string) {
  return request<CompanyDetail>(`/api/companies/${encodeURIComponent(name)}`);
}

export interface ScoreRequest {
  sector: string;
  annual_revenue_usd_millions: number;
  revenue_growth_yoy_pct: number;
  gross_margin_pct: number;
  operating_margin_pct: number;
  rd_expense_pct_of_revenue: number;
  total_debt_usd_millions: number;
  cash_and_equivalents_usd_millions: number;
  compare_to_company?: string | null;
}

export interface ComparisonRow {
  field: string;
  label: string;
  this_company: number;
  sector_target_median: number;
  sector_control_median: number;
}

export interface ScoreResponse {
  predicted_probability_pct: number;
  model_used: string;
  out_of_fold_probability_pct: number | null;
  out_of_fold_company: string | null;
  comparison: ComparisonRow[];
  thin_sector_data: boolean;
  sector_target_count: number;
  sector_control_count: number;
}

export function scoreCompany(body: ScoreRequest) {
  return request<ScoreResponse>("/api/score", {
    method: "POST",
    body: JSON.stringify(body),
  });
}
