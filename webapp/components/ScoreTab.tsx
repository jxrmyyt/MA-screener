"use client";

import { useEffect, useState } from "react";
import {
  ScreenerApiError,
  fetchCompanies,
  fetchCompany,
  scoreCompany,
  type CompanySummary,
  type ScoreResponse,
} from "@/lib/api";
import { ErrorScreen } from "@/components/ErrorScreen";

const FIELDS: { key: keyof typeof DEFAULTS; label: string; step: number }[] = [
  { key: "annual_revenue_usd_millions", label: "Annual revenue ($M)", step: 10 },
  { key: "revenue_growth_yoy_pct", label: "Revenue growth YoY (%)", step: 0.5 },
  { key: "gross_margin_pct", label: "Gross margin (%)", step: 0.5 },
  { key: "operating_margin_pct", label: "Operating margin (%)", step: 0.5 },
  { key: "rd_expense_pct_of_revenue", label: "R&D expense (% of revenue)", step: 0.5 },
  { key: "total_debt_usd_millions", label: "Total debt ($M)", step: 10 },
  { key: "cash_and_equivalents_usd_millions", label: "Cash & equivalents ($M)", step: 10 },
];

const DEFAULTS = {
  annual_revenue_usd_millions: 800,
  revenue_growth_yoy_pct: 8,
  gross_margin_pct: 50,
  operating_margin_pct: 10,
  rd_expense_pct_of_revenue: 20,
  total_debt_usd_millions: 100,
  cash_and_equivalents_usd_millions: 200,
};

export function ScoreTab({ sectors }: { sectors: string[] }) {
  const [companies, setCompanies] = useState<CompanySummary[]>([]);
  const [selectedCompany, setSelectedCompany] = useState("");
  const [sector, setSector] = useState(sectors[0] ?? "");
  const [values, setValues] = useState({ ...DEFAULTS });
  const [result, setResult] = useState<ScoreResponse | null>(null);
  const [error, setError] = useState<ScreenerApiError | null>(null);
  const [submitting, setSubmitting] = useState(false);

  useEffect(() => {
    fetchCompanies().then(setCompanies).catch(() => setCompanies([]));
  }, []);

  async function loadCompany(name: string) {
    setSelectedCompany(name);
    if (!name) return;
    const detail = await fetchCompany(name);
    setSector(detail.sector);
    setValues({
      annual_revenue_usd_millions: detail.annual_revenue_usd_millions,
      revenue_growth_yoy_pct: detail.revenue_growth_yoy_pct,
      gross_margin_pct: detail.gross_margin_pct,
      operating_margin_pct: detail.operating_margin_pct,
      rd_expense_pct_of_revenue: detail.rd_expense_pct_of_revenue,
      total_debt_usd_millions: detail.total_debt_usd_millions,
      cash_and_equivalents_usd_millions: detail.cash_and_equivalents_usd_millions,
    });
  }

  async function submit() {
    setSubmitting(true);
    setError(null);
    try {
      const res = await scoreCompany({
        sector,
        ...values,
        compare_to_company: selectedCompany || null,
      });
      setResult(res);
    } catch (err) {
      setResult(null);
      setError(err instanceof ScreenerApiError ? err : new ScreenerApiError("unknown_error", String(err)));
    } finally {
      setSubmitting(false);
    }
  }

  const verdict =
    result === null
      ? null
      : result.predicted_probability_pct > 60
      ? { tone: "border-amber-200 bg-amber-50 text-amber-800", text: "High likelihood — resembles historical acquisition targets in this sector." }
      : result.predicted_probability_pct > 40
      ? { tone: "border-blue-200 bg-blue-50 text-blue-800", text: "Moderate likelihood — mixed signal." }
      : { tone: "border-green-200 bg-green-50 text-green-800", text: "Low likelihood — resembles companies that stayed independent in this sector." };

  return (
    <div className="space-y-6">
      <div>
        <label className="mb-1 block text-sm font-medium text-slate-700">
          Load a real company (optional)
        </label>
        <select
          value={selectedCompany}
          onChange={(e) => loadCompany(e.target.value)}
          className="w-full rounded-md border border-slate-300 px-3 py-2 text-sm shadow-sm"
        >
          <option value="">-- manual entry --</option>
          {companies.map((c) => (
            <option key={c.company} value={c.company}>
              {c.company} ({c.sector})
            </option>
          ))}
        </select>
        <p className="mt-1 text-xs text-slate-400">
          Prefills the fields below, including sector, so you can sanity-check the model against a
          company you already know the outcome for.
        </p>
      </div>

      <div>
        <label className="mb-1 block text-sm font-medium text-slate-700">Sector</label>
        <select
          value={sector}
          onChange={(e) => setSector(e.target.value)}
          className="w-full rounded-md border border-slate-300 px-3 py-2 text-sm shadow-sm"
        >
          {sectors.map((s) => (
            <option key={s} value={s}>
              {s}
            </option>
          ))}
        </select>
      </div>

      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
        {FIELDS.map((field) => (
          <div key={field.key}>
            <label className="mb-1 block text-sm font-medium text-slate-700">{field.label}</label>
            <input
              type="number"
              step={field.step}
              value={values[field.key]}
              onChange={(e) =>
                setValues((prev) => ({ ...prev, [field.key]: Number(e.target.value) }))
              }
              className="w-full rounded-md border border-slate-300 px-3 py-2 text-sm shadow-sm"
            />
          </div>
        ))}
      </div>

      <button
        onClick={submit}
        disabled={submitting}
        className="rounded-md bg-slate-900 px-5 py-2 text-sm font-medium text-white hover:bg-slate-700 disabled:opacity-50"
      >
        {submitting ? "Scoring…" : "Score this company"}
      </button>

      {error && <ErrorScreen error={error} />}

      {result && (
        <div className="space-y-4 rounded-lg border border-slate-200 p-5">
          <div>
            <p className="text-xs uppercase tracking-wide text-slate-500">
              Predicted acquisition-target probability
            </p>
            <p className="text-3xl font-semibold text-slate-900">
              {result.predicted_probability_pct.toFixed(1)}%
            </p>
          </div>

          {result.out_of_fold_probability_pct !== null && (
            <p className="text-xs text-slate-500">
              The Screener tab shows <strong>{result.out_of_fold_probability_pct.toFixed(1)}%</strong> for{" "}
              {result.out_of_fold_company} — that&apos;s an out-of-fold score (the model never saw its own
              label). This number comes from the final model fit on every company, including this one. A
              big gap between the two means this company&apos;s own data point meaningfully shifted the
              model.
            </p>
          )}

          {verdict && (
            <div className={`rounded-md border px-4 py-2 text-sm ${verdict.tone}`}>{verdict.text}</div>
          )}

          <div>
            <p className="mb-2 text-sm font-medium text-slate-700">
              Why this score — compared within {sector}
            </p>
            <div className="overflow-hidden rounded-lg border border-slate-200">
              <table className="w-full text-sm">
                <thead className="bg-slate-50 text-left text-xs uppercase tracking-wide text-slate-500">
                  <tr>
                    <th className="px-3 py-2">Field</th>
                    <th className="px-3 py-2">This company</th>
                    <th className="px-3 py-2">Typical {sector} target</th>
                    <th className="px-3 py-2">Typical independent {sector} company</th>
                  </tr>
                </thead>
                <tbody>
                  {result.comparison.map((row) => (
                    <tr key={row.field} className="border-t border-slate-100">
                      <td className="px-3 py-2 text-slate-600">{row.label}</td>
                      <td className="px-3 py-2 font-medium text-slate-900">{row.this_company}</td>
                      <td className="px-3 py-2 text-slate-600">{row.sector_target_median}</td>
                      <td className="px-3 py-2 text-slate-600">{row.sector_control_median}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            {result.thin_sector_data && (
              <p className="mt-2 text-xs text-slate-400">
                {sector} has only {result.sector_target_count} target(s) and {result.sector_control_count}{" "}
                control(s) in the dataset — this comparison is thin, treat it as indicative rather than
                robust.
              </p>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
