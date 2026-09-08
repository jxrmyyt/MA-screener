"use client";

import { useEffect, useState } from "react";
import {
  ScreenerApiError,
  fetchScreener,
  type ScreenerResponse,
} from "@/lib/api";
import { ErrorScreen } from "@/components/ErrorScreen";

const DISPLAY_NAMES: Record<string, string> = {
  logistic_regression: "Logistic Regression",
  decision_tree: "Decision Tree",
  svm_rbf: "SVM (RBF)",
  random_forest: "Random Forest",
};

const MODEL_OPTIONS = Object.keys(DISPLAY_NAMES);

export function ScreenerTab({ sectors }: { sectors: string[] }) {
  const [model, setModel] = useState<string | undefined>(undefined);
  const [sector, setSector] = useState<string | undefined>(undefined);
  const [data, setData] = useState<ScreenerResponse | null>(null);
  const [error, setError] = useState<ScreenerApiError | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    fetchScreener(model, sector)
      .then((res) => {
        if (!cancelled) {
          setData(res);
          setError(null);
        }
      })
      .catch((err) => {
        if (!cancelled) {
          setError(err instanceof ScreenerApiError ? err : new ScreenerApiError("unknown_error", String(err)));
        }
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [model, sector]);

  function downloadCsv() {
    if (!data) return;
    const header = "rank,company,sector,predicted target probability\n";
    const rows = data.rows
      .map((r) => `${r.rank},${r.company},${r.sector},${r.predicted_probability_pct}`)
      .join("\n");
    const blob = new Blob([header + rows], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `ma_screener_${data.model}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  }

  return (
    <div>
      <p className="mb-4 text-sm text-slate-500">
        Companies below are all currently independent. Scores are out-of-fold leave-one-out
        predictions, so no company&apos;s own outcome leaked into its score.
      </p>

      <div className="mb-4 flex flex-wrap gap-3">
        <select
          value={model ?? ""}
          onChange={(e) => setModel(e.target.value || undefined)}
          className="rounded-md border border-slate-300 px-3 py-2 text-sm shadow-sm"
        >
          <option value="">Best model by ROC-AUC</option>
          {MODEL_OPTIONS.map((m) => (
            <option key={m} value={m}>
              {DISPLAY_NAMES[m]}
            </option>
          ))}
        </select>

        <select
          value={sector ?? ""}
          onChange={(e) => setSector(e.target.value || undefined)}
          className="rounded-md border border-slate-300 px-3 py-2 text-sm shadow-sm"
        >
          <option value="">All sectors</option>
          {sectors.map((s) => (
            <option key={s} value={s}>
              {s}
            </option>
          ))}
        </select>

        {data && (
          <button
            onClick={downloadCsv}
            className="ml-auto rounded-md border border-slate-300 px-4 py-2 text-sm font-medium text-slate-700 hover:bg-slate-50"
          >
            Download CSV
          </button>
        )}
      </div>

      {error && <ErrorScreen error={error} />}

      {!error && loading && <p className="text-sm text-slate-400">Loading…</p>}

      {!error && !loading && data && (
        <div className="overflow-hidden rounded-lg border border-slate-200">
          <table className="w-full text-sm">
            <thead className="bg-slate-50 text-left text-xs uppercase tracking-wide text-slate-500">
              <tr>
                <th className="px-4 py-2">Rank</th>
                <th className="px-4 py-2">Company</th>
                <th className="px-4 py-2">Sector</th>
                <th className="px-4 py-2">Predicted target probability</th>
              </tr>
            </thead>
            <tbody>
              {data.rows.map((row) => (
                <tr key={row.company} className="border-t border-slate-100">
                  <td className="px-4 py-2 text-slate-400">{row.rank}</td>
                  <td className="px-4 py-2 font-medium text-slate-900">{row.company}</td>
                  <td className="px-4 py-2 text-slate-600">{row.sector}</td>
                  <td className="px-4 py-2">
                    <div className="flex items-center gap-2">
                      <div className="h-2 w-32 overflow-hidden rounded-full bg-slate-100">
                        <div
                          className="h-full rounded-full bg-blue-600"
                          style={{ width: `${row.predicted_probability_pct}%` }}
                        />
                      </div>
                      <span className="text-slate-700">{row.predicted_probability_pct.toFixed(1)}%</span>
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
