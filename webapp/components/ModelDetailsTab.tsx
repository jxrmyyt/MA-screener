"use client";

import { useEffect, useState } from "react";
import {
  ScreenerApiError,
  fetchFeatureImportance,
  fetchModels,
  type FeatureImportanceResponse,
  type Meta,
  type ModelsResponse,
} from "@/lib/api";
import { ErrorScreen } from "@/components/ErrorScreen";
import { FeatureImportanceChart } from "@/components/FeatureImportanceChart";

const DISPLAY_NAMES: Record<string, string> = {
  logistic_regression: "Logistic Regression",
  decision_tree: "Decision Tree",
  svm_rbf: "SVM (RBF)",
  random_forest: "Random Forest",
};

export function ModelDetailsTab({ meta }: { meta: Meta }) {
  const [models, setModels] = useState<ModelsResponse | null>(null);
  const [importance, setImportance] = useState<FeatureImportanceResponse | null>(null);
  const [error, setError] = useState<ScreenerApiError | null>(null);

  useEffect(() => {
    Promise.all([fetchModels(), fetchFeatureImportance()])
      .then(([m, i]) => {
        setModels(m);
        setImportance(i);
      })
      .catch((err) => setError(err instanceof ScreenerApiError ? err : new ScreenerApiError("unknown_error", String(err))));
  }, []);

  if (error) return <ErrorScreen error={error} />;
  if (!models || !importance) return <p className="text-sm text-slate-400">Loading…</p>;

  return (
    <div className="space-y-8">
      <div>
        <h3 className="mb-2 text-sm font-semibold text-slate-900">
          Model comparison (Leave-One-Out Cross-Validation, N={models.n_companies})
        </h3>
        <div className="overflow-hidden rounded-lg border border-slate-200">
          <table className="w-full text-sm">
            <thead className="bg-slate-50 text-left text-xs uppercase tracking-wide text-slate-500">
              <tr>
                <th className="px-3 py-2">Model</th>
                <th className="px-3 py-2">Accuracy</th>
                <th className="px-3 py-2">Precision</th>
                <th className="px-3 py-2">Recall</th>
                <th className="px-3 py-2">F1</th>
                <th className="px-3 py-2">ROC-AUC</th>
              </tr>
            </thead>
            <tbody>
              {Object.entries(models.metrics).map(([key, m]) => (
                <tr
                  key={key}
                  className={`border-t border-slate-100 ${key === models.best_model ? "bg-blue-50" : ""}`}
                >
                  <td className="px-3 py-2 font-medium text-slate-900">
                    {DISPLAY_NAMES[key] ?? key}
                    {key === models.best_model && (
                      <span className="ml-2 rounded-full bg-blue-100 px-2 py-0.5 text-xs text-blue-700">
                        best by ROC-AUC
                      </span>
                    )}
                  </td>
                  <td className="px-3 py-2 text-slate-600">{m.accuracy}</td>
                  <td className="px-3 py-2 text-slate-600">{m.precision}</td>
                  <td className="px-3 py-2 text-slate-600">{m.recall}</td>
                  <td className="px-3 py-2 text-slate-600">{m.f1}</td>
                  <td className="px-3 py-2 text-slate-600">{m.roc_auc}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="mt-2 text-xs text-slate-500">
          {DISPLAY_NAMES[models.best_model] ?? models.best_model} is selected as the production model
          because it has the best ROC-AUC — the right metric for a ranked screener, since it rewards
          correctly ordering every company rather than just getting the 0.5-cutoff call right on each one.
        </p>
      </div>

      <div>
        <h3 className="mb-2 text-sm font-semibold text-slate-900">
          Feature importance ({DISPLAY_NAMES[importance.model] ?? importance.model})
        </h3>
        <FeatureImportanceChart importance={importance.importance} />
      </div>

      <div>
        <h3 className="mb-2 text-sm font-semibold text-slate-900">Companies by sector</h3>
        <div className="overflow-hidden rounded-lg border border-slate-200">
          <table className="w-full text-sm">
            <thead className="bg-slate-50 text-left text-xs uppercase tracking-wide text-slate-500">
              <tr>
                <th className="px-3 py-2">Sector</th>
                <th className="px-3 py-2">Independent (control)</th>
                <th className="px-3 py-2">Acquired (target)</th>
              </tr>
            </thead>
            <tbody>
              {meta.sector_breakdown.map((row) => (
                <tr key={row.sector} className="border-t border-slate-100">
                  <td className="px-3 py-2 font-medium text-slate-900">{row.sector}</td>
                  <td className="px-3 py-2 text-slate-600">{row.controls}</td>
                  <td className="px-3 py-2 text-slate-600">{row.targets}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
