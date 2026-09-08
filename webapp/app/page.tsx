"use client";

import { useEffect, useState } from "react";
import { ScreenerApiError, fetchMeta, type Meta } from "@/lib/api";
import { ErrorScreen } from "@/components/ErrorScreen";
import { ScreenerTab } from "@/components/ScreenerTab";
import { ScoreTab } from "@/components/ScoreTab";
import { ModelDetailsTab } from "@/components/ModelDetailsTab";

const TABS = ["Screener", "Score a company", "Model details"] as const;
type Tab = (typeof TABS)[number];

export default function Home() {
  const [meta, setMeta] = useState<Meta | null>(null);
  const [error, setError] = useState<ScreenerApiError | null>(null);
  const [tab, setTab] = useState<Tab>("Screener");
  const [showLimitations, setShowLimitations] = useState(false);

  useEffect(() => {
    fetchMeta()
      .then(setMeta)
      .catch((err) => setError(err instanceof ScreenerApiError ? err : new ScreenerApiError("unknown_error", String(err))));
  }, []);

  return (
    <main className="mx-auto max-w-5xl px-6 py-10">
      <header className="mb-6">
        <h1 className="text-2xl font-semibold text-slate-900">Cross-Industry M&amp;A Target Screener</h1>
        {meta && (
          <p className="mt-2 text-sm text-slate-500">
            A classifier trained on {meta.n_targets} real, completed acquisitions and {meta.n_controls}{" "}
            comparable companies that remained independent, spanning {meta.sectors.length} sectors (
            {meta.sectors.join(", ")}). Every financial ratio is compared against its own sector&apos;s
            baseline before scoring, so a software company&apos;s R&amp;D spending isn&apos;t judged
            against an industrial company&apos;s norms.
          </p>
        )}
      </header>

      {error && <ErrorScreen error={error} />}

      {meta && (
        <>
          <button
            onClick={() => setShowLimitations((v) => !v)}
            className="mb-6 flex w-full items-center justify-between rounded-md border border-amber-200 bg-amber-50 px-4 py-2 text-left text-sm font-medium text-amber-800"
          >
            <span>⚠ Known limitations (read before trusting any number below)</span>
            <span>{showLimitations ? "−" : "+"}</span>
          </button>
          {showLimitations && (
            <div className="-mt-4 mb-6 space-y-2 rounded-b-md border border-t-0 border-amber-200 bg-amber-50/50 px-4 py-4 text-sm text-slate-700">
              <p>
                <strong>Modest sample ({meta.n_companies} companies across {meta.sectors.length} sectors).</strong>{" "}
                LOOCV squeezes maximum signal out of a limited labeled set, but metrics still carry real
                uncertainty. Some sectors (e.g. Telecom) are too small to have their own normalization
                baseline and fall back to the whole-dataset baseline instead.
              </p>
              <p>
                <strong>Point-in-time mismatch.</strong> Target financials are the last full fiscal year
                before each deal&apos;s announcement; control financials are each company&apos;s most
                recent fiscal year today.
              </p>
              <p>
                <strong>Cross-industry comparability.</strong> Ratios are converted to sector-relative
                z-scores to avoid the model learning &quot;high R&amp;D% = software company,&quot; but this
                only controls for sector median and spread.
              </p>
              <p>
                <strong>Missing data imputed</strong> — sector-z values default to 0 (sector-typical) when
                the raw figure wasn&apos;t disclosed.
              </p>
              <p>
                <strong>This is a pattern-match, not a forecast.</strong> A high score means
                &quot;financially resembles past acquisition targets in its sector,&quot; not &quot;will be
                acquired.&quot;
              </p>
            </div>
          )}

          <div className="mb-6 border-b border-slate-200">
            <nav className="flex gap-6">
              {TABS.map((t) => (
                <button
                  key={t}
                  onClick={() => setTab(t)}
                  className={`border-b-2 px-1 py-3 text-sm font-medium transition ${
                    tab === t
                      ? "border-slate-900 text-slate-900"
                      : "border-transparent text-slate-400 hover:text-slate-600"
                  }`}
                >
                  {t}
                </button>
              ))}
            </nav>
          </div>

          {tab === "Screener" && <ScreenerTab sectors={meta.sectors} />}
          {tab === "Score a company" && <ScoreTab sectors={meta.sectors} />}
          {tab === "Model details" && <ModelDetailsTab meta={meta} />}
        </>
      )}
    </main>
  );
}
