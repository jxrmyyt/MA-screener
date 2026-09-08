import { ScreenerApiError } from "@/lib/api";

const COPY: Record<string, { title: string; body: string }> = {
  pipeline_not_trained: {
    title: "Model isn't trained yet",
    body: "The backend can't find outputs/ from the training pipeline. Run prepare_data.py and train_models.py from the project root, then reload.",
  },
  unknown_sector: {
    title: "Sector not recognized",
    body: "That sector isn't part of the trained dataset — pick one from the dropdown.",
  },
  unknown_model: {
    title: "Model not recognized",
    body: "That model wasn't one of the four trained during evaluation.",
  },
  company_not_found: {
    title: "No such company on file",
    body: "That name isn't in the training dataset — try loading one from the dropdown instead.",
  },
};

export function ErrorScreen({ error }: { error: ScreenerApiError }) {
  const copy = COPY[error.errorCode] ?? {
    title: "Something broke on our end",
    body: error.message,
  };

  return (
    <div className="flex flex-col items-center justify-center rounded-lg border border-red-200 bg-red-50 px-8 py-14 text-center">
      <div className="mb-3 flex h-12 w-12 items-center justify-center rounded-full bg-red-100 text-red-600">
        <WarningIcon />
      </div>
      <h3 className="text-lg font-semibold text-slate-900">{copy.title}</h3>
      <p className="mt-1 max-w-sm text-sm text-slate-500">{copy.body}</p>
    </div>
  );
}

function WarningIcon() {
  return (
    <svg viewBox="0 0 24 24" fill="none" className="h-6 w-6" stroke="currentColor" strokeWidth={2}>
      <path
        strokeLinecap="round"
        strokeLinejoin="round"
        d="M12 9v4m0 4h.01M10.29 3.86l-8.18 14.18A1 1 0 003 19.5h18a1 1 0 00.86-1.46L13.7 3.86a1 1 0 00-1.72 0z"
      />
    </svg>
  );
}
