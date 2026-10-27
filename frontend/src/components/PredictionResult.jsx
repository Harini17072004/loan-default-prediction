import { formatPercent } from "../utils/format";
import { getErrorMessage, getErrorTitle } from "../utils/validation";
import ErrorAlert from "./ErrorAlert";
import { InfoIcon, ShieldCheckIcon } from "./Icons";

function EmptyResult({ serviceAvailable, serviceLoading, serviceMessage }) {
  const checking = serviceLoading;
  const available = serviceAvailable && !checking;

  return (
    <div className="result-empty">
      <span className="result-empty__icon" aria-hidden="true">
        <ShieldCheckIcon size={28} />
      </span>
      <h3>
        {checking || available
          ? "Awaiting applicant"
          : "Prediction service unavailable"}
      </h3>
      <p>
        {checking
          ? serviceMessage || "Checking the prediction service…"
          : available
            ? "Complete the applicant form and submit it to see the model's default probability and class decision."
            : serviceMessage ||
              "The backend model is not ready. Start FastAPI and refresh the dashboard."}
      </p>
      <div className="result-empty__steps" aria-label="Prediction flow">
        <span>1. Enter features</span>
        <span>2. Call FastAPI</span>
        <span>3. Read result</span>
      </div>
    </div>
  );
}

function LoadingResult() {
  return (
    <div className="result-loading" role="status" aria-live="polite">
      <span className="spinner spinner--large" aria-hidden="true" />
      <h3>Scoring applicant…</h3>
      <p>The backend is running the saved preprocessing pipeline and model.</p>
    </div>
  );
}

function ResultDetails({ prediction }) {
  const isDefault = prediction.predicted_class === 1;
  const probability = prediction.default_probability;
  const threshold = prediction.threshold;
  const probabilityPercent = Number.isFinite(probability)
    ? Math.min(100, Math.max(0, probability * 100))
    : 0;
  const thresholdPercent = Number.isFinite(threshold)
    ? Math.min(100, Math.max(0, threshold * 100))
    : 0;
  const tone = isDefault ? "risk" : "safe";

  return (
    <div className={`result-details result-details--${tone}`}>
      <div className="result-summary">
        <div className="result-summary__topline">
          <span className="result-summary__label">Predicted class</span>
          <span className={`result-badge result-badge--${tone}`}>
            {isDefault ? "Higher risk" : "Lower risk"}
          </span>
        </div>
        <h3 className="result-summary__title">
          {isDefault ? "Default predicted" : "No default predicted"}
        </h3>
        <p className="result-summary__copy">
          {isDefault
            ? "The estimated probability is at or above the saved decision threshold."
            : "The estimated probability is below the saved decision threshold."}
        </p>

        <div className="result-probability">
          <strong>{formatPercent(probability)}</strong>
          <span>estimated default probability</span>
        </div>
      </div>

      <div className="risk-meter-block">
        <div
          className="risk-meter"
          role="progressbar"
          aria-label="Estimated default probability"
          aria-valuemin={0}
          aria-valuemax={100}
          aria-valuenow={Number(probabilityPercent.toFixed(2))}
          aria-valuetext={formatPercent(probability)}
        >
          <span
            className={`risk-meter__fill risk-meter__fill--${tone}`}
            style={{ width: `${probabilityPercent}%` }}
          />
          <span
            className="risk-meter__threshold"
            style={{ left: `${thresholdPercent}%` }}
            title={`Decision threshold: ${formatPercent(threshold)}`}
            aria-hidden="true"
          />
        </div>
        <div className="risk-meter__labels">
          <span>0%</span>
          <span>Threshold {formatPercent(threshold)}</span>
          <span>100%</span>
        </div>
      </div>

      <dl className="result-meta">
        <div>
          <dt>Class</dt>
          <dd>{prediction.predicted_class}</dd>
        </div>
        <div>
          <dt>Threshold</dt>
          <dd>{formatPercent(threshold)}</dd>
        </div>
        <div>
          <dt>Model</dt>
          <dd>{prediction.model_name || "—"}</dd>
        </div>
      </dl>

      <div className="result-disclaimer">
        <InfoIcon size={18} />
        <p>
          <strong>This is an estimate, not a guarantee.</strong> A model score
          should support, not replace, human credit review, policy checks, and
          fair-lending oversight.
        </p>
      </div>
    </div>
  );
}

export default function PredictionResult({
  prediction,
  isSubmitting,
  error,
  serviceAvailable,
  serviceLoading = false,
  serviceMessage,
}) {
  return (
    <aside
      className="panel result-panel"
      aria-labelledby="result-title"
      aria-live="polite"
      aria-busy={isSubmitting}
    >
      <div className="panel__header result-panel__header">
        <div>
          <span className="panel__eyebrow">Model output</span>
          <h2 id="result-title">Prediction result</h2>
        </div>
        {prediction ? (
          <span
            className={`result-panel__mark result-panel__mark--${
              prediction.predicted_class === 1 ? "risk" : "safe"
            }`}
            aria-hidden="true"
          />
        ) : null}
      </div>

      {isSubmitting ? (
        <LoadingResult />
      ) : error ? (
        <ErrorAlert
          title={getErrorTitle(error)}
          message={getErrorMessage(error)}
        />
      ) : prediction ? (
        <ResultDetails prediction={prediction} />
      ) : (
        <EmptyResult
          serviceAvailable={serviceAvailable}
          serviceLoading={serviceLoading}
          serviceMessage={serviceMessage}
        />
      )}
    </aside>
  );
}
