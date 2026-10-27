import { lazy, Suspense } from "react";
import { METRIC_CARDS } from "../constants/metrics";
import { formatDate, formatNumber, formatPercent } from "../utils/format";
import { getErrorMessage, getErrorTitle } from "../utils/validation";
import ErrorAlert from "./ErrorAlert";
import { ChartIcon } from "./Icons";
import MetricCard from "./MetricCard";

const PerformanceChart = lazy(() => import("./PerformanceChart"));

function ChartSkeleton() {
  return (
    <div className="chart-skeleton" aria-hidden="true">
      <span className="skeleton" />
      <span className="skeleton" />
      <span className="skeleton" />
      <span className="skeleton" />
      <span className="skeleton" />
    </div>
  );
}

function MetricSkeletons() {
  return (
    <>
      <MetricCard
        label="Model"
        value="—"
        helper="Loading saved model metadata…"
        tone="navy"
        loading
        emphasis
      />
      {METRIC_CARDS.map((metric) => (
        <MetricCard
          key={metric.key}
          label={metric.label}
          value="—"
          helper={metric.helper}
          tone={metric.tone}
          loading
        />
      ))}
    </>
  );
}

export default function MetricsOverview({
  modelInfo,
  loading,
  error,
  onRetry,
}) {
  const metrics = modelInfo?.metrics ?? {};
  const supportTotal =
    (Number.isFinite(metrics.support?.non_default)
      ? metrics.support.non_default
      : 0) +
    (Number.isFinite(metrics.support?.default) ? metrics.support.default : 0);

  return (
    <section className="metrics-section" aria-labelledby="metrics-title">
      <div className="section-heading">
        <div>
          <span className="section-heading__kicker">Saved model</span>
          <h2 id="metrics-title">Verified test performance</h2>
          <p>
            Metrics are loaded from the backend's <code>/model-info</code>{" "}
            endpoint. Nothing on this page is hardcoded.
          </p>
        </div>
        {!loading && !error ? (
          <span className="section-heading__pill">
            Decision threshold {formatPercent(modelInfo?.threshold)}
          </span>
        ) : null}
      </div>

      {error ? (
        <ErrorAlert
          title={getErrorTitle(error)}
          message={getErrorMessage(error)}
          onRetry={onRetry}
          retryLabel="Reload metrics"
        />
      ) : (
        <>
          <div className="metrics-grid" aria-busy={loading}>
            {loading ? (
              <MetricSkeletons />
            ) : (
              <>
                <MetricCard
                  label="Model"
                  value={modelInfo?.model_name || "—"}
                  helper={
                    modelInfo?.estimator_class || "Saved scikit-learn pipeline"
                  }
                  tone="navy"
                  emphasis
                />
                {METRIC_CARDS.map((metric) => (
                  <MetricCard
                    key={metric.key}
                    label={metric.label}
                    value={formatPercent(metrics?.[metric.key])}
                    helper={metric.helper}
                    tone={metric.tone}
                  />
                ))}
              </>
            )}
          </div>

          <div className="panel performance-panel">
            <div className="panel__header">
              <div>
                <span className="panel__eyebrow">
                  <ChartIcon size={15} />
                  Holdout evaluation
                </span>
                <h3>How the model scores</h3>
              </div>
              <div className="performance-panel__meta">
                {supportTotal > 0 ? (
                  <span>{formatNumber(supportTotal)} test rows</span>
                ) : null}
                {modelInfo?.trained_at ? (
                  <span>Trained {formatDate(modelInfo.trained_at)}</span>
                ) : null}
              </div>
            </div>

            {loading ? (
              <ChartSkeleton />
            ) : (
              <Suspense fallback={<ChartSkeleton />}>
                <PerformanceChart metrics={metrics} />
              </Suspense>
            )}

            <p className="chart-footnote">
              Higher is better. Accuracy alone can be misleading when one class
              is much more common, so precision, recall, and ranking metrics are
              shown alongside it.
            </p>
          </div>
        </>
      )}
    </section>
  );
}
