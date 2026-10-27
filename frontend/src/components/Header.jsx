import ApiStatus from "./ApiStatus";
import { ArrowUpRightIcon, LogoMark, RefreshIcon } from "./Icons";
import { formatPercent } from "../utils/format";

export default function Header({
  health,
  isRefreshing,
  onRefresh,
  apiBaseUrl,
  modelName,
  threshold,
}) {
  return (
    <header className="hero">
      <div className="hero__topbar">
        <div className="brand">
          <LogoMark />
          <div className="brand__copy">
            <span className="eyebrow">Credit risk intelligence</span>
            <span className="brand__name">Loan Default Prediction</span>
          </div>
        </div>

        <div className="hero__tools">
          <ApiStatus
            loading={health.loading}
            data={health.data}
            error={health.error}
          />
          <button
            type="button"
            className="button button--ghost button--small"
            onClick={onRefresh}
            disabled={isRefreshing}
          >
            <RefreshIcon size={16} />
            {isRefreshing ? "Refreshing…" : "Refresh data"}
          </button>
          <a
            className="button button--ghost button--small"
            href={`${apiBaseUrl}/docs`}
            target="_blank"
            rel="noreferrer"
          >
            API docs
            <ArrowUpRightIcon size={16} />
          </a>
        </div>
      </div>

      <div className="hero__content">
        <div className="hero__copy">
          <span className="hero__kicker">Machine learning portfolio project</span>
          <h1>
            Loan Default Prediction
            <span>Using Machine Learning</span>
          </h1>
          <p className="hero__description">
            Explore how a trained credit-risk model estimates the probability
            that a loan applicant defaults. The dashboard sends applicant
            details to the FastAPI backend and displays the model's real
            probability, decision threshold, and saved test metrics.
          </p>
          <div className="hero__badges" aria-label="Model summary">
            <span className="hero__badge">
              {modelName || "Loading model information"}
            </span>
            <span className="hero__badge">
              Threshold{" "}
              {Number.isFinite(threshold) ? formatPercent(threshold) : "—"}
            </span>
            <span className="hero__badge">FastAPI + React</span>
          </div>
        </div>

        <aside className="hero__signal" aria-label="Prediction approach">
          <span className="hero__signal-label">Model signal</span>
          <strong>Default risk</strong>
          <p>
            The browser never computes a prediction. It submits the applicant
            profile to the saved pipeline and renders the API response.
          </p>
          <div className="signal-bars" aria-hidden="true">
            <span />
            <span />
            <span />
            <span />
            <span />
            <span />
            <span />
            <span />
          </div>
          <span className="hero__signal-footnote">
            Educational estimate · Not a lending decision
          </span>
        </aside>
      </div>
    </header>
  );
}
