export default function MetricCard({
  label,
  value,
  helper,
  tone = "slate",
  loading = false,
  emphasis = false,
  title,
}) {
  return (
    <article
      className={`metric-card metric-card--${tone}${
        emphasis ? " metric-card--emphasis" : ""
      }`}
      aria-busy={loading}
    >
      <span className="metric-card__label">{label}</span>
      {loading ? (
        <span className="skeleton skeleton--metric" aria-hidden="true" />
      ) : (
        <strong className="metric-card__value" title={title || value}>
          {value}
        </strong>
      )}
      <span className="metric-card__helper">{helper}</span>
    </article>
  );
}
