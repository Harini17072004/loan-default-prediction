const percentFormatter = new Intl.NumberFormat("en-US", {
  style: "percent",
  minimumFractionDigits: 2,
  maximumFractionDigits: 2,
});

const numberFormatter = new Intl.NumberFormat("en-US", {
  maximumFractionDigits: 2,
});

const dateFormatter = new Intl.DateTimeFormat("en-GB", {
  day: "2-digit",
  month: "short",
  year: "numeric",
});

/** Format a 0–1 fraction as a percentage, e.g. 0.62 → "62.00%". */
export function formatPercent(value) {
  return Number.isFinite(value) ? percentFormatter.format(value) : "—";
}

/** Format a value already expressed on a 0–100 scale for chart display. */
export function formatPercentValue(value) {
  return Number.isFinite(value) ? `${value.toFixed(2)}%` : "—";
}

/** Format a count or other number with up to two decimal places. */
export function formatNumber(value) {
  return Number.isFinite(value) ? numberFormatter.format(value) : "—";
}

/** Format an ISO timestamp as a readable date. */
export function formatDate(value) {
  if (!value) return "—";
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? String(value) : dateFormatter.format(date);
}
