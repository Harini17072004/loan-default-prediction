export default function ApiStatus({ loading, data, error }) {
  let tone = "checking";
  let label = "Checking API";
  let detail = "Contacting the FastAPI prediction service.";

  if (!loading && error) {
    tone = "offline";
    label = "API offline";
    detail = error.message || "The prediction API could not be reached.";
  } else if (!loading && data?.model_ready === true) {
    tone = "ready";
    label = "Model ready";
    detail = "The saved model and metadata are loaded.";
  } else if (!loading && data) {
    tone = "degraded";
    label = "Model unavailable";
    detail =
      data.detail ||
      "The API is running, but the saved model artifacts are unavailable.";
  }

  return (
    <span
      className={`api-status api-status--${tone}`}
      role="status"
      aria-live="polite"
      title={detail}
    >
      <span className="api-status__dot" aria-hidden="true" />
      <span>{label}</span>
    </span>
  );
}
