import { AlertTriangleIcon, RefreshIcon } from "./Icons";

export default function ErrorAlert({
  title = "Something went wrong",
  message,
  onRetry,
  retryLabel = "Try again",
}) {
  return (
    <div className="alert" role="alert">
      <AlertTriangleIcon className="alert__icon" size={22} />
      <div className="alert__body">
        <strong>{title}</strong>
        {message ? <p>{message}</p> : null}
      </div>
      {onRetry ? (
        <button
          type="button"
          className="button button--alert button--small"
          onClick={onRetry}
        >
          <RefreshIcon size={15} />
          {retryLabel}
        </button>
      ) : null}
    </div>
  );
}
