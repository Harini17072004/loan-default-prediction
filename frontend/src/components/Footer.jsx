import { ArrowUpRightIcon } from "./Icons";

export default function Footer({ apiBaseUrl }) {
  return (
    <footer className="footer">
      <div className="footer__brand">
        <strong>Loan Default Prediction</strong>
        <span>Educational credit-risk dashboard</span>
      </div>

      <div className="footer__meta">
        <span>
          API <code>{apiBaseUrl}</code>
        </span>
        <a href={`${apiBaseUrl}/docs`} target="_blank" rel="noreferrer">
          OpenAPI docs
          <ArrowUpRightIcon size={14} />
        </a>
        <a
          href="https://www.kaggle.com/datasets/laotse/credit-risk-dataset"
          target="_blank"
          rel="noreferrer"
        >
          Dataset source
          <ArrowUpRightIcon size={14} />
        </a>
      </div>

      <p className="footer__note">
        Built with React, Vite, Recharts, FastAPI, and scikit-learn. Results are
        estimates for demonstration; they are not financial advice.
      </p>
    </footer>
  );
}
