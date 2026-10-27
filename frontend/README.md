# Loan Default Prediction — React dashboard

A React + Vite dashboard for the FastAPI loan-default model. It shows the
saved model's real test metrics, accepts the model's 11-feature applicant
schema, calls `POST /predict`, and renders the returned probability and class.

The browser does **not** calculate predictions. FastAPI loads the saved
preprocessing pipeline and Random Forest model and returns the result.

## Stack

- React 19
- Vite 8
- JavaScript and hand-written CSS
- Recharts for the verified-metrics chart
- Vitest + Testing Library for component and API-client tests

## Prerequisites

- Node.js 20.19 or newer (Node 24 is used locally)
- The FastAPI backend and saved model artifacts in the project root
- The project virtual environment at `../.venv`

## Run locally

From the project root, start the API in one terminal:

```bash
cd ~/loan-default-prediction
.venv/bin/uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

In a second terminal, start the dashboard:

```bash
cd ~/loan-default-prediction/frontend
npm install
cp .env.example .env.local
npm run dev
```

Open <http://127.0.0.1:5173>.

The default API URL is `http://127.0.0.1:8000`. Override it in
`frontend/.env.local`:

```dotenv
VITE_API_BASE_URL=http://127.0.0.1:8000
```

The backend CORS configuration allows the Vite development origins
`http://localhost:5173` and `http://127.0.0.1:5173`. The Vite server uses
`strictPort: true`, so it fails instead of silently moving to port 5174 and
breaking CORS. Use `npm run dev` for API-backed development; `npm run preview`
uses port 4173 and is not part of the backend's CORS allow-list.

## API contract used by the dashboard

| Method | Path          | Purpose                                      |
| ------ | ------------- | -------------------------------------------- |
| GET    | `/health`     | API/model readiness; `503` means degraded    |
| GET    | `/model-info` | Model name, threshold, and actual test metrics |
| POST   | `/predict`    | Return class, probability, and threshold     |

The prediction response is consumed as returned by the backend:

```json
{
  "predicted_class": 0,
  "predicted_label": "no default",
  "default_probability": 0.0156875,
  "threshold": 0.62,
  "model_name": "Random Forest"
}
```

The model feature contract is mirrored in
`src/constants/model.js`:

- Required: `person_age`, `person_income`, `loan_amnt`,
  `loan_percent_income`, `cb_person_cred_hist_length`, `loan_grade`,
  `person_home_ownership`, `loan_intent`, `cb_person_default_on_file`
- Optional/imputed: `person_emp_length`, `loan_int_rate`
- Never accepted: `loan_status` (the target)

Numeric limits mirror the backend's application sanity checks. Client-side
validation is only a UX layer; FastAPI/Pydantic and the prediction service
remain the source of truth, and their errors are displayed in the UI.

## Scripts

```bash
npm run dev       # Vite development server on 5173
npm run build     # production build in dist/
npm run preview   # preview the production build on 4173
npm run test      # Vitest watch mode
npm run test:run  # one-shot test run
```

## Structure

```text
frontend/
├── src/
│   ├── App.jsx                  # dashboard state, API loading, submission
│   ├── main.jsx
│   ├── components/              # header, form, result, metrics, alerts
│   ├── constants/               # real feature schema and metric metadata
│   ├── services/api.js          # reusable API client + typed error shape
│   ├── styles/                  # global and dashboard CSS
│   ├── test/setup.js
│   └── utils/                   # validation, formatting, error helpers
├── .env.example
├── index.html
├── package.json
└── vite.config.js
```

## Behaviour and design notes

- Metrics are loaded from `GET /model-info`; no accuracy, precision, recall, or
  ROC-AUC value is hardcoded in the UI.
- The probability and threshold are displayed as returned. The browser does not
  recalculate the class.
- The form disables submission while the API is unavailable or a request is in
  flight, preventing duplicate submissions.
- Blank optional fields are omitted from the JSON payload so the fitted
  imputer handles them as intended.
- Validation errors from FastAPI are mapped back to the matching form fields
  when Pydantic supplies a `loc` path.
- The dashboard includes explicit limitations and responsible-use guidance.
  It is an educational project, not a lending decision system.

## Tests

```bash
npm run test:run
```

At the latest verified checkpoint (Stage 9), the frontend suite had **4 test
files and 30 passing tests**, covering the API client, schema/validation
helpers, applicant form, and prediction result states. The production build
completed successfully: 253.94 kB main JS (79.18 kB gzip), 357.42 kB lazy chart
chunk (103.56 kB gzip), and 24.62 kB CSS.

## Related docs

- Backend run and endpoint notes: [`../app/README.md`](../app/README.md)
- Model evaluation report: [`../reports/model_report.md`](../reports/model_report.md)
