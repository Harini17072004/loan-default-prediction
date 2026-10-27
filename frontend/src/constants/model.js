/**
 * Frontend form schema that mirrors the saved model contract.
 *
 * The FastAPI service remains the source of truth. These definitions are used
 * for friendly client-side validation and to build the request payload. Numeric
 * bounds match `src/predict.py` (application sanity checks), while the category
 * values match the fitted preprocessing pipeline.
 */

export const FORM_SECTIONS = [
  {
    id: "applicant",
    title: "Applicant profile",
    description: "Personal and financial context for the application.",
    fields: [
      {
        name: "person_age",
        label: "Age",
        type: "number",
        min: 18,
        max: 100,
        step: 1,
        inputMode: "numeric",
        placeholder: "e.g. 35",
        helper: "Applicant age in years (18–100).",
        unit: "years",
      },
      {
        name: "person_income",
        label: "Annual income",
        type: "number",
        min: 1000,
        max: 10000000,
        step: 1000,
        inputMode: "numeric",
        placeholder: "e.g. 100000",
        helper: "Gross annual income in the dataset's currency units.",
      },
      {
        name: "person_emp_length",
        label: "Employment length",
        type: "number",
        min: 0,
        max: 60,
        step: 0.5,
        inputMode: "decimal",
        placeholder: "Optional",
        helper: "Years employed. Leave blank if unknown; the model imputes it.",
        optional: true,
        unit: "years",
      },
      {
        name: "person_home_ownership",
        label: "Home ownership",
        type: "select",
        placeholder: "Select home ownership",
        helper: "Housing status supplied with the application.",
        options: [
          { value: "MORTGAGE", label: "Mortgage" },
          { value: "OWN", label: "Own" },
          { value: "RENT", label: "Rent" },
          { value: "OTHER", label: "Other" },
        ],
      },
    ],
  },
  {
    id: "loan",
    title: "Loan details",
    description: "Terms and purpose of the requested loan.",
    fields: [
      {
        name: "loan_amnt",
        label: "Loan amount",
        type: "number",
        min: 100,
        max: 1000000,
        step: 100,
        inputMode: "numeric",
        placeholder: "e.g. 10000",
        helper: "Requested principal in the dataset's currency units.",
      },
      {
        name: "loan_int_rate",
        label: "Interest rate",
        type: "number",
        min: 0,
        max: 100,
        step: 0.01,
        inputMode: "decimal",
        placeholder: "Optional",
        helper: "Annual interest rate in percent. Leave blank if unknown.",
        optional: true,
        unit: "%",
      },
      {
        name: "loan_percent_income",
        label: "Loan-to-income ratio",
        type: "number",
        min: 0,
        max: 1,
        step: 0.01,
        inputMode: "decimal",
        placeholder: "e.g. 0.10",
        helper: "Loan amount divided by annual income (0–1).",
      },
      {
        name: "loan_grade",
        label: "Loan grade",
        type: "select",
        placeholder: "Select loan grade",
        helper: "Lender-assigned grade from A (lowest risk) to G (highest).",
        options: [
          { value: "A", label: "A — Lowest risk" },
          { value: "B", label: "B — Low risk" },
          { value: "C", label: "C — Moderate risk" },
          { value: "D", label: "D — Elevated risk" },
          { value: "E", label: "E — High risk" },
          { value: "F", label: "F — Very high risk" },
          { value: "G", label: "G — Highest risk" },
        ],
      },
      {
        name: "loan_intent",
        label: "Loan purpose",
        type: "select",
        placeholder: "Select loan purpose",
        helper: "Stated reason for the loan.",
        options: [
          { value: "DEBTCONSOLIDATION", label: "Debt consolidation" },
          { value: "EDUCATION", label: "Education" },
          { value: "HOMEIMPROVEMENT", label: "Home improvement" },
          { value: "MEDICAL", label: "Medical" },
          { value: "PERSONAL", label: "Personal" },
          { value: "VENTURE", label: "Venture" },
        ],
      },
    ],
  },
  {
    id: "credit",
    title: "Credit history",
    description: "Signals from the applicant's credit profile.",
    fields: [
      {
        name: "cb_person_cred_hist_length",
        label: "Credit history length",
        type: "number",
        min: 0,
        max: 80,
        step: 1,
        inputMode: "numeric",
        placeholder: "e.g. 10",
        helper: "Length of credit history in years (0–80).",
        unit: "years",
      },
      {
        name: "cb_person_default_on_file",
        label: "Prior default on file",
        type: "select",
        placeholder: "Select prior default status",
        helper: "Whether a prior default is recorded on file.",
        options: [
          { value: "N", label: "No" },
          { value: "Y", label: "Yes" },
        ],
      },
    ],
  },
];

export const ALL_FIELDS = FORM_SECTIONS.flatMap((section) => section.fields);

export const FIELD_BY_NAME = Object.fromEntries(
  ALL_FIELDS.map((field) => [field.name, field]),
);

export const OPTIONAL_FIELDS = new Set(
  ALL_FIELDS.filter((field) => field.optional).map((field) => field.name),
);

export const REQUIRED_FIELDS = ALL_FIELDS.filter((field) => !field.optional).map(
  (field) => field.name,
);

/** Empty controlled-form state. All values are strings so blanks stay blank. */
export const EMPTY_FORM = Object.fromEntries(
  ALL_FIELDS.map((field) => [field.name, ""]),
);

/** The backend's documented low-risk example applicant. */
export const EXAMPLE_FORM = {
  person_age: "35",
  person_income: "100000",
  person_emp_length: "10",
  loan_amnt: "10000",
  loan_int_rate: "7",
  loan_percent_income: "0.10",
  cb_person_cred_hist_length: "10",
  loan_grade: "A",
  person_home_ownership: "MORTGAGE",
  loan_intent: "PERSONAL",
  cb_person_default_on_file: "N",
};
