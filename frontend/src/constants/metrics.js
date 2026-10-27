/** Display metadata for metrics returned by `GET /model-info`. */

export const METRIC_CARDS = [
  {
    key: "accuracy",
    label: "Accuracy",
    helper: "Share of test cases classified correctly.",
    tone: "teal",
  },
  {
    key: "precision",
    label: "Precision",
    helper: "Of predicted defaults, the share that actually defaulted.",
    tone: "blue",
  },
  {
    key: "recall",
    label: "Recall",
    helper: "Of actual defaults, the share the model caught.",
    tone: "amber",
  },
  {
    key: "roc_auc",
    label: "ROC-AUC",
    helper: "How well the model ranks defaults above non-defaults.",
    tone: "indigo",
  },
];

export const CHART_METRICS = [
  { key: "accuracy", label: "Accuracy", color: "#0f766e" },
  { key: "precision", label: "Precision", color: "#0284c7" },
  { key: "recall", label: "Recall", color: "#d97706" },
  { key: "f1", label: "F1 score", color: "#7c3aed" },
  { key: "roc_auc", label: "ROC-AUC", color: "#4f46e5" },
  { key: "pr_auc", label: "PR-AUC", color: "#0d9488" },
];
