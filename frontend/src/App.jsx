import { useCallback, useEffect, useState } from "react";
import ApplicantForm from "./components/ApplicantForm";
import Footer from "./components/Footer";
import Header from "./components/Header";
import MetricsOverview from "./components/MetricsOverview";
import PredictionResult from "./components/PredictionResult";
import ResponsibleUse from "./components/ResponsibleUse";
import { EMPTY_FORM, EXAMPLE_FORM } from "./constants/model";
import {
  ApiError,
  API_BASE_URL,
  getHealth,
  getModelInfo,
  predict,
} from "./services/api";
import {
  firstErrorField,
  getErrorMessage,
  mapApiFieldErrors,
  validateForm,
} from "./utils/validation";

function focusField(name) {
  if (!name || typeof document === "undefined") return;
  window.setTimeout(() => {
    const element = document.getElementById(`field-${name}`);
    if (!element) return;
    element.focus();
    if (typeof element.scrollIntoView === "function") {
      element.scrollIntoView({ behavior: "smooth", block: "center" });
    }
  }, 0);
}

export default function App() {
  const [dashboardVersion, setDashboardVersion] = useState(0);
  const [health, setHealth] = useState({
    loading: true,
    data: null,
    error: null,
  });
  const [modelInfo, setModelInfo] = useState(null);
  const [modelInfoState, setModelInfoState] = useState({
    loading: true,
    error: null,
  });
  const [form, setForm] = useState(EMPTY_FORM);
  const [fieldErrors, setFieldErrors] = useState({});
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [prediction, setPrediction] = useState(null);
  const [submitError, setSubmitError] = useState(null);

  const refreshDashboard = useCallback(() => {
    setDashboardVersion((version) => version + 1);
  }, []);

  useEffect(() => {
    let active = true;
    setHealth({ loading: true, data: null, error: null });
    setModelInfoState({ loading: true, error: null });

    Promise.allSettled([getHealth(), getModelInfo()]).then(
      ([healthResult, infoResult]) => {
        if (!active) return;

        if (healthResult.status === "fulfilled") {
          setHealth({
            loading: false,
            data: healthResult.value,
            error: null,
          });
        } else {
          setHealth({
            loading: false,
            data: null,
            error: healthResult.reason,
          });
        }

        if (infoResult.status === "fulfilled") {
          setModelInfo(infoResult.value);
          setModelInfoState({ loading: false, error: null });
        } else {
          setModelInfo(null);
          setModelInfoState({ loading: false, error: infoResult.reason });
        }
      },
    );

    return () => {
      active = false;
    };
  }, [dashboardVersion]);

  const healthReady = health.data?.model_ready === true;
  const modelName =
    modelInfo?.model_name || health.data?.model_name || null;
  const threshold = modelInfo?.threshold;

  let disabledReason = null;
  if (health.loading) {
    disabledReason =
      "Checking the prediction service before enabling the form…";
  } else if (health.error) {
    disabledReason =
      "The prediction API is unreachable. Start FastAPI, then use Refresh data.";
  } else if (!healthReady) {
    disabledReason =
      health.data?.detail ||
      "The saved model is unavailable. Check the backend logs, then refresh.";
  }

  const serviceMessage = health.loading
    ? "Checking the prediction service…"
    : health.error
      ? getErrorMessage(health.error)
      : !healthReady
        ? health.data?.detail ||
          "The saved model is not available. Start the backend and refresh."
        : null;

  function clearSubmissionFeedback() {
    setPrediction(null);
    setSubmitError(null);
  }

  function handleChange(name, value) {
    setForm((previous) => ({ ...previous, [name]: value }));
    setFieldErrors((previous) => {
      if (!previous[name]) return previous;
      const next = { ...previous };
      delete next[name];
      return next;
    });
    clearSubmissionFeedback();
  }

  function handleReset() {
    if (isSubmitting) return;
    setForm(EMPTY_FORM);
    setFieldErrors({});
    clearSubmissionFeedback();
  }

  function handleUseExample() {
    if (isSubmitting) return;
    setForm({ ...EXAMPLE_FORM });
    setFieldErrors({});
    clearSubmissionFeedback();
  }

  async function handleSubmit(event) {
    event.preventDefault();
    if (isSubmitting) return;

    if (!healthReady) {
      const error = new ApiError(
        disabledReason || "The prediction service is not ready.",
        {
          status: 503,
          detail: disabledReason,
        },
      );
      setPrediction(null);
      setSubmitError(error);
      return;
    }

    const validation = validateForm(form);
    setFieldErrors(validation.errors);
    setSubmitError(null);

    if (!validation.valid) {
      setPrediction(null);
      focusField(firstErrorField(validation.errors));
      return;
    }

    setIsSubmitting(true);
    setPrediction(null);

    try {
      const result = await predict(validation.payload);
      setPrediction(result);
    } catch (error) {
      const apiError =
        error instanceof ApiError
          ? error
          : new ApiError(
              "The prediction request failed unexpectedly. Please try again.",
              {
                status: 0,
                detail: error instanceof Error ? error.message : "",
              },
            );

      const mappedErrors = mapApiFieldErrors(apiError.errors);
      if (Object.keys(mappedErrors).length > 0) {
        setFieldErrors(mappedErrors);
        focusField(firstErrorField(mappedErrors));
      }
      setSubmitError(apiError);
    } finally {
      setIsSubmitting(false);
    }
  }

  return (
    <div className="app-shell">
      <Header
        health={health}
        isRefreshing={health.loading || modelInfoState.loading}
        onRefresh={refreshDashboard}
        apiBaseUrl={API_BASE_URL}
        modelName={modelName}
        threshold={threshold}
      />

      <main className="dashboard">
        <MetricsOverview
          modelInfo={modelInfo}
          loading={modelInfoState.loading}
          error={modelInfoState.error}
          onRetry={refreshDashboard}
        />

        <section className="workspace" aria-labelledby="workspace-title">
          <div className="section-heading workspace__heading">
            <div>
              <span className="section-heading__kicker">
                Interactive assessment
              </span>
              <h2 id="workspace-title">Score an applicant</h2>
              <p>
                Enter the model's real feature contract, submit it to FastAPI,
                and inspect the returned probability and threshold decision.
              </p>
            </div>
            <span className="section-heading__pill">Backend-only scoring</span>
          </div>

          <div className="workspace__grid">
            <ApplicantForm
              form={form}
              errors={fieldErrors}
              onChange={handleChange}
              onSubmit={handleSubmit}
              onReset={handleReset}
              onUseExample={handleUseExample}
              isSubmitting={isSubmitting}
              disabledReason={disabledReason}
              submitError={submitError}
            />
            <PredictionResult
              prediction={prediction}
              isSubmitting={isSubmitting}
              error={submitError}
              serviceAvailable={healthReady}
              serviceLoading={health.loading}
              serviceMessage={serviceMessage}
            />
          </div>
        </section>

        <ResponsibleUse />
      </main>

      <Footer apiBaseUrl={API_BASE_URL} />
    </div>
  );
}
