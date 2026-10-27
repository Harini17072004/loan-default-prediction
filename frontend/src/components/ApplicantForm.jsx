import { ALL_FIELDS, FORM_SECTIONS } from "../constants/model";
import { getErrorMessage } from "../utils/validation";
import FormField from "./FormField";
import { InfoIcon, ShieldCheckIcon } from "./Icons";

export default function ApplicantForm({
  form,
  errors,
  onChange,
  onSubmit,
  onReset,
  onUseExample,
  isSubmitting,
  disabledReason,
  submitError,
}) {
  const hasFieldErrors = Object.keys(errors).length > 0;
  const submitDisabled = isSubmitting || Boolean(disabledReason);

  return (
    <div className="panel form-panel">
      <div className="panel__header form-panel__header">
        <div>
          <span className="panel__eyebrow">Applicant assessment</span>
          <h2>Applicant details</h2>
          <p>
            Fill in the same features the saved model expects. Two values are
            optional because the training pipeline imputes missing entries.
          </p>
        </div>
        <span className="panel__badge">{ALL_FIELDS.length} features</span>
      </div>

      <form
        className="applicant-form"
        onSubmit={onSubmit}
        onReset={onReset}
        noValidate
        aria-busy={isSubmitting}
      >
        {FORM_SECTIONS.map((section) => (
          <fieldset className="form-section" key={section.id}>
            <legend className="form-section__legend">
              <span>{section.title}</span>
              <small>{section.description}</small>
            </legend>
            <div className="form-grid">
              {section.fields.map((field) => (
                <FormField
                  key={field.name}
                  field={field}
                  value={form[field.name]}
                  error={errors[field.name]}
                  onChange={onChange}
                  disabled={isSubmitting}
                />
              ))}
            </div>
          </fieldset>
        ))}

        <div className="form-footer">
          {submitError && !hasFieldErrors ? (
            <div className="form-error" role="alert">
              <strong>Request failed.</strong> {getErrorMessage(submitError)}
            </div>
          ) : null}

          {disabledReason ? (
            <p className="service-note" role="status">
              <InfoIcon size={16} />
              {disabledReason}
            </p>
          ) : null}

          <div className="form-actions">
            <button
              type="submit"
              className="button button--primary"
              disabled={submitDisabled}
            >
              {isSubmitting ? (
                <>
                  <span className="spinner" aria-hidden="true" />
                  Scoring applicant…
                </>
              ) : (
                <>
                  <ShieldCheckIcon size={18} />
                  Estimate default risk
                </>
              )}
            </button>

            <button
              type="button"
              className="button button--secondary"
              onClick={onReset}
              disabled={isSubmitting}
            >
              Reset form
            </button>

            {onUseExample ? (
              <button
                type="button"
                className="button button--link"
                onClick={onUseExample}
                disabled={isSubmitting}
                title="Fill the form with the backend's documented example applicant"
              >
                Load sample applicant
              </button>
            ) : null}
          </div>

          <p className="form-note">
            The browser sends these values to FastAPI. The saved preprocessing
            pipeline and Random Forest model produce the result; nothing is
            scored locally.
          </p>
        </div>
      </form>
    </div>
  );
}
