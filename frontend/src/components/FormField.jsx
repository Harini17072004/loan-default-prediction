export default function FormField({
  field,
  value,
  error,
  onChange,
  disabled = false,
}) {
  const id = `field-${field.name}`;
  const helpId = `${id}-help`;
  const errorId = `${id}-error`;
  const describedBy =
    [field.helper ? helpId : null, error ? errorId : null]
      .filter(Boolean)
      .join(" ") || undefined;

  const unitLabel = field.unit === "years" ? "yrs" : field.unit;

  function handleChange(event) {
    onChange(field.name, event.target.value);
  }

  return (
    <div className={`field${error ? " field--error" : ""}`}>
      <div className="field__label-row">
        <label className="field__label" htmlFor={id}>
          {field.label}
        </label>
        {field.optional ? (
          <span className="field__optional">Optional</span>
        ) : null}
      </div>

      {field.type === "select" ? (
        <select
          id={id}
          name={field.name}
          className="field__control field__control--select"
          value={value}
          onChange={handleChange}
          required={!field.optional}
          disabled={disabled}
          aria-invalid={Boolean(error)}
          aria-describedby={describedBy}
        >
          <option value="">{field.placeholder}</option>
          {field.options.map((option) => (
            <option key={option.value} value={option.value}>
              {option.label}
            </option>
          ))}
        </select>
      ) : (
        <div
          className={`field__input-wrap${
            unitLabel ? " field__input-wrap--unit" : ""
          }`}
        >
          <input
            id={id}
            name={field.name}
            className="field__control"
            type="number"
            inputMode={field.inputMode}
            value={value}
            onChange={handleChange}
            placeholder={field.placeholder}
            min={field.min}
            max={field.max}
            step={field.step}
            required={!field.optional}
            disabled={disabled}
            autoComplete="off"
            aria-invalid={Boolean(error)}
            aria-describedby={describedBy}
          />
          {unitLabel ? (
            <span className="field__unit" aria-hidden="true">
              {unitLabel}
            </span>
          ) : null}
        </div>
      )}

      {field.helper ? (
        <p className="field__help" id={helpId}>
          {field.helper}
        </p>
      ) : null}
      {error ? (
        <p className="field__error" id={errorId} role="alert">
          {error}
        </p>
      ) : null}
    </div>
  );
}
