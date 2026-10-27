# Interview Preparation — Loan Default Prediction

This guide is written for the actual implementation in this repository. The
numbers below come from `models/model_metadata.json` and
`reports/model_metrics.json`; the code references match the files in the
repository. It is a study aid, not a claim that every answer is the only correct
way to build a credit-risk system.

## 1. One-minute project explanation

> I built an end-to-end loan-default prediction project using a public Kaggle
> credit-risk dataset. I cleaned the data, explored class imbalance and data
> quality, and built a leakage-safe scikit-learn pipeline with median imputation,
> missing-value indicators, log transforms, scaling, ordinal encoding for
> `loan_grade`, and one-hot encoding for nominal features. I compared Logistic
> Regression, Decision Tree, and Random Forest with and without balanced class
> weights, tuned them with stratified 5-fold cross-validation on the training
> split, and selected a Random Forest. I chose a 0.62 decision threshold from
> out-of-fold training predictions to maximize F1, then evaluated once on the
> untouched test set. The saved model reaches 93.58% accuracy, 97.53% precision,
> 72.50% recall, 83.17% F1, and 93.70% ROC-AUC. I served the saved pipeline
> through FastAPI with Pydantic validation and a reusable inference service, and
> built a React/Vite dashboard that loads real metadata and sends every
> prediction request to the backend. The project includes automated Python and
> frontend tests and explicit responsible-use limitations.

## 2. What I actually implemented and contributed

The repository contains work across the full lifecycle:

- **Data analysis:** profiled 32,581 rows, identified 165 duplicate rows,
  inspected missing `person_emp_length` and `loan_int_rate`, checked impossible
  ages/employment values, examined class imbalance, and saved EDA findings and
  figures.
- **Preprocessing:** implemented `src/preprocessing.py` with reusable cleaning,
  a stratified split, and an unfitted `ColumnTransformer` pipeline so all learned
  statistics are fitted on training data only.
- **Model development:** implemented `train.py` to compare six baseline variants,
  tune three algorithm families, select by CV ROC-AUC, choose a threshold from
  out-of-fold probabilities, refit on the training split, and evaluate once on
  the test split.
- **Artifact contract:** saved the complete pipeline with joblib and wrote
  `model_metadata.json` containing feature order, threshold, metrics, dataset
  checksums, and library versions.
- **Inference service:** implemented `src/predict.py` to load the artifacts once,
  validate numeric ranges and categories, reject target leakage and unknown
  fields, call `predict_proba`, and convert the probability to a class at the
  saved threshold.
- **Backend:** implemented a FastAPI app with `/health`, `/model-info`, and
  `/predict`; Pydantic schemas; startup lifespan loading; dependencies; CORS; and
  safe `422`/`500`/`503` error handling.
- **Frontend:** implemented a React 19/Vite dashboard with real metrics loaded
  from `/model-info`, an 11-feature applicant form, client-side validation,
  loading/error/empty/result states, responsive CSS, a lazy-loaded Recharts
  comparison chart, and an API client.
- **Testing and documentation:** wrote Python tests for preprocessing, inference,
  API contracts, CORS, model-unavailable behavior, and prediction; frontend tests
  for validation, API behavior, form behavior, and result states; EDA/model
  reports; setup instructions; and this interview guide.

## 3. Verified numbers

| Item | Value |
| --- | --- |
| Dataset rows | 32,581 raw; 32,416 after duplicate removal |
| Split | 25,932 train / 6,484 test, stratified 80/20 |
| Default rate | About 21.87% in train and test |
| Selected model | Random Forest, balanced class weights |
| Trees | 200 |
| `max_depth` | 20 |
| `min_samples_leaf` | 1 |
| `max_features` | 0.5 (50% of features per split) |
| Threshold | 0.62 |
| Test accuracy | 93.58% |
| Test precision | 97.53% |
| Test recall | 72.50% |
| Test F1 | 83.17% |
| Test ROC-AUC | 93.70% |
| Test PR-AUC | 89.19% |
| TN / FP / FN / TP | 5,040 / 26 / 390 / 1,028 |

## 4. Architecture walkthrough

```text
React dashboard
   |
   | GET /health + GET /model-info on load
   | POST /predict on form submission
   v
FastAPI app (app/main.py)
   |-- Pydantic request model (app/schemas.py)
   |-- PredictionService dependency (src/predict.py)
   |-- joblib pipeline loaded once at startup
   v
ColumnTransformer + RandomForestClassifier
   |
   `--> default_probability
         predicted_class = int(default_probability >= 0.62)
```

At startup, FastAPI's lifespan handler calls the cached
`get_prediction_service()`. The service loads `loan_default_pipeline.joblib` and
`model_metadata.json`, verifies that the metadata feature order matches the
fitted preprocessor, and stores the bundle in application state. Requests reuse
that loaded service rather than loading the 10.75 MB artifact each time.

The dashboard never imports scikit-learn or computes a probability. It sends the
applicant JSON to FastAPI and renders the response. This keeps one authoritative
model implementation and avoids frontend/backend drift.

## 5. Key concepts used in this project

### React frontend

- React state holds the form, field errors, submission state, prediction, and
  dashboard health/model-info state.
- `useEffect` loads `/health` and `/model-info` with `Promise.allSettled`, so one
  request can fail without discarding the other.
- The API client wraps `fetch`, normalizes the base URL, parses JSON/text, and
  throws a typed `ApiError` with status, detail, field errors, and payload.
- Client validation is a UX layer. Pydantic and the inference service remain the
  source of truth.
- The Recharts chart is lazy-loaded with `React.lazy` and `Suspense`, producing a
  separate build chunk.
- The result panel shows empty, loading, error, and result states. It displays
  the backend's probability and threshold; it does not recompute the class.

### FastAPI backend

- `create_app()` is an app factory, which makes testing easier.
- The lifespan handler owns model loading and cleanup.
- `get_service()` returns the ready service or raises `503`.
- `get_optional_service()` lets `/predict` validate an invalid body as `422`
  before deciding whether a valid request should be `503`.
- Pydantic models define request/response shape, numeric bounds, optional fields,
  and extra-field rejection.
- Exception handlers translate known failures into safe JSON responses.
- CORS is restricted to the two local Vite origins.

### REST, Pydantic, CORS, and status codes

- `/health` is a readiness endpoint, not just a process-alive check.
- `/model-info` exposes real saved metadata, so the dashboard does not hardcode
  metrics.
- `/predict` is a stateless POST that accepts one applicant and returns one
  score.
- `200` means success, `422` means invalid input, `500` means an unexpected
  inference/server failure, and `503` means the model dependency is unavailable.
- CORS is a browser security rule. The API allows `http://localhost:5173` and
  `http://127.0.0.1:5173`; a different frontend origin needs an explicit entry.

### Scikit-learn pipeline

The preprocessing pipeline has four branches inside a `ColumnTransformer`:

1. **Base numeric:** median imputation with missing indicators, then
   `StandardScaler`.
2. **Log numeric:** median imputation, `log1p`, then scaling for
   `person_income` and `loan_amnt`.
3. **Ordinal:** most-frequent imputation, then ordinal encoding of
   `loan_grade` with categories `A`–`G`.
4. **Nominal:** most-frequent imputation, then one-hot encoding for home
   ownership, loan intent, and prior default on file.

`remainder="drop"` ensures the target and any unexpected column cannot become a
feature. The preprocessor and classifier are saved as one pipeline, so serving
cannot accidentally apply a different transformation.

### Random Forest

- A Random Forest is an ensemble of decision trees trained on bootstrap samples
  with random feature subsets at splits.
- Bagging and feature randomness reduce the variance of a single overfit tree.
- `n_estimators=200` controls the number of trees.
- `max_depth=20` limits tree depth.
- `min_samples_leaf=1` allows fine-grained leaves.
- `max_features=0.5` gives each split a random half of the features.
- `class_weight="balanced"` gives the minority class more weight during
  training.
- The forest exposes `predict_proba`, which this project uses before applying
  the threshold.
- Impurity-based feature importance is useful for a quick explanation but is not
  causal and can favor continuous/high-cardinality features.

### Class imbalance

The default class is about 21.9% of the data. A constant “no default” prediction
would already score about 78% accuracy, so accuracy alone is misleading. This
project uses:

- balanced class weights during training,
- precision/recall/F1 and PR-AUC for minority-class evaluation,
- ROC-AUC for ranking quality,
- an F1-maximizing threshold selected from out-of-fold training predictions.

SMOTE is not used. Class weighting changes the training loss; threshold
selection changes the operating point after training. Both address different
parts of the imbalance problem.

### Precision, recall, F1, and ROC-AUC

Using the positive class “default”:

- **Precision** = TP / (TP + FP): of applicants flagged as default, how many
  actually defaulted?
- **Recall** = TP / (TP + FN): of actual defaults, how many did we catch?
- **F1** = 2 × precision × recall / (precision + recall): harmonic mean.
- **ROC-AUC** measures ranking across all thresholds; 0.5 is random and 1.0 is
  perfect.
- **PR-AUC** is especially informative when the positive class is uncommon.
- **Accuracy** is the share of all predictions that are correct.

At threshold 0.62: precision 97.53% and recall 72.50%. That means the model is
very precise when it flags risk, but it misses about 27.5% of actual defaults.

### Threshold selection

`find_best_threshold()` sweeps thresholds from 0.01 to 0.99 in 197 steps,
evaluates precision/recall/F1 at each point, and chooses the highest F1. Ties go
to the higher threshold. The sweep is run on out-of-fold predictions for the
training split, not on the test set. The selected threshold is 0.62. The test set
is only used once afterward.

### Leakage prevention

- Duplicate rows are removed before the split.
- The split is stratified and performed before fitting preprocessing.
- Every learned statistic — medians, most-frequent categories, scaler means and
  standard deviations, encoders — lives inside the pipeline and is fitted on the
  training split only.
- Test data is not used for model selection, threshold selection, or calibration.
- `loan_status` is explicitly rejected by the inference service and Pydantic.
- Metadata feature order must match the fitted preprocessor; the service refuses
  to guess.

## 6. Interview questions and answers

### Data and problem framing

**1. What problem does this project solve?**
It predicts a binary outcome for a loan applicant: `loan_status = 1` means
default and `0` means no default. The goal is to estimate risk probability and
demonstrate an end-to-end ML system, not to make real lending decisions.

**2. Why is this a classification problem rather than regression?**
The target is a discrete binary label. The model returns a probability for class
1, but the final output is a class decision at a chosen threshold.

**3. What is the class distribution and why does it matter?**
About 21.9% of rows are defaults, so the data is imbalanced. A majority-class
baseline would get roughly 78% accuracy while catching zero defaults. That is why
this project emphasizes recall, F1, PR-AUC, and threshold tuning.

**4. Why not use accuracy as the only metric?**
Accuracy treats false positives and false negatives equally and is dominated by
the majority class. In credit risk, missing a default can be much more expensive
than flagging a good applicant for review, so precision and recall matter.

**5. How did you clean the data?**
I removed 165 exact duplicates; replaced ages above 100 and impossible employment
lengths (above 60 years or longer than the applicant's age) with `NaN`; coerced
numeric fields; checked the binary target; and left genuine outliers in place.
The fitted pipeline handles the missing values.

**6. Why replace invalid values with `NaN` instead of deleting rows?**
The rest of the row can still contain useful information. Deleting the whole row
throws away valid features. The pipeline's imputer fills only the missing value
using training data, and the missing indicator tells the model that the value was
absent.

**7. Why remove duplicate rows before splitting?**
If duplicates land in both train and test, the test score can be inflated by
memorization. Removing exact duplicates reduces that leakage risk.

**8. Why use a stratified split?**
Stratification preserves the roughly 22% default rate in both training and test
sets. With an imbalanced target, a plain random split could produce noticeably
different class proportions.

**9. What data-quality issues did you find?**
There were 165 exact duplicates, a small number of impossible ages and employment
lengths, 895 missing employment lengths, and 3,116 missing interest rates. The
EDA report records these findings and the decisions taken.

**10. Did you remove income or loan-amount outliers?**
No. The analysis found genuinely large but plausible values. I retained them and
used `log1p` plus tree-based splits rather than deleting potentially valid
applicants.

### Preprocessing and pipeline design

**11. Why use `log1p` for income and loan amount?**
Both are strongly right-skewed. `log1p` compresses the long right tail while
handling zero values safely (`log1p(0) = 0`). This makes the numeric scale more
manageable for the model and prevents extreme values from dominating scaled
inputs.

**12. Why median imputation rather than mean imputation?**
The median is robust to skew and extreme values. Income and loan amounts have
heavy tails, so a mean can be pulled upward by a few large observations.

**13. Why add missing indicators?**
Missingness can carry information. For example, missing employment length had a
different default rate from present employment length. `add_indicator=True`
gives the model an explicit 0/1 signal in addition to the imputed value.

**14. Why scale numeric features if the final model is a Random Forest?**
Tree splits do not require scaling, but the same preprocessing pipeline was used
while comparing linear models and trees. Scaling is necessary for Logistic
Regression and harmless for the forest. Keeping one pipeline also reduces the
chance of training/serving mismatch.

**15. How are categorical features encoded?**
`loan_grade` is ordinal and encoded as A–G because the categories have a natural
risk order. Home ownership, loan intent, and prior-default-on-file are nominal and
one-hot encoded. One-hot encoding avoids implying a false ordering.

**16. What does `ColumnTransformer` do?**
It applies different preprocessing branches to different feature groups and
concatenates the results. Here it handles base numeric, log numeric, ordinal, and
nominal branches, then passes the encoded matrix to the classifier.

**17. Why save the whole pipeline instead of only the trained classifier?**
The pipeline contains the learned imputers, missing indicators, scalers, and
encoders. Saving only the classifier would force serving code to recreate and
refit those transformations, which risks leakage and train/serve skew. The
artifact is a single reproducible unit.

**18. How did you prevent preprocessing leakage?**
The split happens before any fitted transformer is applied. All learned
statistics are inside the pipeline and fitted only on `X_train`. Test rows are
transformed using the training-fitted pipeline.

**19. What happens if an unseen category reaches the pipeline?**
The ordinal encoder uses `unknown_value=-1` and the one-hot encoder uses
`handle_unknown="ignore"`, but the inference service rejects unknown category
values before calling the pipeline. The encoder settings are a safety net, not a
license to send arbitrary categories.

### Model selection and evaluation

**20. Why compare three algorithms?**
Logistic Regression provides an interpretable linear baseline; a Decision Tree
captures nonlinearity but can overfit; Random Forest adds bagging and feature
randomness to reduce variance. Comparing all three gives evidence for the final
choice instead of assuming the most complex model is best.

**21. What were the baseline results?**
On 5-fold training CV, Random Forest had the strongest ROC-AUC (about 0.930 for
the balanced baseline) and PR-AUC. The single Decision Tree and Logistic
Regression were materially lower. The tuned Random Forest reached mean CV ROC-AUC
of 0.9362.

**22. Why did Random Forest win?**
The data contains nonlinear interactions and mixed feature types. The forest
captured those interactions better than a linear model and was much more stable
than one unpruned tree. It also produced a strong minority-class ranking.

**23. What do the selected Random Forest hyperparameters mean?**
`n_estimators=200` builds 200 trees; `max_depth=20` limits each tree;  
`min_samples_leaf=1` allows very small leaves; `max_features=0.5` gives each
split a random 50% of features; and `class_weight="balanced"` increases the
minority-class weight.

**24. Why use 5-fold cross-validation?**
Five stratified folds give a reasonable bias/variance trade-off without making
training excessively slow. Each fold preserves the class ratio. The final model
is then refit on the full training split.

**25. What is out-of-fold prediction and why use it?**
For each fold, a model is trained on the other folds and predicts the held-out
fold. Every training row gets a prediction from a model that did not see it, so
the resulting probability vector is suitable for threshold selection without
touching the test set.

**26. Why choose 0.62 instead of 0.50?**
The default cutoff is arbitrary. A sweep over out-of-fold training probabilities
selected the threshold with the highest F1 for the default class. At 0.62 the
test precision increased to 97.53% and F1 to 83.17%, while recall fell to 72.50%
compared with the 0.50 cutoff. That is a deliberate precision/recall trade-off.

**27. Why not optimize the threshold on the test set?**
That would leak test information into model selection and produce an optimistically
biased estimate. The test set is reserved for one final evaluation after model and
threshold choices are frozen.

**28. How do precision and recall trade off in lending?**
Lowering the threshold flags more applicants as risky: recall rises, but more good
applicants are falsely flagged (lower precision). Raising it does the opposite.
The “right” point depends on the relative cost of a missed default versus a
rejected good applicant and on human review capacity.

**29. What do the actual confusion-matrix values mean?**
At threshold 0.62 there are 5,040 true negatives, 26 false positives, 390 false
negatives, and 1,028 true positives. The model catches 1,028 of 1,418 actual
defaults and misses 390.

**30. What is the difference between ROC-AUC and PR-AUC?**
ROC-AUC measures how well the model ranks positives above negatives across all
thresholds and is insensitive to class prevalence. PR-AUC focuses on the positive
class and is often more informative when that class is uncommon. This project
reports both.

**31. Why is `class_weight="balanced"` not enough by itself?**
Balanced weights change how the model is trained and can shift probabilities
toward the minority class. They do not choose the final operating threshold. This
project treats weighting and threshold selection as separate decisions.

**32. How does scikit-learn compute balanced weights?**
It gives each class a weight inversely proportional to its frequency, so the
minority class contributes more to the loss. The exact implementation is
`n_samples / (n_classes * class_count)` for each class.

### Serving and API design

**33. How does FastAPI make the model available to requests?**
The lifespan handler loads the service once at startup and stores it on
`app.state`. A dependency returns that service. The model is not reloaded for
every request.

**34. Why is `get_prediction_service()` cached?**
It is decorated with `lru_cache(maxsize=1)`, so the default artifact is loaded
once per process. This avoids repeated disk I/O and repeated artifact validation.

**35. What happens if the model is missing or corrupt?**
Startup catches the artifact error, logs it, and leaves the app in a degraded
state. `/health` returns `503` with `model_ready: false`; `/model-info` and a
valid `/predict` request also return `503`. The process does not crash during
startup.

**36. Why does `/predict` use an optional service dependency?**
It lets Pydantic validate the request first. An invalid body therefore returns
`422` even if the model is unavailable. A valid body with no model returns `503`.
This makes the error behavior more predictable.

**37. What is Pydantic doing in this project?**
It validates JSON types, required fields, numeric bounds, optional fields, and
extra fields. It also generates the OpenAPI schema shown in `/docs`. `extra="forbid"`
rejects unexpected keys, and validators reject booleans as numbers, blank optional
strings, employment longer than age, and the target column.

**38. Why reject `loan_status`?**
It is the target. Including it would be direct target leakage and would make the
demo invalid. The rejection exists in both the Pydantic model and the inference
service so it cannot be bypassed by using the service directly.

**39. Why have both client-side and server-side validation?**
Client validation improves the user experience and catches common mistakes
immediately. Server validation is the security and correctness boundary because a
browser can be bypassed. The backend remains the source of truth.

**40. What does CORS do and why is it configured narrowly?**
CORS tells the browser which origins may call the API. The app allows only
`http://localhost:5173` and `http://127.0.0.1:5173` for local development. A
different port or host must be added explicitly; `allow_credentials=False`
because this demo does not use cookies or authenticated sessions.

**41. What do HTTP 422, 500, and 503 mean here?**
`422` means the request failed validation, such as a missing field or invalid
category. `500` means an unexpected inference or server failure. `503` means the
model dependency is unavailable. The handlers keep internal details out of the
response and log them server-side.

**42. How does the frontend know the model is ready?**
It calls `/health` and checks `model_ready`. Only then does it enable form
submission. It loads model name, threshold, metrics, and features from
`/model-info`; these are not hardcoded in the UI.

**43. How does the frontend avoid duplicate submissions?**
`isSubmitting` guards the submit handler and the submit button state. The form
is disabled while a request is in flight.

**44. How are backend validation errors shown on the right fields?**
The API client keeps Pydantic's `errors` array. The frontend maps each error's
`loc` path to the matching field name and displays the message. The first invalid
field is focused and scrolled into view.

**45. What happens when the backend is unreachable?**
`fetch` throws, the API client wraps it in an `ApiError` with status `0`, and the
dashboard shows “API offline”, hides model metrics, and disables submission. A
refresh rechecks `/health` and `/model-info`.

**46. What is the stale-result limitation?**
The prediction is stored in React state. If the backend later becomes
unavailable, a previously successful prediction can remain visible; it is the last
real backend result, not a fresh one. The app does not poll health continuously,
so the issue becomes visible when the user refreshes. The proposed safe fix is to
clear the prediction on refresh or when health transitions to not-ready.

### Testing, reproducibility, and production

**47. What do the tests cover?**
The Python suite covers preprocessing, inference, API contracts, validation,
target/unknown-field rejection, optional fields, CORS, unavailable/corrupt model
behavior, and real low/high-risk predictions. The frontend suite covers validation
helpers, API-client behavior, the applicant form, and prediction-result states.
At the last verified checkpoint: 63 Python tests and 30 frontend tests passed.

**48. How is the project reproducible?**
Python and npm dependencies are pinned; the split uses `random_state=42`;
cross-validation is stratified and seeded; the model and metadata are saved
together; the dataset and artifacts have SHA-256 values; and the training report
records library versions. The dataset download helper verifies the expected
checksum.

**49. How do you know the served model matches its metadata?**
`load_model_bundle()` compares the metadata feature list with the fitted
preprocessor's `feature_names_in_`, checks that the model exposes
`predict_proba`, verifies the positive class exists, and checks that the
preprocessor's output feature count matches the model's `n_features_in_`.
Mismatches are rejected as artifact errors.

**50. What would you change for production?**
Add authentication, authorization, rate limiting, request logging with privacy
controls, a container image, pinned infrastructure, model monitoring, drift
detection, fairness testing, calibration review, a model registry, canary
deployment, and human-review workflows. The current app is intentionally a local
educational demo.

**51. How would you monitor drift?**
Track input distributions (income, loan amount, grade, intent), prediction
distribution, threshold-flag rate, and delayed ground-truth default rates by
segment. Set alerts for population shift and performance degradation, and
revalidate fairness and calibration on newer vintages.

**52. Why not use deep learning here?**
The dataset is tabular and modest in size, and a tree ensemble provides strong
performance with much less tuning and infrastructure. A deep model would add
complexity without evidence that it improves this task.

**53. What is the biggest weakness of the model?**
It is trained on simulated data, has no temporal validation, and has no fairness
audit. It also relies partly on lender-assigned fields. High test accuracy does
not establish that it would be safe or fair in a real lending environment.

**54. Why return both probability and class?**
The class is useful for a UI decision, while the probability preserves the
information needed for a different threshold or a human review policy. Returning
only a label would hide the model's uncertainty.

## 7. Honest limitations and trade-offs

- **Synthetic data:** the Kaggle dataset is simulated; results do not transfer
  automatically to a real loan book.
- **No temporal validation:** future loans may have different distributions and
  default behavior.
- **No fairness audit:** age, income, home ownership, and other fields can
  correlate with protected characteristics.
- **Lender-assigned fields:** grade and interest rate may encode existing policy
  decisions rather than independent risk.
- **Calibration is not guaranteed:** a probability is a model score, not an
  audited real-world default frequency.
- **Recall is 72.50%:** about 390 of 1,418 test defaults are missed at the chosen
  threshold.
- **Random Forest trade-off:** stronger nonlinear performance than logistic
  regression, but a larger artifact and less transparent decision logic.
- **Local-only API:** no authentication, rate limiting, privacy controls, or
  production process manager.
- **Stale frontend result:** a previous prediction can remain on screen after a
  backend outage; the UI does not continuously poll health.
- **Pickle/joblib trust:** loading untrusted model artifacts is unsafe; only load
  artifacts from a trusted source.
- **No project license file:** the dataset is CC0, but the code license has not
  been selected in the repository.

## 8. Proposed stale-result fix (not applied)

The minimal safe change is to clear the previous result as soon as a refresh
starts, so an unavailable backend cannot leave an old score looking current:

```diff
   const refreshDashboard = useCallback(() => {
+    setPrediction(null);
+    setSubmitError(null);
     setDashboardVersion((version) => version + 1);
   }, []);
```

A more robust version also clears the result when the health check transitions to
not-ready:

```diff
+  useEffect(() => {
+    if (!health.loading && !healthReady) {
+      setPrediction(null);
+      setSubmitError(null);
+    }
+  }, [health.loading, healthReady]);
```

This is deliberately not applied yet because it changes frontend behavior and
should be reviewed before merging. Continuous `/health` polling would detect a
server that dies while the page is open, but that adds requests and complexity;
the refresh-time fix is the smallest change for the current workflow.

## 9. Short closing answer

> The project demonstrates the full path from data quality and leakage-safe
> preprocessing to model selection, threshold tuning, artifact packaging, a
> validated FastAPI service, and a real React dashboard. The most important
> engineering decision was not the forest itself; it was keeping every learned
> transformation inside the saved pipeline and keeping all scoring on the
> backend. The most important honest caveat is that strong test metrics on
> simulated data do not make this a safe or fair real-world lending system.
