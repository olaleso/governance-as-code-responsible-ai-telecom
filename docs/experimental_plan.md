# Experimental plan — Paper 2

## Baseline
Train Logistic Regression, Random Forest and XGBoost using a fixed stratified 70/15/15 split.

## E1 — Data quality degradation
Inject controlled:
- missingness: 5%, 10%, 20%;
- duplicate rows/IDs;
- invalid categorical values;
- negative/implausible numeric values.

Measure model performance and governance-rule detection.

## E2 — Representation/bias experiments
Construct controlled group imbalance and assess:
- demographic parity;
- disparate impact;
- equal opportunity;
- false-positive-rate differences.

## E3 — Data minimisation
Compare:
- full feature set;
- direct identifiers excluded;
- sensitive attributes excluded;
- reduced feature set.

Measure privacy-risk reduction vs predictive utility.

## E4 — Explainability
Use SHAP in the next milestone to assess:
- global feature importance;
- local explanations;
- sensitive/proxy feature influence.

## Primary thesis principle
Predictive performance and governance fitness are separate evaluation dimensions.
