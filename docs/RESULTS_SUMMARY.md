# Results summary

This note summarises selected machine-readable outputs committed in the repository. It is not a substitute for the manuscript.

## Baseline versus governance failure

For the repeated 2% invalid-`Contract` scenario, mean paired ROC-AUC changes across 30 seeds were:

- Logistic Regression: -0.000036
- Random Forest: -0.000500
- XGBoost: 0.000057

The governance engine classified this scenario as `BLOCK` across the repeated evaluation, showing that a machine-executable governance failure can occur with little or no deterioration in ROC-AUC.

## Fairness robustness

Under 25% controlled positive-label flipping, the selection-rate ratio remained materially below the prototype threshold across models while mean ROC-AUC decreases were small. Decision frequencies are stored in `outputs/robustness/robustness_decision_frequencies.csv`.

## Data minimisation

Mean ROC-AUC retained by the seven-feature configuration was:

- Logistic Regression: 99.43%
- Random Forest: 97.90%
- XGBoost: 99.75%

These results support the distinction between predictive utility and governance evidence while also showing that the size of the minimisation trade-off is model-dependent.
