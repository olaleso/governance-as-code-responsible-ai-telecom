# Milestone 4 — Fairness and Controlled Bias Experiments

Copy the ZIP contents into the existing `rtagc_paper2_starter` project folder and replace files when prompted.

Run:

```powershell
python run_fairness_experiments.py
```

## Scenarios

- F0: clean baseline
- F1: retain 50% of female training records
- F2: retain 25% of female training records
- F3: flip 10% of positive churn labels among female training records
- F4: flip 25% of positive churn labels among female training records
- F5: flip 50% of positive churn labels among female training records

The test set remains clean and fixed across all scenarios.

## Metrics

- Selection-rate ratio
- Absolute demographic-parity difference
- Absolute equal-opportunity difference
- Absolute false-positive-rate difference
- Accuracy
- F1
- ROC-AUC

The gender scenarios are controlled technical experiments only. They are not claims about actual discriminatory practices in Nigerian or other telecom operators.

Send back:
- `outputs/metrics/fairness_scenario_results.csv`
- `outputs/metrics/fairness_subgroup_metrics.csv`
- console output
