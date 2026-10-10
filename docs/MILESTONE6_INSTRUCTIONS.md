# Milestone 6 — Explainability

This milestone compares the full XGBoost model with the 7-feature minimised XGBoost model using SHAP.

## 1. Install the additional packages

From the existing activated virtual environment:

```powershell
python -m pip install -r requirements_m6.txt
```

## 2. Run the experiment

```powershell
python run_explainability.py
```

## Outputs

The script writes files under:

`outputs/explainability/`

Key outputs include:

- `explainability_model_comparison.csv`
- `X0_full_features_shap_raw_features.csv`
- `X1_minimum_candidate_set_shap_raw_features.csv`
- `shared_feature_explanation_shift.csv`
- local-explanation CSVs
- two SHAP importance PNG figures
- `explainability_experiment_summary.json`

## Purpose

This experiment tests whether aggressive data minimisation can preserve predictive utility while reducing exposure to sensitive/proxy features and producing a more compact explanation structure.

The SHAP values are model-explanation measures, not causal effects.
