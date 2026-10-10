# Governance-as-Code for Responsible AI in Telecommunications

Reproducibility repository for the study:

**Governance-as-Code for Responsible AI in Telecommunications: Design, Experimental Evaluation, and a Nigerian Regulatory Crosswalk**

This repository contains the executable research artefact, machine-readable governance controls, controlled experiment scripts, publication tables/figures, and repeated-seed robustness outputs used in the associated journal manuscript.

## Research contribution

The study evaluates whether selected data-governance and responsible-AI requirements can be operationalised as machine-executable controls alongside conventional machine-learning evaluation. The core engineering mapping is:

**governance requirement -> measurable property -> executable rule -> automated evidence -> PASS / REVIEW / BLOCK decision -> remediation target**

The experimental testbed is a telecommunications customer-churn workflow. The scientific contribution is the governance mechanism and its evaluation, not a new churn-prediction algorithm.

## Important scope statement

The experimental dataset is IBM's public **Telco Customer Churn** sample, which represents a fictional telecommunications company. It is used as a controlled technical testbed and must **not** be represented as Nigerian subscriber data.

Nigeria-specific relevance is introduced through the regulatory and governance crosswalk, including the Nigeria Data Protection Act 2023, NDPC implementation guidance, the National Artificial Intelligence Strategy, telecommunications-sector context, and international AI/data-governance standards. The executable controls provide technical evidence; they do **not** constitute legal-compliance certification.

## Experimental design

Three classifiers are evaluated:

- Logistic Regression
- Random Forest
- XGBoost

The study includes:

1. deterministic baseline modelling;
2. executable data-quality rules and controlled degradation scenarios;
3. governance-versus-predictive-performance experiments;
4. controlled fairness/bias experiments;
5. privacy-oriented feature minimisation;
6. SHAP-based explainability analysis;
7. publication-table and figure generation; and
8. 30-seed paired robustness validation covering 450 model fits.

## Dataset provenance

Expected file:

```text
data/raw/Telco-Customer-Churn.csv
```

Recorded SHA-256:

```text
16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91
```

The raw CSV is intentionally excluded from version control.

## Tested software environment

Python 3.12. Exact package versions are recorded in `requirements-tested.txt`.

## Reproducing the study

```bash
python download_data.py
python run_baseline.py
python run_data_quality_experiments.py
python run_governance_model_impact.py
python run_fairness_experiments.py
python run_privacy_minimisation.py
python run_explainability.py
python build_publication_package.py
python run_robustness_validation.py
```

The robustness script performs **450 model fits**.

## Governance-rule interpretation

All rule thresholds in this repository are **experimental research controls**. They are not asserted to be Nigerian legal thresholds, universally valid fairness cut-offs, or a substitute for organisational/legal review.

## Repeated-seed robustness headline results

Across 30 paired random seeds:

- the 2% invalid-contract scenario remained a governance **BLOCK** across all models/seeds while ROC-AUC changed by approximately zero for Logistic Regression and XGBoost;
- the 25% controlled label-flip scenario produced fairness **BLOCK** outcomes in all Logistic Regression and Random Forest runs and 29/30 XGBoost runs while mean ROC-AUC declines remained below 0.006; and
- the seven-feature XGBoost model retained approximately **99.75%** of baseline ROC-AUC on average.

These results support the distinction between **predictive fitness** and **governance fitness**.

## Licence

Code in this repository is released under the MIT License. The IBM source dataset is not redistributed here and remains subject to its source terms.
