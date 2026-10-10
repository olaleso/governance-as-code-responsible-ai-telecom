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

Key robustness scenarios include a clean baseline, 20% missingness, 2% invalid `Contract` values, 25% controlled positive-label flipping, and a seven-feature minimised model.

## Repository structure

```text
.
├── config/                     # experiment configuration
├── data/                       # data provenance notes; raw data not committed
├── docs/                       # methods, provenance, governance and milestone notes
├── governance/                 # YAML governance rule specifications
├── manuscript/                 # manuscript-status note; draft binary not tracked pre-submission
├── outputs/
│   ├── explainability/         # SHAP comparison outputs
│   ├── metrics/                # baseline and experiment metrics
│   ├── publication/            # tables, figures and draft result text
│   └── robustness/             # repeated-seed results and confidence intervals
├── src/                        # reusable data/governance modules
├── build_publication_package.py
├── download_data.py
├── run_baseline.py
├── run_data_quality_experiments.py
├── run_governance_model_impact.py
├── run_fairness_experiments.py
├── run_privacy_minimisation.py
├── run_explainability.py
└── run_robustness_validation.py
```

## Dataset provenance

Expected file:

```text
data/raw/Telco-Customer-Churn.csv
```

The study used the IBM archived sample with recorded SHA-256:

```text
16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91
```

The raw CSV is intentionally excluded from version control. Use `download_data.py` to obtain the public source and verify the local file before running experiments.

## Tested software environment

The research environment used Python 3.12. Exact package versions are recorded in `requirements-tested.txt`. A less restrictive installation specification is provided in `requirements.txt`.

## Quick start

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python download_data.py
python run_baseline.py
```

### Linux/macOS

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python download_data.py
python run_baseline.py
```

## Reproducing the full experimental sequence

Run the scripts in this order:

```bash
python run_baseline.py
python run_data_quality_experiments.py
python run_governance_model_impact.py
python run_fairness_experiments.py
python run_privacy_minimisation.py
python run_explainability.py
python build_publication_package.py
python run_robustness_validation.py
```

The robustness script performs **450 model fits**, so it will take substantially longer than the earlier stages.

## Governance-rule interpretation

All rule thresholds in this repository are **experimental research controls**. They are not asserted to be Nigerian legal thresholds, universally valid fairness cut-offs, or a substitute for organisational/legal review.

Examples include:

- model-feature missingness `<= 0.02`;
- duplicate customer identifier rate `== 0`;
- invalid categorical/numeric rate `== 0`;
- symmetric selection-rate ratio `>= 0.80`;
- absolute equal-opportunity difference `<= 0.10`; and
- absolute false-positive-rate difference `<= 0.10`.

See `docs/GOVERNANCE_RULES.md` and `docs/REGULATORY_CROSSWALK.csv` for interpretation boundaries.

## Repeated-seed robustness headline results

Across 30 paired random seeds:

- the 2% invalid-contract scenario remained a governance **BLOCK** across all models/seeds while ROC-AUC changed by approximately zero for Logistic Regression and XGBoost;
- the 25% controlled label-flip scenario produced fairness **BLOCK** outcomes in all Logistic Regression and Random Forest runs and 29/30 XGBoost runs while mean ROC-AUC declines remained below 0.006; and
- the seven-feature XGBoost model retained approximately **99.75%** of baseline ROC-AUC on average.

These results support the central distinction between **predictive fitness** and **governance fitness**.

## Reproducibility notes

See:

- `docs/REPRODUCIBILITY.md`
- `docs/DATA_PROVENANCE.md`
- `docs/GOVERNANCE_RULES.md`
- `docs/REGULATORY_CROSSWALK.csv`
- `outputs/robustness/robustness_metadata.json`

## Manuscript and archival release

The manuscript binary is deliberately not tracked in the public-facing reproducibility tree during pre-submission development. Once the author list, journal policy, and citation metadata are finalised, the repository can be tagged for archival release and connected to Zenodo for a DOI.

## Licence

Code in this repository is released under the MIT License. The IBM source dataset is not redistributed here and remains subject to its source terms.
