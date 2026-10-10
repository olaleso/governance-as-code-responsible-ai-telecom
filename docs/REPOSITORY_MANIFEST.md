# Repository manifest

## Executable pipeline

- `download_data.py` — public source acquisition.
- `run_baseline.py` — deterministic baseline models and audit.
- `run_data_quality_experiments.py` — executable data-quality checks under controlled corruption.
- `run_governance_model_impact.py` — governance status versus predictive performance.
- `run_fairness_experiments.py` — controlled representation/label-bias scenarios.
- `run_privacy_minimisation.py` — staged feature minimisation.
- `run_explainability.py` — SHAP-based attribution analysis.
- `build_publication_package.py` — publication-oriented tables, figures and draft result text.
- `run_robustness_validation.py` — 30-seed paired robustness analysis.

## Machine-readable controls

- `governance/governance_rules.yaml`
- `governance/fairness_rules.yaml`
- `governance/privacy_rules.yaml`

## Primary result artefacts

- `outputs/metrics/`
- `outputs/explainability/`
- `outputs/publication/tables/`
- `outputs/publication/figures/`
- `outputs/robustness/`

## Archival metadata still to finalise

Before a Zenodo release, the authors should finalise:

- author order and family/given-name parsing;
- ORCID identifiers;
- corresponding author details;
- journal citation or preprint citation;
- final repository version number; and
- DOI metadata.

A `CITATION.cff` and `.zenodo.json` should be added only after these metadata have been confirmed.
