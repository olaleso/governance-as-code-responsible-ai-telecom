# Milestone 7 — Publication Consolidation

Copy this ZIP's contents into the root of your existing `rtagc_paper2_starter` project.

Run:

```powershell
python build_publication_package.py
```

No additional packages should be required beyond the environment already used for Milestone 6.

The script expects these experiment outputs to exist:

- `outputs/metrics/baseline_model_metrics.csv`
- `outputs/metrics/dq_model_impact_results.csv`
- `outputs/metrics/fairness_scenario_results.csv`
- `outputs/metrics/privacy_minimisation_results.csv`
- `outputs/explainability/explainability_model_comparison.csv`

It creates:

`outputs/publication/tables/`
- five publication-oriented CSV tables

`outputs/publication/figures/`
- five publication-oriented PNG figures

`outputs/publication/text/`
- `Methods_Draft.md`
- `Results_Draft.md`
- `Limitations_and_Scope.md`

After running, send back:
- `outputs/publication/text/Results_Draft.md`
- `outputs/publication/tables/Table2_Data_Quality_Governance_Impact.csv`
- `outputs/publication/tables/Table3_Fairness_Experiment_Results.csv`
- `outputs/publication/tables/Table4_Privacy_Minimisation_Results.csv`
- `outputs/publication/tables/Table5_Explainability_Comparison.csv`

This milestone freezes the current first-paper experimental scope. Drift, lineage and continuous monitoring remain reserved for subsequent PhD studies unless later evidence shows they are necessary for this paper.
