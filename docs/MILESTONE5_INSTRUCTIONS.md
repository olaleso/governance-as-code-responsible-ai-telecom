# Milestone 5 — Privacy and Data Minimisation

Copy the ZIP contents into your existing `rtagc_paper2_starter` folder.

Run:

```powershell
python run_privacy_minimisation.py
```

Scenarios:
- P0: all non-ID features
- P1: remove gender and SeniorCitizen
- P2: also remove Partner, Dependents and PaymentMethod
- P3: reduced operational/service feature set
- P4: minimum candidate feature set

The goal is to quantify privacy/data-minimisation gains against predictive utility loss.

Send back:
- `outputs/metrics/privacy_minimisation_results.csv`
- console output
