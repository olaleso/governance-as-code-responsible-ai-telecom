# Milestone 8 — Repeated-Random-Seed Robustness Validation

This milestone tests whether the paper's central findings persist across **30 pre-specified random seeds**.

## Why this design

The full experimental suite is intentionally not repeated because that would add substantial computation without proportionate evidential value. Instead, the repeated-seed analysis targets the paper's four main claims:

1. clean baseline predictive performance;
2. a data-quality REVIEW scenario (20% missingness);
3. a data-quality BLOCK scenario (2% invalid contract values);
4. the key fairness BLOCK scenario (25% positive-label flipping);
5. the seven-feature privacy/data-minimisation scenario.

All three models are rerun for every seed.

The analysis is **paired within seed**: each perturbed scenario is compared with the baseline generated from the same train/test split. This allows the manuscript to report mean performance changes and 95% confidence intervals rather than relying on one split.

## Run

Copy the ZIP contents into the existing `rtagc_paper2_starter` root and run:

```powershell
python run_robustness_validation.py
```

No additional Python packages should be required.

## Important

This script performs **450 model fits** (30 seeds × 5 scenarios × 3 models), so it will take longer than previous milestones. Allow it to finish.

## Outputs

Under `outputs/robustness/`:

- `robustness_raw_results.csv`
- `robustness_performance_summary.csv`
- `robustness_decision_frequencies.csv`
- `robustness_fairness_summary.csv`
- `robustness_key_claims.csv`
- `robustness_metadata.json`
- `Figure_R1_RepeatedSeed_ROCAUC.png`
- `Figure_R2_Fairness_Robustness.png`
- `Figure_R3_Privacy_Utility_Robustness.png`

## Send back

Please send:

- `robustness_key_claims.csv`
- `robustness_decision_frequencies.csv`
- `robustness_fairness_summary.csv`
- console output

Once these are available, the manuscript can be revised to report repeated-seed means, SDs and 95% confidence intervals.
