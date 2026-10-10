# Robustness outputs

This directory versions the compact, publication-relevant outputs from the 30-seed paired robustness analysis:

- `robustness_key_claims.csv`
- `robustness_performance_summary.csv`
- `robustness_fairness_summary.csv`
- `robustness_decision_frequencies.csv`
- `robustness_metadata.json`

The complete 450-row `robustness_raw_results.csv` and PNG figures are deterministic generated artefacts and are intentionally not required for the source release. Recreate them with:

```bash
python run_robustness_validation.py
```

The generated raw file is the source from which the committed summaries are calculated.
