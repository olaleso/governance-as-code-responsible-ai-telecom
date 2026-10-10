# Data provenance

## Experimental dataset

The computational experiments use IBM's public **Telco Customer Churn** sample as a fictional telecommunications testbed.

The dataset is not represented as Nigerian subscriber data and is not used to estimate population-level behaviour in Nigeria.

## Acquisition

Use:

```bash
python download_data.py
```

Expected local path:

```text
data/raw/Telco-Customer-Churn.csv
```

Recorded SHA-256 for the file used in the reported experiments:

```text
16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91
```

## Baseline audit

The reported source contained:

- 7,043 rows;
- 21 columns;
- 0 duplicated customer identifiers;
- 11 missing cells, all in `TotalCharges`;
- no invalid negative tenure values; and
- churn prevalence of approximately 26.54%.

The machine-readable audit is stored at `outputs/metrics/baseline_dataset_audit.json`.

## Redistribution policy

The raw source file is intentionally excluded from Git. The repository contains acquisition code, provenance information, a hash, a data dictionary, and all derived experimental outputs required to verify the reported computational results.
