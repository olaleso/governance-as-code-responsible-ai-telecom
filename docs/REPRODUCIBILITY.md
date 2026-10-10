# Reproducibility protocol

## Environment

Reported experiments were run with Python 3.12. Use `requirements-tested.txt` for the exact package versions recorded during the experiment and `requirements.txt` for a compatible environment specification.

## Deterministic baseline

The primary experiment configuration is stored in `config/experiment.yaml`.

The canonical initial random seed is:

```text
20261008
```

The baseline uses a stratified 70/15/15 train/validation/test design. Customer identifiers are excluded from model features.

## Controlled perturbations

Data-quality and fairness perturbations are applied to training data while the corresponding evaluation holdout remains clean. This separates the injected governance defect from changes in the evaluation population.

## Robustness protocol

The robustness analysis uses 30 pre-specified seeds from `20261008` to `20261037`.

For every seed:

1. a stratified split is generated;
2. the baseline and all targeted robustness scenarios share that seed's test holdout;
3. all three models are retrained; and
4. each perturbed result is paired with the same seed/model baseline.

The robustness run therefore contains 450 model fits:

```text
30 seeds x 5 scenarios x 3 models = 450 fits
```

Mean, standard deviation, and 95% Student-t confidence intervals are calculated across seeds.

See `outputs/robustness/robustness_metadata.json` for the machine-readable protocol description.

## Recommended reproduction sequence

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

## Interpretation boundaries

Reproduction of the code demonstrates computational reproducibility of the prototype and reported experiments. It does not imply legal certification, external validity for Nigerian subscribers, causal identification, or universal fairness thresholds.
