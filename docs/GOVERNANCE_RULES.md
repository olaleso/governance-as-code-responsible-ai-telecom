# Machine-executable governance controls

The prototype stores governance controls as versioned YAML. A control contains a domain, metric, comparison operator, threshold, severity, and failure action.

The implementation is intentionally deterministic: measured evidence is compared with the configured rule, and the result contributes to a `PASS`, `REVIEW`, or `BLOCK` decision.

## Data-quality controls

Current data-quality controls include:

- model-feature missingness;
- duplicate customer identifiers;
- non-negative tenure;
- categorical-domain validity; and
- numeric-domain validity.

The negative-tenure scenario can trigger both the tenure-specific rule and the broader numeric-validity rule. These are nested technical checks and should not be interpreted as two independent real-world incidents.

## Fairness controls

The controlled fairness experiment uses binary gender solely as a technical subgroup comparison available in the source dataset.

Metrics include:

- symmetric selection-rate ratio: `min(group selection rates) / max(group selection rates)`;
- absolute equal-opportunity difference; and
- absolute false-positive-rate difference.

The current thresholds (`0.80` and `0.10`) are experimental screening controls. They are not universal definitions of fairness and are not asserted as legal thresholds.

## Privacy/minimisation controls

The prototype checks exclusion of direct identifiers and evaluates progressively smaller feature sets. Candidate proxy variables are labelled as *candidate* proxies; the experiment does not claim that correlation or proxy status has been legally or causally established.

Data minimisation is evaluated as a technical exposure-versus-utility experiment. It does not by itself establish legal necessity or privacy compliance.

## Explainability evidence

SHAP is used to describe model attribution. SHAP values are not interpreted as causal effects.

## Decision semantics

- `PASS`: no configured rule failure requiring review/block.
- `REVIEW`: one or more configured controls require review but do not produce a critical block.
- `BLOCK`: at least one configured high/critical blocking condition is met.

The framework is designed as technical decision support. Production deployment should preserve human, legal, security, privacy, and domain review for requirements that cannot be reduced to numerical controls.
