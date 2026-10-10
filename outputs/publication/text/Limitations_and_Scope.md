# Current Limitations to Preserve in the Manuscript

1. The IBM Telco Customer Churn sample represents a fictional telecommunications company and is not Nigerian subscriber data.
2. Nigerian relevance should therefore be established through the governance, regulatory and sector context rather than by claiming population representativeness.
3. Fairness scenarios are controlled perturbation experiments and do not demonstrate discrimination by a real operator.
4. The fairness thresholds used in the prototype are experimental governance controls, not universal legal thresholds.
5. Removal of sensitive variables does not prove absence of indirect discrimination because retained variables may still encode proxy information.
6. SHAP measures model attribution, not causal influence.
7. A single train/test split is useful for the current prototype, but repeated-seed or repeated-cross-validation experiments should be considered before the final journal submission if computationally feasible.
8. Drift, lineage and continuous monitoring are intentionally outside the scope of this first paper and remain candidates for later PhD studies.
