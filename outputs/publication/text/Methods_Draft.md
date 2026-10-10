# Draft Methods — Experimental Evaluation

## Experimental design

The study used a fixed train/validation/test strategy with a clean held-out test population. Governance defects were introduced only into training data for controlled degradation experiments so that changes in predictive performance could be interpreted against a stable evaluation population.

Three supervised classification models were used: Logistic Regression, Random Forest and XGBoost. Performance was evaluated using accuracy, precision, recall, F1 score and ROC-AUC.

## Governance-as-Code evaluation

Governance requirements were represented as machine-readable rules with explicit metrics, comparison operators, thresholds, severities and failure actions. The experimental controls are research artefacts and must not be interpreted as statutory legal thresholds.

## Data-quality experiments

Controlled scenarios introduced increasing missingness, duplicate customer identifiers, invalid categorical values and invalid negative tenure values. Each scenario was assessed by the data-quality governance engine and classified as PASS, REVIEW or BLOCK.

## Fairness experiments

Gender was used solely as a controlled protected-attribute comparison for technical testing. The experiments included representation reduction and controlled positive-label flipping. Evaluation included selection-rate ratio, demographic-parity difference, equal-opportunity difference and false-positive-rate difference. The experiments do not assert discriminatory practice by any real telecommunications operator.

## Privacy and data minimisation

Progressively smaller feature sets were evaluated to quantify the relationship between reduced feature exposure and predictive utility. Direct identifiers were excluded from model training throughout. Sensitive and selected proxy-candidate variables were subsequently removed before more aggressive feature minimisation.

## Explainability

SHAP was applied to XGBoost models to quantify global and local feature contributions. Mean absolute SHAP values were aggregated from transformed features to their originating raw variables. SHAP values were treated as explanations of model behaviour rather than causal estimates.
