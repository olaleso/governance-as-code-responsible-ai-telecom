# Draft Results — Governance-as-Code for Responsible AI

## Baseline predictive performance

Three classification models were evaluated using a fixed clean test holdout. The best baseline ROC-AUC was 0.8401 for xgboost. Across all three models, baseline ROC-AUC values were tightly clustered, indicating broadly comparable discriminative performance.

## Data-quality governance experiments

Controlled training-data defects were introduced while the test holdout remained clean and fixed. The machine-executable rules distinguished clean data from scenarios containing excessive missingness, duplicate identifiers, invalid categorical values and negative tenure values. Crucially, governance failure did not necessarily correspond to large deterioration in predictive performance. A representative blocked data-quality scenario (E6_invalid_contract_02pct, logistic_regression) still achieved ROC-AUC 0.8394, with a change of only -0.0002 relative to that model's clean baseline.

These results support treating predictive adequacy and governance adequacy as separate properties.

## Fairness experiments

The clean baseline passed the specified fairness controls for all three models. Controlled under-representation alone did not trigger the current fairness thresholds. However, controlled label manipulation progressively increased group-level disparity. In the 25% controlled positive-label-flip scenario, Logistic Regression retained ROC-AUC 0.8343, while the fairness engine returned BLOCK. The selection-rate ratio fell to 0.600 and the absolute equal-opportunity difference rose to 0.250.

At the most severe label-bias level, all three models were blocked by the fairness controls, while ROC-AUC remained materially above random discrimination. This demonstrates that conventional performance metrics can conceal substantial group-level governance concerns.

## Privacy and data minimisation

The full experimental model used 19 non-identifier features. The minimum candidate configuration retained 7 features. For XGBoost, the reduced configuration retained 99.71% of baseline ROC-AUC, while test accuracy changed from 0.7890 to 0.8032 and F1 changed from 0.5670 to 0.6015.

The result indicates that substantial feature minimisation can reduce data exposure without necessarily causing material loss of predictive utility.

## Explainability

SHAP analysis compared the full XGBoost model with the minimum candidate feature set. The full model attributed 2.44% of total mean absolute SHAP contribution to the designated sensitive features and 9.51% to the selected proxy-candidate features. These shares were 0% in the minimised model because those variables were excluded.

The minimised model retained 99.64% of the full model's ROC-AUC. The dominant feature remained Contract, with its share of total explanation increasing as the feature space contracted.

SHAP values are interpreted as model-attribution measures rather than causal effects.

## Integrated finding

Across the experiments, strong or stable predictive performance did not guarantee satisfactory governance outcomes. Conversely, governance-oriented interventions such as data minimisation could materially reduce feature exposure while preserving most predictive utility. These findings motivate Governance-as-Code as a complementary assurance layer alongside conventional model-performance evaluation.
