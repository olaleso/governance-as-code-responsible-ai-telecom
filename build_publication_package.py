from pathlib import Path
import json
import math
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(".")
METRICS = ROOT / "outputs" / "metrics"
EXPL = ROOT / "outputs" / "explainability"
PUB = ROOT / "outputs" / "publication"
TABLES = PUB / "tables"
FIGURES = PUB / "figures"
TEXT = PUB / "text"

for d in [PUB, TABLES, FIGURES, TEXT]:
    d.mkdir(parents=True, exist_ok=True)

required = {
    "baseline": METRICS / "baseline_model_metrics.csv",
    "dq": METRICS / "dq_model_impact_results.csv",
    "fairness": METRICS / "fairness_scenario_results.csv",
    "privacy": METRICS / "privacy_minimisation_results.csv",
    "explainability": EXPL / "explainability_model_comparison.csv",
}

missing = [str(p) for p in required.values() if not p.exists()]
if missing:
    raise FileNotFoundError(
        "Missing required experiment files:\n- " + "\n- ".join(missing)
    )

baseline = pd.read_csv(required["baseline"])
dq = pd.read_csv(required["dq"])
fair = pd.read_csv(required["fairness"])
privacy = pd.read_csv(required["privacy"])
expl = pd.read_csv(required["explainability"])

# ---------- TABLE 1: baseline ----------
t1_cols = [
    "model","test_accuracy","test_precision","test_recall","test_f1","test_roc_auc"
]
t1 = baseline[t1_cols].copy()
t1.to_csv(TABLES / "Table1_Baseline_Model_Performance.csv", index=False)

# ---------- TABLE 2: data-quality governance vs performance ----------
t2_cols = [
    "scenario","model","dq_decision","failed_dq_rules",
    "test_accuracy","test_f1","test_roc_auc",
    "delta_accuracy","delta_f1","delta_roc_auc"
]
t2 = dq[t2_cols].copy()
t2.to_csv(TABLES / "Table2_Data_Quality_Governance_Impact.csv", index=False)

# ---------- TABLE 3: fairness ----------
t3_cols = [
    "scenario","model","fairness_decision","failed_fairness_rules",
    "test_accuracy","test_f1","test_roc_auc","selection_rate_ratio",
    "absolute_equal_opportunity_difference",
    "absolute_false_positive_rate_difference"
]
t3 = fair[t3_cols].copy()
t3.to_csv(TABLES / "Table3_Fairness_Experiment_Results.csv", index=False)

# ---------- TABLE 4: privacy/minimisation ----------
t4_cols = [
    "scenario","model","feature_count","features_removed",
    "privacy_status","test_accuracy","test_f1","test_roc_auc",
    "delta_roc_auc","utility_retention_auc_pct"
]
t4 = privacy[t4_cols].copy()
t4.to_csv(TABLES / "Table4_Privacy_Minimisation_Results.csv", index=False)

# ---------- TABLE 5: explainability ----------
t5 = expl.copy()
t5.to_csv(TABLES / "Table5_Explainability_Comparison.csv", index=False)

# ---------- FIGURE 1: baseline ROC-AUC ----------
plt.figure(figsize=(8,5))
plt.bar(t1["model"], t1["test_roc_auc"])
plt.ylim(max(0, t1["test_roc_auc"].min()-0.03), min(1, t1["test_roc_auc"].max()+0.03))
plt.ylabel("Test ROC-AUC")
plt.xlabel("Model")
plt.title("Baseline predictive performance")
plt.xticks(rotation=15, ha="right")
plt.tight_layout()
plt.savefig(FIGURES / "Figure1_Baseline_ROCAUC.png", dpi=220)
plt.close()

# ---------- FIGURE 2: DQ governance decision vs ROC-AUC ----------
pivot = dq.pivot(index="scenario", columns="model", values="test_roc_auc")
plt.figure(figsize=(11,6))
for col in pivot.columns:
    plt.plot(range(len(pivot.index)), pivot[col].values, marker="o", label=col)
plt.xticks(range(len(pivot.index)), pivot.index, rotation=35, ha="right")
plt.ylabel("Test ROC-AUC")
plt.xlabel("Data-quality scenario")
plt.title("Predictive performance under data-governance degradation")
plt.legend()
plt.tight_layout()
plt.savefig(FIGURES / "Figure2_DQ_Governance_vs_ROCAUC.png", dpi=220)
plt.close()

# ---------- FIGURE 3: fairness selection-rate ratio ----------
fp = fair.pivot(index="scenario", columns="model", values="selection_rate_ratio")
plt.figure(figsize=(11,6))
for col in fp.columns:
    plt.plot(range(len(fp.index)), fp[col].values, marker="o", label=col)
plt.axhline(0.80, linestyle="--", linewidth=1)
plt.xticks(range(len(fp.index)), fp.index, rotation=35, ha="right")
plt.ylabel("Selection-rate ratio")
plt.xlabel("Fairness scenario")
plt.title("Fairness degradation under controlled label bias")
plt.legend()
plt.tight_layout()
plt.savefig(FIGURES / "Figure3_Fairness_Selection_Rate_Ratio.png", dpi=220)
plt.close()

# ---------- FIGURE 4: privacy utility trade-off ----------
pp = privacy.pivot(index="scenario", columns="model", values="utility_retention_auc_pct")
plt.figure(figsize=(11,6))
for col in pp.columns:
    plt.plot(range(len(pp.index)), pp[col].values, marker="o", label=col)
plt.axhline(100, linestyle="--", linewidth=1)
plt.xticks(range(len(pp.index)), pp.index, rotation=30, ha="right")
plt.ylabel("ROC-AUC utility retained (%)")
plt.xlabel("Feature-minimisation scenario")
plt.title("Predictive utility retained after data minimisation")
plt.legend()
plt.tight_layout()
plt.savefig(FIGURES / "Figure4_Privacy_Utility_Tradeoff.png", dpi=220)
plt.close()

# ---------- FIGURE 5: explainability ----------
plt.figure(figsize=(8,5))
plt.bar(expl["scenario"], expl["sensitive_feature_explanation_share"] * 100)
plt.ylabel("Sensitive-feature explanation share (%)")
plt.xlabel("Model configuration")
plt.title("Sensitive-feature contribution before and after minimisation")
plt.xticks(rotation=15, ha="right")
plt.tight_layout()
plt.savefig(FIGURES / "Figure5_Sensitive_Feature_Explanation_Share.png", dpi=220)
plt.close()

def pct(x, digits=2):
    return f"{100*x:.{digits}f}%"

def num(x, digits=4):
    return f"{x:.{digits}f}"

# ---------- DYNAMIC RESULTS NARRATIVE ----------
best_baseline = baseline.loc[baseline["test_roc_auc"].idxmax()]
min_priv = privacy[privacy["scenario"]=="P4_minimum_candidate_set"].copy()
min_xgb = min_priv[min_priv["model"]=="xgboost"].iloc[0]
full_xgb = privacy[(privacy["scenario"]=="P0_full_features") & (privacy["model"]=="xgboost")].iloc[0]

f4_lr = fair[(fair["scenario"]=="F4_female_positive_label_flip_25pct") &
             (fair["model"]=="logistic_regression")]
f4_lr_text = ""
if not f4_lr.empty:
    r = f4_lr.iloc[0]
    f4_lr_text = (
        f"In the 25% controlled positive-label-flip scenario, Logistic Regression "
        f"retained ROC-AUC {num(r['test_roc_auc'])}, while the fairness engine returned "
        f"{r['fairness_decision']}. The selection-rate ratio fell to "
        f"{num(r['selection_rate_ratio'],3)} and the absolute equal-opportunity "
        f"difference rose to {num(r['absolute_equal_opportunity_difference'],3)}."
    )

dq_block = dq[dq["dq_decision"]=="BLOCK"].copy()
dq_example_text = ""
if not dq_block.empty:
    r = dq_block.iloc[(dq_block["delta_roc_auc"].abs()).argmin()]
    dq_example_text = (
        f"A representative blocked data-quality scenario ({r['scenario']}, {r['model']}) "
        f"still achieved ROC-AUC {num(r['test_roc_auc'])}, with a change of only "
        f"{num(r['delta_roc_auc'],4)} relative to that model's clean baseline."
    )

x0 = expl[expl["scenario"]=="X0_full_features"].iloc[0]
x1 = expl[expl["scenario"]=="X1_minimum_candidate_set"].iloc[0]

results_md = f"""# Draft Results — Governance-as-Code for Responsible AI

## Baseline predictive performance

Three classification models were evaluated using a fixed clean test holdout. The best baseline ROC-AUC was {num(best_baseline['test_roc_auc'])} for {best_baseline['model']}. Across all three models, baseline ROC-AUC values were tightly clustered, indicating broadly comparable discriminative performance.

## Data-quality governance experiments

Controlled training-data defects were introduced while the test holdout remained clean and fixed. The machine-executable rules distinguished clean data from scenarios containing excessive missingness, duplicate identifiers, invalid categorical values and negative tenure values. Crucially, governance failure did not necessarily correspond to large deterioration in predictive performance. {dq_example_text}

These results support treating predictive adequacy and governance adequacy as separate properties.

## Fairness experiments

The clean baseline passed the specified fairness controls for all three models. Controlled under-representation alone did not trigger the current fairness thresholds. However, controlled label manipulation progressively increased group-level disparity. {f4_lr_text}

At the most severe label-bias level, all three models were blocked by the fairness controls, while ROC-AUC remained materially above random discrimination. This demonstrates that conventional performance metrics can conceal substantial group-level governance concerns.

## Privacy and data minimisation

The full experimental model used {int(full_xgb['feature_count'])} non-identifier features. The minimum candidate configuration retained {int(min_xgb['feature_count'])} features. For XGBoost, the reduced configuration retained {min_xgb['utility_retention_auc_pct']:.2f}% of baseline ROC-AUC, while test accuracy changed from {num(full_xgb['test_accuracy'])} to {num(min_xgb['test_accuracy'])} and F1 changed from {num(full_xgb['test_f1'])} to {num(min_xgb['test_f1'])}.

The result indicates that substantial feature minimisation can reduce data exposure without necessarily causing material loss of predictive utility.

## Explainability

SHAP analysis compared the full XGBoost model with the minimum candidate feature set. The full model attributed {100*x0['sensitive_feature_explanation_share']:.2f}% of total mean absolute SHAP contribution to the designated sensitive features and {100*x0['proxy_candidate_explanation_share']:.2f}% to the selected proxy-candidate features. These shares were 0% in the minimised model because those variables were excluded.

The minimised model retained {x1['utility_retention_auc_pct']:.2f}% of the full model's ROC-AUC. The dominant feature remained {x1['top_feature']}, with its share of total explanation increasing as the feature space contracted.

SHAP values are interpreted as model-attribution measures rather than causal effects.

## Integrated finding

Across the experiments, strong or stable predictive performance did not guarantee satisfactory governance outcomes. Conversely, governance-oriented interventions such as data minimisation could materially reduce feature exposure while preserving most predictive utility. These findings motivate Governance-as-Code as a complementary assurance layer alongside conventional model-performance evaluation.
"""
(TEXT / "Results_Draft.md").write_text(results_md, encoding="utf-8")

methods_md = """# Draft Methods — Experimental Evaluation

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
"""
(TEXT / "Methods_Draft.md").write_text(methods_md, encoding="utf-8")

limitations_md = """# Current Limitations to Preserve in the Manuscript

1. The IBM Telco Customer Churn sample represents a fictional telecommunications company and is not Nigerian subscriber data.
2. Nigerian relevance should therefore be established through the governance, regulatory and sector context rather than by claiming population representativeness.
3. Fairness scenarios are controlled perturbation experiments and do not demonstrate discrimination by a real operator.
4. The fairness thresholds used in the prototype are experimental governance controls, not universal legal thresholds.
5. Removal of sensitive variables does not prove absence of indirect discrimination because retained variables may still encode proxy information.
6. SHAP measures model attribution, not causal influence.
7. A single train/test split is useful for the current prototype, but repeated-seed or repeated-cross-validation experiments should be considered before the final journal submission if computationally feasible.
8. Drift, lineage and continuous monitoring are intentionally outside the scope of this first paper and remain candidates for later PhD studies.
"""
(TEXT / "Limitations_and_Scope.md").write_text(limitations_md, encoding="utf-8")

manifest = {
    "tables": [p.name for p in sorted(TABLES.glob("*.csv"))],
    "figures": [p.name for p in sorted(FIGURES.glob("*.png"))],
    "text": [p.name for p in sorted(TEXT.glob("*.md"))],
}
(PUB / "publication_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

print("\nMilestone 7 publication package created.\n")
print("Tables:")
for p in sorted(TABLES.glob("*.csv")):
    print(" -", p)
print("Figures:")
for p in sorted(FIGURES.glob("*.png")):
    print(" -", p)
print("Draft text:")
for p in sorted(TEXT.glob("*.md")):
    print(" -", p)
