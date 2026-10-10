from pathlib import Path
import json
import numpy as np
import pandas as pd
import yaml

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from src.data import load_telco
from src.governance.fairness import subgroup_metrics, aggregate_fairness, evaluate_fairness_rules, records

def evaluate_model(model, X_test, y_test):
    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)[:, 1]
    return pred, {
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred, zero_division=0),
        "recall": recall_score(y_test, pred, zero_division=0),
        "f1": f1_score(y_test, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, prob),
    }

def preprocessor(X):
    numeric = X.select_dtypes(include=["number"]).columns.tolist()
    categorical = [c for c in X.columns if c not in numeric]
    return ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]), numeric),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]), categorical),
    ])

def models(seed):
    return {
        "logistic_regression": LogisticRegression(max_iter=2000, random_state=seed),
        "random_forest": RandomForestClassifier(
            n_estimators=500, min_samples_leaf=2, n_jobs=-1, random_state=seed
        ),
        "xgboost": XGBClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=4,
            subsample=0.9, colsample_bytree=0.9,
            eval_metric="logloss", random_state=seed
        ),
    }

def fairness_decision(results):
    failed = [r for r in results if not r.passed]
    if any(r.severity == "critical" and r.action == "block" for r in failed):
        return "BLOCK"
    if failed:
        return "REVIEW"
    return "PASS"

def underrepresent_group(df, group_value, keep_fraction, seed):
    rng = np.random.default_rng(seed)
    keep_other = df[df["gender"] != group_value]
    target = df[df["gender"] == group_value]
    n_keep = max(1, int(round(len(target) * keep_fraction)))
    chosen = rng.choice(target.index.to_numpy(), size=n_keep, replace=False)
    return pd.concat([keep_other, target.loc[chosen]], axis=0).sample(frac=1, random_state=seed)

def flip_positive_labels(df, group_value, flip_fraction, seed):
    out = df.copy(deep=True)
    rng = np.random.default_rng(seed)
    eligible = out[(out["gender"] == group_value) & (out["Churn"] == 1)]
    n = int(round(len(eligible) * flip_fraction))
    if n > 0:
        chosen = rng.choice(eligible.index.to_numpy(), size=n, replace=False)
        out.loc[chosen, "Churn"] = 0
    return out

def build_scenarios(train, seed):
    return {
        "F0_baseline": train.copy(),
        "F1_female_keep_50pct": underrepresent_group(train, "Female", 0.50, seed + 1),
        "F2_female_keep_25pct": underrepresent_group(train, "Female", 0.25, seed + 2),
        "F3_female_positive_label_flip_10pct": flip_positive_labels(train, "Female", 0.10, seed + 3),
        "F4_female_positive_label_flip_25pct": flip_positive_labels(train, "Female", 0.25, seed + 4),
        "F5_female_positive_label_flip_50pct": flip_positive_labels(train, "Female", 0.50, seed + 5),
    }

def main():
    cfg = yaml.safe_load(Path("config/experiment.yaml").read_text(encoding="utf-8"))
    seed = int(cfg["project"]["random_seed"])
    df = load_telco(cfg["dataset"]["raw_path"])

    train, temp = train_test_split(
        df, test_size=0.30, stratify=df["Churn"], random_state=seed
    )
    _, test = train_test_split(
        temp, test_size=0.50, stratify=temp["Churn"], random_state=seed
    )

    features = [c for c in df.columns if c not in {"Churn", "customerID"}]
    X_test = test[features]
    y_test = test["Churn"].astype(int)
    protected_test = test["gender"]

    results_rows = []
    subgroup_rows = []
    rule_rows = []

    for scenario, tr in build_scenarios(train, seed).items():
        X_train = tr[features]
        y_train = tr["Churn"].astype(int)

        for model_name, estimator in models(seed).items():
            pipe = Pipeline([
                ("preprocess", preprocessor(X_train)),
                ("model", estimator)
            ])
            pipe.fit(X_train, y_train)
            pred, perf = evaluate_model(pipe, X_test, y_test)

            sub = subgroup_metrics(y_test, pred, protected_test)
            fair = aggregate_fairness(sub)
            fair_rules = evaluate_fairness_rules(fair)
            decision = fairness_decision(fair_rules)

            results_rows.append({
                "scenario": scenario,
                "model": model_name,
                "training_rows": int(len(tr)),
                "female_training_rows": int((tr["gender"] == "Female").sum()),
                "male_training_rows": int((tr["gender"] == "Male").sum()),
                "fairness_decision": decision,
                "failed_fairness_rules": int(sum(not r.passed for r in fair_rules)),
                **{f"test_{k}": v for k, v in perf.items()},
                **fair,
            })

            for _, row in sub.iterrows():
                subgroup_rows.append({
                    "scenario": scenario,
                    "model": model_name,
                    **row.to_dict()
                })

            for rec in records(fair_rules):
                rule_rows.append({
                    "scenario": scenario,
                    "model": model_name,
                    **rec
                })

    results = pd.DataFrame(results_rows)
    baseline = results[results["scenario"] == "F0_baseline"].set_index("model")
    for metric in ["accuracy", "f1", "roc_auc"]:
        results[f"delta_{metric}"] = (
            results[f"test_{metric}"] -
            results["model"].map(baseline[f"test_{metric}"])
        )

    out = Path("outputs/metrics")
    out.mkdir(parents=True, exist_ok=True)
    results.to_csv(out/"fairness_scenario_results.csv", index=False)
    pd.DataFrame(subgroup_rows).to_csv(out/"fairness_subgroup_metrics.csv", index=False)
    pd.DataFrame(rule_rows).to_csv(out/"fairness_rule_results.csv", index=False)

    summary = {
        "protected_attribute": "gender",
        "comparison_note": "Groups are used as a technical controlled comparison; no normative privileged/unprivileged designation is made.",
        "clean_test_holdout": True,
        "scenario_count": int(results["scenario"].nunique()),
        "model_count": int(results["model"].nunique()),
        "result_rows": int(len(results)),
    }
    (out/"fairness_experiment_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    cols = [
        "scenario","model","fairness_decision",
        "test_accuracy","test_f1","test_roc_auc",
        "selection_rate_ratio",
        "absolute_equal_opportunity_difference",
        "absolute_false_positive_rate_difference",
        "failed_fairness_rules"
    ]
    print("\nMilestone 4 complete: fairness and controlled bias experiments.\n")
    print(results[cols].to_string(index=False))
    print("\nWritten:")
    print(" - outputs/metrics/fairness_scenario_results.csv")
    print(" - outputs/metrics/fairness_subgroup_metrics.csv")
    print(" - outputs/metrics/fairness_rule_results.csv")
    print(" - outputs/metrics/fairness_experiment_summary.json")

if __name__ == "__main__":
    main()
