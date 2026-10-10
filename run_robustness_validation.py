from pathlib import Path
import json
import math
import numpy as np
import pandas as pd
import yaml
import matplotlib.pyplot as plt
from scipy.stats import t as student_t

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
from src.governance.data_quality import compute_quality_metrics, evaluate_data_quality_rules
from src.governance.scenarios import inject_missingness, inject_invalid_categories
from src.governance.fairness import subgroup_metrics, aggregate_fairness, evaluate_fairness_rules

N_SEEDS = 30
BASE_SEED = 20261008
SEEDS = [BASE_SEED + i for i in range(N_SEEDS)]

OUT = Path("outputs/robustness")
OUT.mkdir(parents=True, exist_ok=True)

def evaluate(model, X_test, y_test):
    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)[:, 1]
    return pred, {
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred, zero_division=0),
        "recall": recall_score(y_test, pred, zero_division=0),
        "f1": f1_score(y_test, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, prob),
    }

def make_preprocessor(X):
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

def model_specs(seed):
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

def governance_decision(results):
    failed = [r for r in results if not r.passed]
    if any(r.action == "block" and r.severity in {"high", "critical"} for r in failed):
        return "BLOCK"
    if failed:
        return "REVIEW"
    return "PASS"

def fairness_decision(results):
    failed = [r for r in results if not r.passed]
    if any(r.severity == "critical" and r.action == "block" for r in failed):
        return "BLOCK"
    if failed:
        return "REVIEW"
    return "PASS"

def flip_positive_labels(df, group_value, flip_fraction, seed):
    out = df.copy(deep=True)
    rng = np.random.default_rng(seed)
    eligible = out[(out["gender"] == group_value) & (out["Churn"] == 1)]
    n = int(round(len(eligible) * flip_fraction))
    if n > 0:
        chosen = rng.choice(eligible.index.to_numpy(), size=n, replace=False)
        out.loc[chosen, "Churn"] = 0
    return out

def mean_sd_ci(series, confidence=0.95):
    s = pd.Series(series).dropna().astype(float)
    n = len(s)
    mean = s.mean()
    sd = s.std(ddof=1)
    if n <= 1:
        return mean, sd, np.nan, np.nan
    alpha = 1 - confidence
    crit = student_t.ppf(1 - alpha/2, df=n-1)
    half = crit * sd / math.sqrt(n)
    return mean, sd, mean-half, mean+half

def fit_and_score(train, test, features, seed, scenario, governance_status=None, fairness_metrics=None):
    rows = []
    X_train, y_train = train[features], train["Churn"].astype(int)
    X_test, y_test = test[features], test["Churn"].astype(int)

    for model_name, estimator in model_specs(seed).items():
        pipe = Pipeline([
            ("preprocess", make_preprocessor(X_train)),
            ("model", estimator)
        ])
        pipe.fit(X_train, y_train)
        pred, perf = evaluate(pipe, X_test, y_test)

        row = {
            "seed": seed,
            "scenario": scenario,
            "model": model_name,
            "feature_count": len(features),
            **perf,
        }
        if governance_status is not None:
            row["governance_decision"] = governance_status
        if fairness_metrics is not None:
            sub = subgroup_metrics(y_test, pred, test["gender"])
            fm = aggregate_fairness(sub)
            frules = evaluate_fairness_rules(fm)
            row["fairness_decision"] = fairness_decision(frules)
            row.update(fm)
            row["failed_fairness_rules"] = sum(not r.passed for r in frules)
        rows.append(row)
    return rows

def main():
    cfg = yaml.safe_load(Path("config/experiment.yaml").read_text(encoding="utf-8"))
    df = load_telco(cfg["dataset"]["raw_path"])

    full_features = [c for c in df.columns if c not in {"Churn", "customerID"}]
    min_features = [
        "tenure", "InternetService", "Contract", "MonthlyCharges",
        "TotalCharges", "TechSupport", "OnlineSecurity"
    ]

    all_rows = []

    for i, seed in enumerate(SEEDS, start=1):
        print(f"[{i:02d}/{N_SEEDS}] Seed {seed}")

        train, temp = train_test_split(
            df, test_size=0.30, stratify=df["Churn"], random_state=seed
        )
        _, test = train_test_split(
            temp, test_size=0.50, stratify=temp["Churn"], random_state=seed
        )

        all_rows += fit_and_score(
            train, test, full_features, seed, "B0_baseline",
            governance_status="PASS"
        )

        dq_missing = inject_missingness(train, 0.20, seed + 1000)
        dq_results = evaluate_data_quality_rules(compute_quality_metrics(dq_missing))
        all_rows += fit_and_score(
            dq_missing, test, full_features, seed, "DQ1_missing_20pct",
            governance_status=governance_decision(dq_results)
        )

        dq_invalid = inject_invalid_categories(train, 0.02, seed + 2000)
        dq_results = evaluate_data_quality_rules(compute_quality_metrics(dq_invalid))
        all_rows += fit_and_score(
            dq_invalid, test, full_features, seed, "DQ2_invalid_contract_02pct",
            governance_status=governance_decision(dq_results)
        )

        fair_train = flip_positive_labels(train, "Female", 0.25, seed + 3000)
        all_rows += fit_and_score(
            fair_train, test, full_features, seed, "F1_label_flip_25pct",
            governance_status=None,
            fairness_metrics=True
        )

        all_rows += fit_and_score(
            train, test, min_features, seed, "P1_minimum_7_features",
            governance_status="PASS"
        )

    raw = pd.DataFrame(all_rows)

    base = raw[raw["scenario"] == "B0_baseline"][
        ["seed","model","accuracy","precision","recall","f1","roc_auc"]
    ].rename(columns={
        "accuracy":"baseline_accuracy",
        "precision":"baseline_precision",
        "recall":"baseline_recall",
        "f1":"baseline_f1",
        "roc_auc":"baseline_roc_auc"
    })
    raw = raw.merge(base, on=["seed","model"], how="left")
    for metric in ["accuracy","precision","recall","f1","roc_auc"]:
        raw[f"delta_{metric}"] = raw[metric] - raw[f"baseline_{metric}"]
    raw["auc_retention_pct"] = 100 * raw["roc_auc"] / raw["baseline_roc_auc"]

    raw.to_csv(OUT / "robustness_raw_results.csv", index=False)

    metrics = ["accuracy","precision","recall","f1","roc_auc","delta_accuracy","delta_f1","delta_roc_auc","auc_retention_pct"]
    summary_rows = []
    for (scenario, model), g in raw.groupby(["scenario","model"], sort=False):
        row = {"scenario":scenario, "model":model, "n_seeds":g["seed"].nunique()}
        for metric in metrics:
            mean, sd, lo, hi = mean_sd_ci(g[metric])
            row[f"{metric}_mean"] = mean
            row[f"{metric}_sd"] = sd
            row[f"{metric}_ci95_low"] = lo
            row[f"{metric}_ci95_high"] = hi
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUT / "robustness_performance_summary.csv", index=False)

    decision_rows = []
    for (scenario, model), g in raw.groupby(["scenario","model"], sort=False):
        if "governance_decision" in g and g["governance_decision"].notna().any():
            vc = g["governance_decision"].dropna().value_counts()
            decision_rows.append({
                "scenario":scenario,
                "model":model,
                "PASS_count":int(vc.get("PASS",0)),
                "REVIEW_count":int(vc.get("REVIEW",0)),
                "BLOCK_count":int(vc.get("BLOCK",0)),
                "n":int(vc.sum())
            })
        if "fairness_decision" in g and g["fairness_decision"].notna().any():
            vc = g["fairness_decision"].dropna().value_counts()
            decision_rows.append({
                "scenario":scenario+"_fairness",
                "model":model,
                "PASS_count":int(vc.get("PASS",0)),
                "REVIEW_count":int(vc.get("REVIEW",0)),
                "BLOCK_count":int(vc.get("BLOCK",0)),
                "n":int(vc.sum())
            })
    pd.DataFrame(decision_rows).to_csv(OUT / "robustness_decision_frequencies.csv", index=False)

    fair = raw[raw["scenario"]=="F1_label_flip_25pct"].copy()
    fair_metrics = [
        "selection_rate_ratio",
        "absolute_demographic_parity_difference",
        "absolute_equal_opportunity_difference",
        "absolute_false_positive_rate_difference"
    ]
    fair_rows = []
    for model, g in fair.groupby("model"):
        row = {"model":model, "n_seeds":g["seed"].nunique()}
        for metric in fair_metrics:
            mean, sd, lo, hi = mean_sd_ci(g[metric])
            row[f"{metric}_mean"] = mean
            row[f"{metric}_sd"] = sd
            row[f"{metric}_ci95_low"] = lo
            row[f"{metric}_ci95_high"] = hi
        fair_rows.append(row)
    pd.DataFrame(fair_rows).to_csv(OUT / "robustness_fairness_summary.csv", index=False)

    key = summary[summary["scenario"].isin([
        "DQ1_missing_20pct",
        "DQ2_invalid_contract_02pct",
        "F1_label_flip_25pct",
        "P1_minimum_7_features"
    ])][[
        "scenario","model","n_seeds",
        "roc_auc_mean","roc_auc_sd","roc_auc_ci95_low","roc_auc_ci95_high",
        "delta_roc_auc_mean","delta_roc_auc_ci95_low","delta_roc_auc_ci95_high",
        "auc_retention_pct_mean"
    ]].copy()
    key.to_csv(OUT / "robustness_key_claims.csv", index=False)

    plt.figure(figsize=(11,6))
    scenarios = list(summary["scenario"].drop_duplicates())
    x = np.arange(len(scenarios))
    width = 0.24
    models = ["logistic_regression","random_forest","xgboost"]
    for j, model in enumerate(models):
        vals=[]; errs=[]
        for sc in scenarios:
            r = summary[(summary.scenario==sc)&(summary.model==model)].iloc[0]
            vals.append(r["roc_auc_mean"])
            errs.append(r["roc_auc_ci95_high"]-r["roc_auc_mean"])
        plt.errorbar(x+(j-1)*width, vals, yerr=errs, marker="o", capsize=3, label=model)
    plt.xticks(x, scenarios, rotation=25, ha="right")
    plt.ylabel("Mean test ROC-AUC (95% CI)")
    plt.xlabel("Scenario")
    plt.title("Repeated-seed robustness of predictive performance")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT/"Figure_R1_RepeatedSeed_ROCAUC.png", dpi=220)
    plt.close()

    fs = pd.read_csv(OUT/"robustness_fairness_summary.csv")
    plt.figure(figsize=(8,5))
    x = np.arange(len(fs))
    y = fs["selection_rate_ratio_mean"]
    yerr = fs["selection_rate_ratio_ci95_high"] - y
    plt.errorbar(x, y, yerr=yerr, marker="o", capsize=4)
    plt.axhline(0.80, linestyle="--", linewidth=1)
    plt.xticks(x, fs["model"], rotation=15)
    plt.ylabel("Mean selection-rate ratio (95% CI)")
    plt.xlabel("Model")
    plt.title("Fairness robustness under 25% controlled label flipping")
    plt.tight_layout()
    plt.savefig(OUT/"Figure_R2_Fairness_Robustness.png", dpi=220)
    plt.close()

    ps = summary[summary["scenario"]=="P1_minimum_7_features"].copy()
    plt.figure(figsize=(8,5))
    x = np.arange(len(ps))
    y = ps["auc_retention_pct_mean"]
    yerr = ps["auc_retention_pct_ci95_high"] - y
    plt.errorbar(x, y, yerr=yerr, marker="o", capsize=4)
    plt.axhline(100, linestyle="--", linewidth=1)
    plt.xticks(x, ps["model"], rotation=15)
    plt.ylabel("Mean baseline ROC-AUC retained (%)")
    plt.xlabel("Model")
    plt.title("Robustness of utility retention after 7-feature minimisation")
    plt.tight_layout()
    plt.savefig(OUT/"Figure_R3_Privacy_Utility_Robustness.png", dpi=220)
    plt.close()

    metadata = {
        "n_seeds": N_SEEDS,
        "seed_start": BASE_SEED,
        "seed_end": BASE_SEED + N_SEEDS - 1,
        "paired_design": True,
        "test_split_changes_by_seed": True,
        "within_seed_same_test_holdout_across_scenarios": True,
        "scenarios": [
            "B0_baseline",
            "DQ1_missing_20pct",
            "DQ2_invalid_contract_02pct",
            "F1_label_flip_25pct",
            "P1_minimum_7_features"
        ],
        "models": ["logistic_regression","random_forest","xgboost"],
        "confidence_interval": "95% Student-t interval across seeds"
    }
    (OUT/"robustness_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("\nRobustness validation complete.\n")
    print(f"Seeds: {N_SEEDS}")
    print(f"Model fits: {len(raw)}")
    print("\nKey repeated-seed results:")
    print(key.to_string(index=False))
    print("\nWritten to outputs/robustness/:")
    for p in sorted(OUT.iterdir()):
        print(" -", p.name)

if __name__ == "__main__":
    main()
