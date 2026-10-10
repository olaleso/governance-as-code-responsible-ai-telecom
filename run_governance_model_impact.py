from pathlib import Path
import json
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
from src.governance.data_quality import compute_quality_metrics, evaluate_data_quality_rules
from src.governance.scenarios import inject_missingness, inject_duplicates, inject_invalid_categories, inject_negative_tenure

def evaluate(model, X_test, y_test):
    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)[:, 1]
    return {
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
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median")),("scaler", StandardScaler())]), numeric),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical),
    ])

def models(seed):
    return {
        "logistic_regression": LogisticRegression(max_iter=2000, random_state=seed),
        "random_forest": RandomForestClassifier(n_estimators=500, min_samples_leaf=2, n_jobs=-1, random_state=seed),
        "xgboost": XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=4, subsample=0.9, colsample_bytree=0.9, eval_metric="logloss", random_state=seed),
    }

def decision(results):
    failed = [r for r in results if not r.passed]
    if any(r.action == "block" and r.severity in {"high","critical"} for r in failed):
        return "BLOCK"
    if failed:
        return "REVIEW"
    return "PASS"

def corrupt(df, scenario, seed):
    if scenario == "E0_baseline": return df.copy()
    if scenario == "E1_missing_05pct": return inject_missingness(df,0.05,seed+1)
    if scenario == "E2_missing_10pct": return inject_missingness(df,0.10,seed+2)
    if scenario == "E3_missing_20pct": return inject_missingness(df,0.20,seed+3)
    if scenario == "E4_duplicates_02pct": return inject_duplicates(df,0.02,seed+4)
    if scenario == "E5_duplicates_05pct": return inject_duplicates(df,0.05,seed+5)
    if scenario == "E6_invalid_contract_02pct": return inject_invalid_categories(df,0.02,seed+6)
    if scenario == "E7_negative_tenure_02pct": return inject_negative_tenure(df,0.02,seed+7)
    raise ValueError(scenario)

def main():
    cfg = yaml.safe_load(Path("config/experiment.yaml").read_text(encoding="utf-8"))
    seed = int(cfg["project"]["random_seed"])
    df = load_telco(cfg["dataset"]["raw_path"])
    train, temp = train_test_split(df, test_size=0.30, stratify=df["Churn"], random_state=seed)
    _, test = train_test_split(temp, test_size=0.50, stratify=temp["Churn"], random_state=seed)

    features = [c for c in df.columns if c not in {"Churn","customerID"}]
    X_test, y_test = test[features], test["Churn"].astype(int)

    scenarios = ["E0_baseline","E1_missing_05pct","E2_missing_10pct","E3_missing_20pct",
                 "E4_duplicates_02pct","E5_duplicates_05pct","E6_invalid_contract_02pct","E7_negative_tenure_02pct"]

    rows = []
    for scenario in scenarios:
        tr = corrupt(train, scenario, seed)
        dq = evaluate_data_quality_rules(compute_quality_metrics(tr))
        dq_decision = decision(dq)
        X_train, y_train = tr[features], tr["Churn"].astype(int)

        for name, estimator in models(seed).items():
            pipe = Pipeline([("preprocess", make_preprocessor(X_train)),("model", estimator)])
            pipe.fit(X_train, y_train)
            m = evaluate(pipe, X_test, y_test)
            rows.append({"scenario":scenario,"model":name,"dq_decision":dq_decision,
                         "failed_dq_rules":sum(not r.passed for r in dq), **{f"test_{k}":v for k,v in m.items()}})

    perf = pd.DataFrame(rows)
    baseline = perf[perf.scenario=="E0_baseline"].set_index("model")
    for metric in ["accuracy","precision","recall","f1","roc_auc"]:
        perf[f"delta_{metric}"] = perf[f"test_{metric}"] - perf["model"].map(baseline[f"test_{metric}"])

    out = Path("outputs/metrics")
    out.mkdir(parents=True, exist_ok=True)
    perf.to_csv(out/"dq_model_impact_results.csv", index=False)
    print(perf[["scenario","model","dq_decision","test_accuracy","test_f1","test_roc_auc","delta_accuracy","delta_f1","delta_roc_auc"]].to_string(index=False))

if __name__ == "__main__":
    main()
