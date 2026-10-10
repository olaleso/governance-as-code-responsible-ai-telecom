from pathlib import Path
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

def eval_model(model, X_test, y_test):
    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)[:,1]
    return {
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred, zero_division=0),
        "recall": recall_score(y_test, pred, zero_division=0),
        "f1": f1_score(y_test, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, prob),
    }

def preprocessor(X):
    num = X.select_dtypes(include=["number"]).columns.tolist()
    cat = [c for c in X.columns if c not in num]
    return ColumnTransformer([
        ("num", Pipeline([("imputer",SimpleImputer(strategy="median")),("scaler",StandardScaler())]), num),
        ("cat", Pipeline([("imputer",SimpleImputer(strategy="most_frequent")),("onehot",OneHotEncoder(handle_unknown="ignore"))]), cat),
    ])

def models(seed):
    return {
        "logistic_regression": LogisticRegression(max_iter=2000, random_state=seed),
        "random_forest": RandomForestClassifier(n_estimators=500, min_samples_leaf=2, n_jobs=-1, random_state=seed),
        "xgboost": XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=4,
                               subsample=0.9, colsample_bytree=0.9, eval_metric="logloss",
                               random_state=seed),
    }

def scenarios(all_features):
    sensitive = {"gender","SeniorCitizen"}
    behavioural_proxy = {"Partner","Dependents","PaymentMethod"}
    reduced = {
        "tenure","PhoneService","MultipleLines","InternetService","OnlineSecurity",
        "OnlineBackup","DeviceProtection","TechSupport","StreamingTV","StreamingMovies",
        "Contract","PaperlessBilling","MonthlyCharges","TotalCharges"
    }
    minimum = {"tenure","InternetService","Contract","MonthlyCharges","TotalCharges","TechSupport","OnlineSecurity"}
    return {
        "P0_full_features": list(all_features),
        "P1_remove_sensitive": [c for c in all_features if c not in sensitive],
        "P2_remove_sensitive_and_proxies": [c for c in all_features if c not in sensitive|behavioural_proxy],
        "P3_reduced_operational_set": [c for c in all_features if c in reduced],
        "P4_minimum_candidate_set": [c for c in all_features if c in minimum],
    }

def main():
    cfg = yaml.safe_load(Path("config/experiment.yaml").read_text(encoding="utf-8"))
    seed = int(cfg["project"]["random_seed"])
    df = load_telco(cfg["dataset"]["raw_path"])

    train, temp = train_test_split(df, test_size=0.30, stratify=df["Churn"], random_state=seed)
    _, test = train_test_split(temp, test_size=0.50, stratify=temp["Churn"], random_state=seed)

    all_features = [c for c in df.columns if c not in {"Churn","customerID"}]
    sets = scenarios(all_features)
    sensitive = {"gender","SeniorCitizen"}

    rows = []
    for scenario, feats in sets.items():
        X_train, y_train = train[feats], train["Churn"].astype(int)
        X_test, y_test = test[feats], test["Churn"].astype(int)

        for name, estimator in models(seed).items():
            pipe = Pipeline([("preprocess",preprocessor(X_train)),("model",estimator)])
            pipe.fit(X_train, y_train)
            m = eval_model(pipe, X_test, y_test)

            rows.append({
                "scenario": scenario,
                "model": name,
                "feature_count": len(feats),
                "features_removed": len(all_features)-len(feats),
                "retained_feature_fraction": len(feats)/len(all_features),
                "direct_identifier_count": 0,
                "sensitive_attribute_count": sum(c in sensitive for c in feats),
                "privacy_status": "REVIEW" if any(c in sensitive for c in feats) else "PASS",
                **{f"test_{k}":v for k,v in m.items()}
            })

    out = pd.DataFrame(rows)
    baseline = out[out.scenario=="P0_full_features"].set_index("model")
    for metric in ["accuracy","f1","roc_auc"]:
        out[f"delta_{metric}"] = out[f"test_{metric}"] - out["model"].map(baseline[f"test_{metric}"])
    out["utility_retention_auc_pct"] = 100 * out["test_roc_auc"] / out["model"].map(baseline["test_roc_auc"])

    p = Path("outputs/metrics")
    p.mkdir(parents=True, exist_ok=True)
    out.to_csv(p/"privacy_minimisation_results.csv", index=False)

    print("\nMilestone 5 complete: privacy/data-minimisation experiment.\n")
    print(out[[
        "scenario","model","feature_count","privacy_status","test_accuracy","test_f1",
        "test_roc_auc","delta_roc_auc","utility_retention_auc_pct"
    ]].to_string(index=False))
    print("\nWritten:")
    print(" - outputs/metrics/privacy_minimisation_results.csv")

if __name__ == "__main__":
    main()
