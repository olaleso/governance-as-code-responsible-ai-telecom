from pathlib import Path
import json
import numpy as np
import pandas as pd
import yaml
import shap
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from xgboost import XGBClassifier

from src.data import load_telco

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

def build_preprocessor(X):
    numeric = X.select_dtypes(include=["number"]).columns.tolist()
    categorical = [c for c in X.columns if c not in numeric]
    return ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), numeric),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), categorical),
    ])

def clean_feature_name(name):
    if "__" in name:
        name = name.split("__", 1)[1]
    return name

def raw_feature_name(transformed_name, raw_features):
    n = clean_feature_name(transformed_name)
    if n in raw_features:
        return n
    for raw in sorted(raw_features, key=len, reverse=True):
        if n.startswith(raw + "_"):
            return raw
    return n

def fit_xgb(X_train, y_train, seed):
    prep = build_preprocessor(X_train)
    Xt = prep.fit_transform(X_train)
    model = XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="logloss",
        random_state=seed,
    )
    model.fit(Xt, y_train)
    return prep, model

def shap_analysis(prep, model, X_test, raw_features, scenario, outdir, sample_n=500, seed=20261008):
    rng = np.random.default_rng(seed)
    n = min(sample_n, len(X_test))
    idx = rng.choice(np.arange(len(X_test)), size=n, replace=False)
    X_sample = X_test.iloc[idx].copy()
    Xt = prep.transform(X_sample)
    explainer = shap.TreeExplainer(model)
    values = explainer.shap_values(Xt)
    if isinstance(values, list):
        values = values[-1]
    values = np.asarray(values)

    feature_names = prep.get_feature_names_out()
    mean_abs = np.abs(values).mean(axis=0)

    transformed = pd.DataFrame({
        "scenario": scenario,
        "transformed_feature": feature_names,
        "raw_feature": [raw_feature_name(f, raw_features) for f in feature_names],
        "mean_abs_shap": mean_abs,
    }).sort_values("mean_abs_shap", ascending=False)

    raw = (
        transformed.groupby(["scenario", "raw_feature"], as_index=False)["mean_abs_shap"]
        .sum()
        .sort_values("mean_abs_shap", ascending=False)
    )
    raw["share_of_total_explanation"] = raw["mean_abs_shap"] / raw["mean_abs_shap"].sum()

    transformed.to_csv(outdir / f"{scenario}_shap_transformed_features.csv", index=False)
    raw.to_csv(outdir / f"{scenario}_shap_raw_features.csv", index=False)

    top = raw.head(12).sort_values("mean_abs_shap")
    plt.figure(figsize=(9, 6))
    plt.barh(top["raw_feature"], top["mean_abs_shap"])
    plt.xlabel("Mean absolute SHAP value")
    plt.ylabel("Raw feature")
    plt.title(f"Top SHAP features — {scenario}")
    plt.tight_layout()
    plt.savefig(outdir / f"{scenario}_shap_top_features.png", dpi=180)
    plt.close()

    probs = model.predict_proba(Xt)[:, 1]
    top_local_idx = np.argsort(probs)[-10:][::-1]
    local_rows = []
    for rank, j in enumerate(top_local_idx, start=1):
        contrib = pd.DataFrame({
            "transformed_feature": feature_names,
            "raw_feature": [raw_feature_name(f, raw_features) for f in feature_names],
            "shap_value": values[j],
        })
        agg = contrib.groupby("raw_feature", as_index=False)["shap_value"].sum()
        agg["abs_shap"] = agg["shap_value"].abs()
        agg = agg.sort_values("abs_shap", ascending=False).head(8)
        for _, row in agg.iterrows():
            local_rows.append({
                "scenario": scenario,
                "sample_rank": rank,
                "predicted_churn_probability": float(probs[j]),
                "raw_feature": row["raw_feature"],
                "shap_value": float(row["shap_value"]),
                "abs_shap": float(row["abs_shap"]),
            })
    pd.DataFrame(local_rows).to_csv(outdir / f"{scenario}_local_explanations.csv", index=False)
    return raw

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

    full_features = [c for c in df.columns if c not in {"Churn", "customerID"}]
    min_features = [
        "tenure","InternetService","Contract","MonthlyCharges",
        "TotalCharges","TechSupport","OnlineSecurity"
    ]
    scenarios = {
        "X0_full_features": full_features,
        "X1_minimum_candidate_set": min_features,
    }

    outdir = Path("outputs/explainability")
    outdir.mkdir(parents=True, exist_ok=True)
    comparison_rows = []
    raw_tables = {}

    for scenario, features in scenarios.items():
        X_train = train[features]
        y_train = train["Churn"].astype(int)
        X_test = test[features]
        y_test = test["Churn"].astype(int)
        prep, model = fit_xgb(X_train, y_train, seed)
        metrics = evaluate(Pipeline([("preprocess", prep), ("model", model)]), X_test, y_test)
        raw = shap_analysis(prep, model, X_test, features, scenario, outdir, sample_n=500, seed=seed)
        raw_tables[scenario] = raw

        sensitive = {"gender", "SeniorCitizen"}
        proxy_candidates = {"Partner", "Dependents", "PaymentMethod"}
        sensitive_share = raw.loc[raw["raw_feature"].isin(sensitive), "share_of_total_explanation"].sum()
        proxy_share = raw.loc[raw["raw_feature"].isin(proxy_candidates), "share_of_total_explanation"].sum()

        comparison_rows.append({
            "scenario": scenario,
            "feature_count": len(features),
            **metrics,
            "sensitive_feature_explanation_share": float(sensitive_share),
            "proxy_candidate_explanation_share": float(proxy_share),
            "top_feature": raw.iloc[0]["raw_feature"],
            "top_feature_share": float(raw.iloc[0]["share_of_total_explanation"]),
        })

    comparison = pd.DataFrame(comparison_rows)
    baseline_auc = comparison.loc[comparison["scenario"] == "X0_full_features", "roc_auc"].iloc[0]
    comparison["delta_roc_auc_vs_full"] = comparison["roc_auc"] - baseline_auc
    comparison["utility_retention_auc_pct"] = 100 * comparison["roc_auc"] / baseline_auc
    comparison.to_csv(outdir / "explainability_model_comparison.csv", index=False)

    full = raw_tables["X0_full_features"][["raw_feature","share_of_total_explanation"]].rename(
        columns={"share_of_total_explanation":"full_share"}
    )
    mini = raw_tables["X1_minimum_candidate_set"][["raw_feature","share_of_total_explanation"]].rename(
        columns={"share_of_total_explanation":"minimum_share"}
    )
    shared = full.merge(mini, on="raw_feature", how="inner")
    shared["absolute_share_change"] = (shared["minimum_share"] - shared["full_share"]).abs()
    shared.to_csv(outdir / "shared_feature_explanation_shift.csv", index=False)

    summary = {
        "model": "XGBoost",
        "test_holdout_clean_and_fixed": True,
        "shap_sample_size": min(500, len(test)),
        "full_feature_count": len(full_features),
        "minimum_feature_count": len(min_features),
        "purpose": "Compare explanation structure before and after data minimisation."
    }
    (outdir / "explainability_experiment_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print("\nMilestone 6 complete: explainability analysis.\n")
    print(comparison.to_string(index=False))
    print("\nWritten to outputs/explainability/:")
    for p in sorted(outdir.iterdir()):
        print(" -", p.name)

if __name__ == "__main__":
    main()
