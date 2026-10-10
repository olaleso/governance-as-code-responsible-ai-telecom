from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import yaml

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

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

def dataset_audit(df):
    return {
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "duplicate_customer_ids": int(df["customerID"].duplicated().sum()),
        "duplicate_customer_id_rate": float(df["customerID"].duplicated().mean()),
        "missing_cells": int(df.isna().sum().sum()),
        "missing_rate_overall": float(df.isna().mean().mean()),
        "missing_TotalCharges": int(df["TotalCharges"].isna().sum()),
        "invalid_negative_tenure": int((df["tenure"] < 0).sum()),
        "target_churn_rate": float(df["Churn"].mean()),
    }

def main():
    cfg = yaml.safe_load(Path("config/experiment.yaml").read_text(encoding="utf-8"))
    seed = int(cfg["project"]["random_seed"])
    df = load_telco(cfg["dataset"]["raw_path"])

    audit = dataset_audit(df)
    Path("outputs/metrics").mkdir(parents=True, exist_ok=True)
    Path("outputs/models").mkdir(parents=True, exist_ok=True)
    Path("outputs/metrics/baseline_dataset_audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )

    y = df["Churn"].astype(int)
    X = df.drop(columns=["Churn", "customerID"])

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=seed
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=seed
    )

    numeric = X_train.select_dtypes(include=["number"]).columns.tolist()
    categorical = [c for c in X_train.columns if c not in numeric]

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocessor = ColumnTransformer([
        ("num", numeric_pipe, numeric),
        ("cat", categorical_pipe, categorical),
    ])

    model_specs = {
        "logistic_regression": LogisticRegression(max_iter=2000, random_state=seed),
        "random_forest": RandomForestClassifier(
            n_estimators=500, min_samples_leaf=2, n_jobs=-1, random_state=seed
        ),
        "xgboost": XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=seed,
        ),
    }

    rows = []
    for name, estimator in model_specs.items():
        pipe = Pipeline([
            ("preprocess", preprocessor),
            ("model", estimator),
        ])
        pipe.fit(X_train, y_train)

        val_metrics = evaluate(pipe, X_val, y_val)
        test_metrics = evaluate(pipe, X_test, y_test)

        row = {"model": name}
        row.update({f"validation_{k}": v for k, v in val_metrics.items()})
        row.update({f"test_{k}": v for k, v in test_metrics.items()})
        rows.append(row)

        joblib.dump(pipe, Path("outputs/models") / f"{name}.joblib")

    out = pd.DataFrame(rows)
    out.to_csv("outputs/metrics/baseline_model_metrics.csv", index=False)

    print("\nBaseline complete.")
    print("Dataset audit:")
    print(json.dumps(audit, indent=2))
    print("\nModel metrics:")
    print(out.to_string(index=False))

if __name__ == "__main__":
    main()
