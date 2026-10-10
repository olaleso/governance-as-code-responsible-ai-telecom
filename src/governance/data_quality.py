from dataclasses import asdict
import pandas as pd
import yaml
from src.governance.rules import RuleResult, compare

CATEGORICAL_ALLOWED = {
    "gender":{"Female","Male"},
    "Partner":{"Yes","No"},
    "Dependents":{"Yes","No"},
    "PhoneService":{"Yes","No"},
    "MultipleLines":{"Yes","No","No phone service"},
    "InternetService":{"DSL","Fiber optic","No"},
    "OnlineSecurity":{"Yes","No","No internet service"},
    "OnlineBackup":{"Yes","No","No internet service"},
    "DeviceProtection":{"Yes","No","No internet service"},
    "TechSupport":{"Yes","No","No internet service"},
    "StreamingTV":{"Yes","No","No internet service"},
    "StreamingMovies":{"Yes","No","No internet service"},
    "Contract":{"Month-to-month","One year","Two year"},
    "PaperlessBilling":{"Yes","No"},
    "PaymentMethod":{"Electronic check","Mailed check","Bank transfer (automatic)","Credit card (automatic)"},
}
MODEL_FEATURES = [
    "gender","SeniorCitizen","Partner","Dependents","tenure","PhoneService",
    "MultipleLines","InternetService","OnlineSecurity","OnlineBackup",
    "DeviceProtection","TechSupport","StreamingTV","StreamingMovies","Contract",
    "PaperlessBilling","PaymentMethod","MonthlyCharges","TotalCharges"
]

def compute_quality_metrics(df):
    block = df[MODEL_FEATURES]
    invalid_cat = 0
    cat_cells = 0
    for col, allowed in CATEGORICAL_ALLOWED.items():
        s = df[col].dropna()
        cat_cells += len(s)
        invalid_cat += int((~s.isin(allowed)).sum())
    invalid_tenure = int(((df["tenure"].notna()) & (df["tenure"] < 0)).sum())
    invalid_monthly = int(((df["MonthlyCharges"].notna()) & (df["MonthlyCharges"] < 0)).sum())
    invalid_total = int(((df["TotalCharges"].notna()) & (df["TotalCharges"] < 0)).sum())
    dup_rows = int(df["customerID"].duplicated(keep=False).sum())
    return {
        "row_count": int(len(df)),
        "model_feature_missing_rate": float(block.isna().to_numpy().mean()),
        "duplicate_customer_id_rate": float(dup_rows / len(df)),
        "invalid_tenure_rate": float(invalid_tenure / len(df)),
        "invalid_numeric_rate": float((invalid_tenure + invalid_monthly + invalid_total)/(3*len(df))),
        "invalid_category_rate": float(invalid_cat / cat_cells) if cat_cells else 0.0,
        "duplicate_customer_rows": dup_rows,
        "invalid_category_cells": invalid_cat,
    }

def evaluate_data_quality_rules(metrics, rules_path="governance/governance_rules.yaml"):
    doc = yaml.safe_load(open(rules_path, "r", encoding="utf-8"))
    results = []
    for rule in doc["rules"]:
        if rule["domain"] != "data_quality" or rule["metric"] not in metrics:
            continue
        observed = metrics[rule["metric"]]
        results.append(RuleResult(
            rule_id=rule["id"],
            passed=bool(compare(observed, rule["operator"], rule["threshold"])),
            observed=observed,
            expected=f'{rule["operator"]} {rule["threshold"]}',
            severity=rule["severity"],
            action=rule["failure_action"],
        ))
    return results

def results_as_records(results):
    return [asdict(r) for r in results]
