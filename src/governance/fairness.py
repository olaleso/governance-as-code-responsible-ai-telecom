from dataclasses import dataclass, asdict
import math
import numpy as np
import pandas as pd
import yaml

from src.governance.rules import compare

@dataclass
class FairnessRuleResult:
    rule_id: str
    passed: bool
    observed: float
    expected: str
    severity: str
    action: str

def _safe_div(a, b):
    return float(a / b) if b else float("nan")

def subgroup_metrics(y_true, y_pred, protected):
    df = pd.DataFrame({
        "y_true": np.asarray(y_true),
        "y_pred": np.asarray(y_pred),
        "group": np.asarray(protected),
    })
    rows = []
    for group, g in df.groupby("group", dropna=False):
        tp = int(((g.y_true == 1) & (g.y_pred == 1)).sum())
        tn = int(((g.y_true == 0) & (g.y_pred == 0)).sum())
        fp = int(((g.y_true == 0) & (g.y_pred == 1)).sum())
        fn = int(((g.y_true == 1) & (g.y_pred == 0)).sum())
        n = int(len(g))
        rows.append({
            "group": str(group),
            "n": n,
            "positive_predictions": int((g.y_pred == 1).sum()),
            "selection_rate": _safe_div(int((g.y_pred == 1).sum()), n),
            "true_positive_rate": _safe_div(tp, tp + fn),
            "false_positive_rate": _safe_div(fp, fp + tn),
            "precision": _safe_div(tp, tp + fp),
            "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        })
    return pd.DataFrame(rows)

def aggregate_fairness(subgroups):
    if len(subgroups) != 2:
        raise ValueError("This prototype expects exactly two groups.")
    a, b = subgroups.iloc[0], subgroups.iloc[1]

    rates = [a["selection_rate"], b["selection_rate"]]
    finite_rates = [x for x in rates if not math.isnan(x)]
    if len(finite_rates) < 2 or max(finite_rates) == 0:
        ratio = float("nan")
    else:
        ratio = min(finite_rates) / max(finite_rates)

    return {
        "group_a": a["group"],
        "group_b": b["group"],
        "selection_rate_a": float(a["selection_rate"]),
        "selection_rate_b": float(b["selection_rate"]),
        "selection_rate_ratio": float(ratio),
        "absolute_demographic_parity_difference": float(abs(a["selection_rate"] - b["selection_rate"])),
        "absolute_equal_opportunity_difference": float(abs(a["true_positive_rate"] - b["true_positive_rate"])),
        "absolute_false_positive_rate_difference": float(abs(a["false_positive_rate"] - b["false_positive_rate"])),
    }

def evaluate_fairness_rules(metrics, rules_path="governance/fairness_rules.yaml"):
    doc = yaml.safe_load(open(rules_path, "r", encoding="utf-8"))
    results = []
    for rule in doc["rules"]:
        observed = metrics[rule["metric"]]
        passed = False if (isinstance(observed, float) and math.isnan(observed)) else compare(
            observed, rule["operator"], rule["threshold"]
        )
        results.append(FairnessRuleResult(
            rule_id=rule["id"],
            passed=bool(passed),
            observed=float(observed),
            expected=f'{rule["operator"]} {rule["threshold"]}',
            severity=rule["severity"],
            action=rule["failure_action"],
        ))
    return results

def records(results):
    return [asdict(r) for r in results]
