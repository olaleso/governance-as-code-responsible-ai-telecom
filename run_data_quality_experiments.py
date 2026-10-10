from pathlib import Path
import json, pandas as pd, yaml
from src.data import load_telco
from src.governance.data_quality import compute_quality_metrics, evaluate_data_quality_rules, results_as_records
from src.governance.scenarios import build_scenarios

def decision(results):
    failed = [r for r in results if not r.passed]
    if any(r.action == "block" and r.severity in {"high","critical"} for r in failed):
        return "BLOCK"
    if failed:
        return "REVIEW"
    return "PASS"

def main():
    cfg = yaml.safe_load(Path("config/experiment.yaml").read_text(encoding="utf-8"))
    df = load_telco(cfg["dataset"]["raw_path"])
    scenarios = build_scenarios(df, int(cfg["project"]["random_seed"]))
    metric_rows, rule_rows = [], []
    for name, sdf in scenarios.items():
        metrics = compute_quality_metrics(sdf)
        results = evaluate_data_quality_rules(metrics)
        metric_rows.append({"scenario":name, **metrics,
                            "failed_rule_count":sum(not r.passed for r in results),
                            "data_quality_decision":decision(results)})
        rule_rows += [{"scenario":name, **r} for r in results_as_records(results)]
    out = Path("outputs/metrics"); out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(metric_rows).to_csv(out/"dq_scenario_metrics.csv", index=False)
    pd.DataFrame(rule_rows).to_csv(out/"dq_rule_results.csv", index=False)
    summary = {r["scenario"]:{"failed_rule_count":r["failed_rule_count"],
                              "decision":r["data_quality_decision"]} for r in metric_rows}
    (out/"dq_experiment_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    cols=["scenario","model_feature_missing_rate","duplicate_customer_id_rate",
          "invalid_tenure_rate","invalid_category_rate","failed_rule_count","data_quality_decision"]
    print("\nData Quality Governance experiments complete.\n")
    print(pd.DataFrame(metric_rows)[cols].to_string(index=False))

if __name__ == "__main__":
    main()
