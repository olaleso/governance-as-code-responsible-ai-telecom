from dataclasses import dataclass
from typing import Any

@dataclass
class RuleResult:
    rule_id: str
    passed: bool
    observed: Any
    expected: Any
    severity: str
    action: str

def compare(observed, operator, threshold):
    if operator == "<=": return observed <= threshold
    if operator == ">=": return observed >= threshold
    if operator == "==": return observed == threshold
    if operator == "<": return observed < threshold
    if operator == ">": return observed > threshold
    raise ValueError(f"Unsupported operator: {operator}")
