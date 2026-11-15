"""Retest only the two 4B pause failures after tightening the retry response contract."""

import argparse
import ast
import json
from pathlib import Path

import workspace_forecast

from examples.estimation.audit_activity_progress import PROGRESS, progress_case
from model_client import model_settings


def run(output):
    model = model_settings()[1]
    result = json.loads(output.read_text()) if output.is_file() else {"model": model, "cases": []}
    if result["model"] != model:
        raise ValueError("The output belongs to another checkpoint")
    for split, name in [("development", "partial"), ("validation", "finished")]:
        if any(row["case"] == name for row in result["cases"]):
            continue
        example = next(row for row in PROGRESS[split] if row[0] == name)
        result["cases"].append(progress_case(*example))
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2) + "\n")
        print(name, result["cases"][-1]["status"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--frozen-module", type=Path,
                        help="Read the original _INSTRUCTIONS string literal from the frozen workspace_forecast.py.")
    args = parser.parse_args()
    if args.frozen_module:
        tree = ast.parse(args.frozen_module.read_text())
        prompt = next(node.value for node in tree.body if isinstance(node, ast.Assign)
                      and any(isinstance(name, ast.Name) and name.id == "_INSTRUCTIONS" for name in node.targets))
        workspace_forecast._INSTRUCTIONS = ast.literal_eval(prompt)
    run(args.output)
