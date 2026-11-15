"""Replay saved activity judgments through the final app's evidence check, without inference."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import activity
from examples.estimation import activity_fixture
from examples.estimation.audit_activity_progress import binary
from examples.estimation.run import task_inputs


def assess(task, observation, examples, raw, expected, **case):
    model_source = raw.get("source") or raw["label_source"]
    guarded = model_source == "model" and activity.ungrounded_activity_label(
        activity.classification_context(task), observation, examples, raw["label"], raw["reason"])
    return {**case, "app": observation.get("app"), "domain": observation.get("domain"),
            "title": observation.get("title"), "device": activity.device_label(observation),
            "phone_app": observation.get("phone_app") or "",
            "expected": expected, "model_label": raw["label"], "model_source": model_source,
            "model_reason": raw["reason"], "final_label": "uncertain" if guarded else raw["label"],
            "final_source": "insufficient evidence" if guarded else model_source}


def native_cases(result):
    directory = ROOT / "data" / "tasks"
    cases = []
    for profile, sample in result["profiles"].items():
        history = json.loads((directory / "histories" / f"{profile}.json").read_text())
        task = next(t for t in history["tasks"] if t["id"] == sample["task"])
        context, _ = task_inputs(task, history["id"], directory)
        observed = activity_fixture.replay_history(history, directory=directory)[task["id"]]["report"]["intervals"]
        for i, (raw, row) in enumerate(zip(sample["rows"], observed, strict=True)):
            assert all(row[key] == raw[key] for key in ("title", "domain", "seconds"))
            assert row["label"] == raw["expected"] and activity.device_label(row) == raw["device"]
            assert (row["label_source"] == "sample annotation") == raw["eligible_for_model"]
            if raw["eligible_for_model"]:
                cases.append(assess(context, row, [], raw, raw["expected"], profile=profile, task_id=task["id"], interval=i))
    assert len(cases) == result["eligible"]
    return cases


def rescore(source):
    result = json.loads(source.read_text())
    if result["component"] == "feedback":
        cases = [assess(scenario["task"], row, scenario["feedback"], row, row["expected"], scenario=scenario["scenario"])
                 for scenario in result["scenarios"] for row in scenario["intervals"]]
    elif result["component"] == "replay":
        cases = native_cases(result)
    else:
        raise ValueError("Use a saved feedback probe or SQLite replay.")
    return {"model": result["model"], "source": source.name, "model_calls": 0, "n": len(cases),
            "changed": sum(r["model_label"] != r["final_label"] for r in cases),
            "corrected_errors": sum(r["expected"] == r["final_label"] != r["model_label"] for r in cases),
            "introduced_errors": sum(r["expected"] == r["model_label"] != r["final_label"] for r in cases),
            "cases": cases,
            "by_class": {label: binary([r["expected"] == label for r in cases],
                                       [r["final_label"] == label for r in cases]) for label in activity.LABELS}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rescore(args.source), indent=2) + "\n")
