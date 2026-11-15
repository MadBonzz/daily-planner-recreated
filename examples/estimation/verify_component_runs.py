"""Check that a component audit contains every intended case for a named Bonsai model.

This validates coverage and attribution. Model mistakes and explicit fallbacks remain
part of the results. An interrupted run must not be mistaken for a complete suite.
"""

import argparse
import json
from pathlib import Path

from examples.estimation.audit_chat import CASES
from examples.estimation.audit_activity_progress import ACTIVITY, PROGRESS
from examples.estimation.forecast_audit import MODELS, OUT, challenge, confidence_cases, published


def verify(model, root, *, include_solo=False):
    short = MODELS[model].removeprefix("bonsai-")
    issues = []

    def read(path):
        try:
            return json.loads((root / path).read_text())
        except (OSError, ValueError) as error:
            issues.append(f"{path}: {error}")
            return {}

    def check(ok, description):
        if not ok:
            issues.append(description)

    groups = {"revised": [t["id"] for t, _, _ in published() + challenge()],
              "integrated": ["patch-note", "patch-tool"],
              "warning-trial": ["mara-rooms", "bake-dishes", "dev-april", "dark-frame-series"]}
    groups["confidence-trial-warning-panel"] = groups["warning-trial"]
    for variant in ("confidence-control", "confidence-trial"):
        groups[variant] = [t["id"] for t, _, _ in confidence_cases()]
    if include_solo:
        groups["solo"] = ["patch-tool"]
    for group, names in groups.items():
        for name in names:
            record = read(Path("forecast") / group / MODELS[model] / (name + ".json"))
            if not record:
                continue
            check(record.get("model") == model and record.get("task") == name, f"{group}/{name} model or case")
            receipts = record.get("receipts", [])
            provenance = record.get("prediction", {}).get("provenance", {})
            if record.get("fixed"):
                check(provenance.get("decision") == "fixed" and not receipts, f"{group}/{name} fixed event")
            else:
                check(provenance.get("decision") in {"direct", "fallback"} and receipts, f"{group}/{name} forecast")
                check(any(r.get("response_model") == model for r in receipts), f"{group}/{name} responding model")

    cycle = read(f"matrix/lifecycle-{short}.json")
    stages = [row.get("stage") for row in cycle.get("steps", [])]
    check(cycle.get("model") == model and stages == ["after_add", "after_independent_completion", "after_related_completion"]
          and len(cycle.get("checks", {})) == 8,
          "three-stage matrix lifecycle including the completed peer")
    desk = read(f"focus-workspace/{short}-fallback.json")
    check(desk.get("model") == model and len(desk.get("results", [])) == 2, "two workspace fallback decisions")
    context = read(f"forecast/context-{MODELS[model]}.json")
    check(context.get("model") == model and bool(context.get("receipts")), "context limit diagnostic")

    for experiment in ("priority-before", "priority-final", "sequence-repair"):
        ranking = read(f"chat/{experiment}-{model}.json")
        check(set(ranking) == {"fit-20", "budget-bare", "bare-half"}
              and all(row.get("model") == model for row in ranking.values()), f"{experiment} planner priorities")
    for version in ("before", "after"):
        chat = read(f"chat/{version}-{model}.json")
        expected = {name for split, name, *_ in CASES if version == "after" or split == "development"}
        check(set(chat) == expected and all(v.get("model") == model for v in chat.values()),
              f"{version} chat questions and model identities")
        for split, cases in PROGRESS.items():
            for component, result_field in (("activity", "model_labeled"), ("progress", "unavailable")):
                path = f"activity-progress/{version}/{short}-{split}-{component}.json"
                data = read(path)
                check(data.get("model") == model and result_field in data, f"finished {path}")
                expected_cases = cases if component == "progress" else ACTIVITY[split][1]
                check([row["case"] for row in data.get("cases", [])] == [c[0] for c in expected_cases],
                      path + " cases")
        feedback = read(f"activity-progress/{version}/{short}-validation-feedback.json")
        scenarios = feedback.get("scenarios", [])
        check(feedback.get("model") == model and "model_labeled" in feedback
              and [s["scenario"] for s in scenarios] == ["no feedback", "related feedback", "different task"]
              and all(len(s["intervals"]) == 3 for s in scenarios), version + " activity corrections")
    pause_trial = read(f"activity-progress/follow-on/{short}-pause.json")
    check(pause_trial.get("model") == model and {r["case"] for r in pause_trial.get("cases", [])} == {"partial", "finished"},
          "two targeted pause citation retries")
    self_report = read(f"activity-progress/follow-on/{short}-self-report.json")
    check(self_report.get("model") == model and "unavailable" in self_report and bool(self_report.get("instruction_sha256"))
          and [row["case"] for row in self_report.get("cases", [])]
          == ["finished", "new test result", "copied text", "old evidence", "hypothetical"],
          "first-hand progress and three negative controls")
    replay = read(f"activity-progress/after/{short}-validation-replay.json")
    check(replay.get("model") == model and len(replay.get("profiles", {})) == 10 and "eligible" in replay,
          "ten native ActivityWatch history replays")
    for experiment, count in (("feedback", 9), ("replay", 56)):
        follow_on = read(f"activity-progress/follow-on/{short}-{experiment}.json")
        check(follow_on.get("model") == model and follow_on.get("n") == count
              and follow_on.get("model_calls") == 0 and len(follow_on.get("cases", [])) == count,
              f"guarded activity {experiment} coverage")
    guarded = read(f"chat/scope-guard-{model}.json").get("budget-bare", {})
    check(guarded.get("model") == model and bool(guarded.get("model_calls")), "open-plan scope follow-up")
    return issues


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--root", type=Path, default=OUT.parent)
    parser.add_argument("--include-solo", action="store_true", help="Also check the isolated latency run")
    args = parser.parse_args()
    issues = verify(args.model, args.root, include_solo=args.include_solo)
    if issues:
        raise SystemExit("\n".join(issues))
    print(f"Verified complete behavioral audit for {args.model}")
