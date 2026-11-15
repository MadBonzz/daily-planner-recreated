"""Measure the effective blocking policy and the workspace gate independently against hand-labeled cases.

Run against the local Kev System 1 endpoint with
python -m examples.estimation.audit_focus_workspace --output data/results/component-audit/focus-workspace/run.json
"""
import argparse
from contextlib import ExitStack
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import focus_domains
import focus_judge
import retrieval
import workspace_eligibility
from model_client import ModelUnavailable, model_settings, planner_settings
from planner_store import PlannerStore


DATA = Path(__file__).resolve().parents[2] / "data/results/component-audit/focus-workspace"


def metrics(rows):
    cells = {name: 0 for name in ("tp", "fp", "fn", "tn")}
    unavailable = sum(row["predicted"] is None for row in rows)
    for row in rows:
        if row["predicted"] is None:
            continue
        cells[{(True, True): "tp", (False, True): "fp", (True, False): "fn",
               (False, False): "tn"}[row["expected"], row["predicted"]]] += 1
    tp, fp, fn = cells["tp"], cells["fp"], cells["fn"]
    return {**cells, **({"unavailable": unavailable} if unavailable else {}),
            "precision": tp / (tp + fp) if tp + fp else None,
            "recall": tp / (tp + fn) if tp + fn else None}


def focus_case(root, case):
    store = PlannerStore(root)
    sources = []
    for i, visits in enumerate(case["visits"]):
        source = store.create_task(title=f"Prepare previous edition {i+1}", description="Finish the preceding assignment.",
                                   day="2026-03-01", estimate_minutes=30, prediction={})
        intervals = [{"app": "Browser", "domain": case["domain"], "label": "off_task",
                      "seconds": 180, "idle": False, **visit} for visit in visits]
        store.transition(source["id"], "finish", spent_seconds=1800, report={"intervals": intervals})
        sources.append(source["id"])
    task = store.create_task(title=case["title"], description=case["description"], details=case.get("details", ""),
                             pdf_text=case.get("pdf_text", ""), day="2026-03-02", estimate_minutes=30,
                             prediction={"related_tasks": [{"task_id": key} for key in sources]})
    prediction = task["prediction"]
    # The baseline revision exposed this same estimation hook under its singleton-only name.
    review = (getattr(focus_domains, "review_sites", None) or focus_domains.review_singles)
    review(store, task, prediction, case.get("pdf_text", ""), {})
    with store._db(write=True) as db:
        db.execute("UPDATE tasks SET prediction=? WHERE id=?", (json.dumps(prediction), task["id"]))
    policy = focus_domains.policy(store, task["id"])
    if policy["pending"]:
        raise RuntimeError("Model did not finish pending focus decisions")
    return case["domain"] in policy["domains"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipts", type=Path, nargs="+", default=[],
                        help="Use previous live responses only when the complete Kev request matches exactly.")
    parser.add_argument("--receipts-only", action="store_true", help="Fail if there is no exact prior receipt.")
    parser.add_argument("--workspace-fallback", action="store_true",
                        help="Simulate Kev downtime and invoke the real chat model on one eligible and one ineligible task.")
    parser.add_argument("--split", choices=("development", "held_out", "all"), default="all")
    parser.add_argument("--component", choices=("focus", "workspace", "all"), default="all")
    args = parser.parse_args()
    cases = json.loads((DATA / "cases.json").read_text())
    config = planner_settings()
    endpoint = config.get("PLANNER_DECISION_ENDPOINT") or "http://127.0.0.1:1234/v1/systemone"
    model = config.get("PLANNER_DECISION_MODEL") or retrieval.DECISION_MODEL
    if args.workspace_fallback:
        model = model_settings(config)[1]
        cases = [case for case in cases if case["id"] in {"desk-missing-paper", "desk-writing"}]
    result = (json.loads(args.output.read_text()) if args.output.exists()
              else {"model": model, "results": [], "metrics": {}})
    if result["model"] != model:
        raise ValueError("Use a fresh output file for a different model.")
    completed = {row["id"] for row in result["results"]}
    recorded = {}
    for archive in args.receipts:
        for prior in json.loads(archive.read_text())["results"]:
            for call in prior["live_calls"]:
                if "request" in call and "replayed_from" not in call:
                    recorded[focus_judge.key(call["request"])] = (archive.name, call)
    transport = retrieval._local_post
    for case in cases:
        if case["id"] in completed:
            continue
        if args.split not in ("all", case["split"]) or args.component not in ("all", case["component"]):
            continue
        receipt = []

        def record(url, request, settings):
            assert url == endpoint
            previous = recorded.get(focus_judge.key(request))
            if previous:
                filename, call = previous
                assert request == call["request"]
                receipt.append({**call, "replayed_from": filename})
                return {"answers": call["answers"]}
            if args.receipts_only:
                raise ValueError(f"No exact receipt for {case['id']}")
            reply = transport(url, request, settings)
            receipt.append({"request": request, "answers": reply["answers"]})
            return reply

        if args.workspace_fallback:
            request = workspace_eligibility.payload(case, config)
            chat = workspace_eligibility.chat_json
            def record_chat(messages, schema, **options):
                reply = chat(messages, schema, **options)
                receipt.append({"messages": messages, "schema": schema, "answer": reply})
                return reply

            with (patch.object(workspace_eligibility, "package", return_value={}),
                  patch.object(workspace_eligibility, "_local_post", side_effect=ModelUnavailable("Kev is unavailable")),
                  patch.object(workspace_eligibility, "chat_json", record_chat)):
                try:
                    predicted, source = workspace_eligibility.decide(request, config)
                except ModelUnavailable as exc:
                    predicted = None
                    receipt.append({"error": str(exc)})
                else:
                    assert source == model, f"The app attributed the answer to {source!r} instead of {model!r}"
        elif case["component"] == "focus":
            with TemporaryDirectory() as tmp, ExitStack() as stack:
                stack.enter_context(patch.object(retrieval, "_local_post", record))
                # New examples represent novel tasks. Only ship past Dev verdicts for the actual public demo.
                stack.enter_context(patch.object(focus_judge, "packaged", return_value=None))
                predicted = focus_case(Path(tmp), case)
        else:
            request = workspace_eligibility.payload(case, config)
            with patch.object(workspace_eligibility, "_local_post", record):
                predicted = workspace_eligibility.live(request, config)
        row = {"id": case["id"], "component": case["component"], "split": case["split"],
               "expected": case["expected"], "predicted": predicted, "live_calls": receipt}
        result["results"].append(row)
        selected = [r for r in result["results"] if r["component"] == row["component"] and r["split"] == row["split"]]
        result["metrics"][row["component"] + "/" + row["split"]] = metrics(selected)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
        print(row["id"], predicted, "correct" if predicted == case["expected"] else "MISS", flush=True)


if __name__ == "__main__":
    main()
