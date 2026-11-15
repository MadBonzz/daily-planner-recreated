"""Run one chronological planner simulation and score forecasts made at task creation."""

import argparse
from contextlib import contextmanager, ExitStack, nullcontext
from datetime import date, datetime, time
import hashlib
import json
from pathlib import Path
from statistics import mean
from tempfile import TemporaryDirectory
from unittest.mock import patch

import activity
import dev_schedule
import estimator
from examples.estimation.run import DATASET, histories
from model_client import forecast_endpoint, ModelUnavailable, planner_settings
from planner_store import PlannerStore
import retrieval


@contextmanager
def _offline():
    """Exercise the normal fallback while cutting off every model entry point in this process."""
    def unavailable(*_args, **_kwargs):
        raise ModelUnavailable("Offline replay. Model inference was disabled.")

    with ExitStack() as stack:
        for module, name in ((estimator, "chat_json"), (activity, "chat_json"), (estimator, "_search"),
                             (retrieval, "decide_related")):
            stack.enter_context(patch.object(module, name, unavailable))
        stack.enter_context(patch.object(estimator, "index_current_task", lambda *_args: None))
        stack.enter_context(patch.object(estimator, "forecast_endpoint",
                                        lambda: nullcontext({"provider": None, "model": None})))
        yield


def _kind(prediction):
    return {"fixed": "scheduled", "direct": "model"}.get((prediction.get("provenance") or {}).get("decision"), "fallback")


def _forecast(prediction, ids):
    used = prediction.get("used_task_ids", [r["task_id"] for r in prediction.get("evidence", [])])
    result = {k: prediction[k] for k in ("basis", "point_min", "low_min", "high_min", "confidence",
              "model_rationale", "confidence_reason", "explanation", "model_error", "retrieval_error") if k in prediction}
    result.update(source=_kind(prediction), provenance=prediction.get("provenance"),
                  selected_tasks=[ids[row["task_id"]] for row in prediction.get("related_tasks", [])],
                  used_tasks=[ids[source] for source in used])
    result["warnings"] = [{"advice": item["advice"], "sources": [ids[row["task_id"]] for row in item["sources"]]}
                          for item in prediction.get("cautions", [])]
    return result


def _mae(rows):
    if not rows:
        return {"count": 0, "user_mae_min": None, "initial_planner_mae_min": None}
    return {"count": len(rows),
            "user_mae_min": round(mean(abs(r["actual_min"] - r["user_min"]) for r in rows), 3),
            "initial_planner_mae_min": round(mean(abs(r["actual_min"] - r["initial_planner_min"]) for r in rows), 3)}


def _score(tasks, ids):
    scored = []
    for task in tasks:
        initial = task["creation_prediction"]
        if task["status"] == "completed" and task["actual_seconds"] > 0 and initial and "point_min" in initial:
            scored.append({"task": ids[task["id"]], "source": _kind(initial), "actual_min": task["actual_seconds"] / 60,
                           "user_min": task["estimate_minutes"], "initial_planner_min": initial["point_min"]})
    groups = {name: _mae([row for row in scored if row["source"] == name])
              for name in ("model", "fallback", "scheduled")}
    return {"overall": _mae(scored), "by_source": groups, "scored_tasks": scored}


def _capture(tasks, ids, previous):
    changes = {}
    for task in tasks:
        key = ids[task["id"]]
        state = {"status": task["status"], "recorded_min": round(task["actual_seconds"] / 60, 3)}
        if task["prediction"] and "point_min" in task["prediction"]:
            state["forecast"] = _forecast(task["prediction"], ids)
        if task["report"]:
            report = task["report"]
            state["report"] = {k: report[k] for k in ("summary", "elapsed_seconds", "switches", "totals", "warnings")}
        if key not in previous:
            state.update(title=task["title"], user_min=task["estimate_minutes"])
        delta = {field: value for field, value in state.items() if value != previous.get(key, {}).get(field)}
        if delta:
            changes[key] = delta
        previous.setdefault(key, {}).update(state)
    return changes


def replay(selector, output, *, actions=None, offline=False, anchor=None):
    history = histories(selector)[0]
    profile = history["id"]
    schedule_file = DATASET / "schedules" / (profile + ".json")
    schedule = json.loads(schedule_file.read_text(encoding="utf-8"))
    if actions is None:
        actions = len(schedule)
    if not 1 <= actions <= len(schedule):
        raise ValueError(f"Choose between 1 and {len(schedule)} actions for {profile}.")
    output = Path(output).resolve()
    if output.is_relative_to(DATASET.resolve()):
        raise ValueError("Choose a run directory outside data/tasks.")
    output.mkdir(parents=True, exist_ok=False)
    anchor = anchor or date.today()
    manifest = {"history": profile, "anchor_day": anchor.isoformat(), "mode": "offline" if offline else "configured",
                "requested_actions": actions, "total_actions": len(schedule),
                "schedule_sha256": hashlib.sha256(schedule_file.read_bytes()).hexdigest(),
                "fixture_sha256": hashlib.sha256((DATASET / "histories" / (profile + ".json")).read_bytes()).hexdigest(),
                "activitywatch_sha256": history.get("activitywatch_sha256")}
    if not offline:
        config = planner_settings()
        with forecast_endpoint() as endpoint:
            manifest["forecast_model"] = endpoint
        manifest["embedding_model"] = config.get("PLANNER_EMBEDDING_MODEL") or retrieval.EMBEDDING_MODEL
        manifest["decision_model"] = config.get("PLANNER_DECISION_MODEL") or retrieval.DECISION_MODEL
        manifest["relevance_threshold"] = retrieval.relevance_threshold(config)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    with TemporaryDirectory() as working, (output / "events.jsonl").open("x", encoding="utf-8") as log:
        store = PlannerStore(working)
        now = datetime.combine(anchor, time(12)).timestamp()
        dev_schedule.initialize(store, profile, DATASET, replay=True, now=now)
        store.clock = lambda: store.get_setting("dev_calendar").get("clock", now)
        ids, previous = {}, {}
        with _offline() if offline else nullcontext():
            predict = estimator.predict
            for number, event in enumerate(schedule[:actions], 1):
                jobs = []

                def record(task, history, index_dir):
                    prediction = predict(task, history, index_dir)
                    jobs.append((task["id"], [row["id"] for row in history], prediction))
                    return prediction

                with patch.object(estimator, "predict", record):
                    dev_schedule.advance(store, profile)
                if event["action"] == "create":
                    ids[store.get_setting("dev_task_" + event["task"])] = event["task"]
                tasks = store.list_tasks()
                entry = {"number": number, **event, "as_of": datetime.fromtimestamp(store.clock()).isoformat(),
                         "changes": _capture(tasks, ids, previous)}
                current = previous[event["task"]]
                entry.update(status=current["status"], recorded_min=current["recorded_min"],
                             forecast_jobs=[{"task": ids[target_id], "history_available": [ids[s] for s in history],
                                             "point_min": p["point_min"], "source": _kind(p)}
                                            for target_id, history, p in jobs])
                log.write(json.dumps(entry, ensure_ascii=False) + "\n")
                log.flush()
        result = {"history": profile, "actions_applied": actions, "schedule_complete": actions == len(schedule),
                  **_score(tasks, ids)}
        (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("history", help="History ID or two-digit prefix")
    parser.add_argument("--output", type=Path, required=True, help="A new directory for this run")
    parser.add_argument("--actions", type=int, help="Stop after this many actions (default: entire schedule)")
    parser.add_argument("--offline", action="store_true", help="Run the planner fallback with all model calls disabled")
    parser.add_argument("--anchor-day", type=date.fromisoformat, help="Replay day zero (YYYY-MM-DD, default: today)")
    args = parser.parse_args(argv)
    try:
        result = replay(args.history, args.output, actions=args.actions, offline=args.offline, anchor=args.anchor_day)
    except (ValueError, OSError) as error:
        parser.exit(1, f"{error}\n")
    print(json.dumps({"output": str(args.output), **result["overall"], "actions_applied": result["actions_applied"]}))


if __name__ == "__main__":
    main()
