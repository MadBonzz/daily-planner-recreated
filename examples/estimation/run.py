"""Explore and score the synthetic planner histories. Run as python -m examples.estimation.run."""

import argparse
import hashlib
from collections import Counter
from datetime import date, datetime, timezone
import json
import math
import os
from pathlib import Path
import re
from statistics import mean, median
import subprocess
import sys
from tempfile import TemporaryDirectory
import time
from unittest.mock import patch
from urllib.parse import urlsplit

import estimator
from examples.estimation.scoring import grade, rubric_summary, validate_rubric
from model_client import ModelUnavailable, model_settings
from planner_store import PlannerStore


REPO = Path(__file__).resolve().parents[2]
DATASET = REPO / "data" / "tasks"
INPUT_KEYS = ("day", "title", "description", "category", "details", "estimate_minutes")


def documents(task, history_id, directory):
    if not task.get("pdf"):
        return []
    if not valid_id(history_id):
        raise ValueError("Invalid fixture history ID")
    stem = task["pdf"] if isinstance(task["pdf"], str) else task["id"]
    if not valid_id(stem):
        raise ValueError("Invalid fixture PDF reference")
    path = directory / "pdfs" / history_id / f"{stem}.pdf"
    return [(path.name, path.read_bytes())]


def task_inputs(task, history_id, directory):
    """Only pre-task fields are permitted here. Outcome, sessions and 'check' stay in the fixture."""
    files = documents(task, history_id, directory)
    fields = {key: task[key] for key in INPUT_KEYS}
    if "fixed_duration" in task:
        fields["fixed_duration"] = task["fixed_duration"]
    fields["pdf_text"] = "\n".join(estimator.extract_pdf_text(data) for _, data in files)
    return fields, files


def valid_id(value):
    return isinstance(value, str) and re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]*", value)


def positive_minutes(value):
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def validate(history, directory, *, check_pdfs=True):
    history_id = history["id"]
    if not valid_id(history_id) or not history["name"] or not history["scenario"]:
        raise ValueError("Each history needs an ID, name, and scenario")
    tasks, holdouts = history["tasks"], history["holdouts"]
    if not 5 <= len(tasks) <= 10 or not holdouts:
        raise ValueError(f"{history_id}: expected 5–10 historical tasks and at least one holdout")
    seen = set()
    for task in tasks + holdouts:
        task_id = task["id"]
        if not valid_id(task_id) or task_id in seen:
            raise ValueError(f"{history_id}: invalid or duplicate task ID {task_id!r}")
        seen.add(task_id)
        day = date.fromisoformat(task["day"])
        if day.isoformat() != task["day"]:
            raise ValueError(f"{task_id}: use a YYYY-MM-DD date")
        if not all(isinstance(task[key], str) for key in ("title", "description", "category", "details")):
            raise ValueError(f"{task_id}: invalid task text")
        if not task["title"].strip() or not task["description"].strip():
            raise ValueError(f"{task_id}: a title and description are required")
        if "completion_note" in task and not isinstance(task["completion_note"], str):
            raise ValueError(f"{task_id}: completion_note must be text")
        if "fixed_duration" in task and type(task["fixed_duration"]) is not bool:
            raise ValueError(f"{task_id}: fixed_duration must be a boolean")
        sessions = task["sessions_minutes"]
        if not isinstance(sessions, list) or not sessions or any(
            type(n) is not int or n <= 0 for n in [*sessions, task["actual_minutes"], task["estimate_minutes"]]
        ) or sum(sessions) != task["actual_minutes"]:
            raise ValueError(f"{task_id}: invalid timings or sessions that do not sum to actual_minutes")
        if task.get("pdf") and check_pdfs:
            fields, _ = task_inputs(task, history_id, directory)
            if not fields["pdf_text"].strip():
                raise ValueError(f"{history_id}/{task_id}: attachment has no readable text")
    open_rows = history.get("open_tasks", [])
    if sum(row["status"] == "running" for row in open_rows) > 1:
        raise ValueError(f"{history_id}: only one open task can be running")
    for row in open_rows:
        task_id = row["id"]
        if not valid_id(task_id) or task_id in seen:
            raise ValueError(f"{history_id}: invalid or duplicate open task ID {task_id!r}")
        seen.add(task_id)
        if date.fromisoformat(row["day"]).isoformat() != row["day"]:
            raise ValueError(f"{task_id}: use a YYYY-MM-DD date")
        if "fixed_duration" in row and type(row["fixed_duration"]) is not bool:
            raise ValueError(f"{task_id}: fixed_duration must be a boolean")
        if row["status"] not in {"ready", "paused", "running"}:
            raise ValueError(f"{task_id}: invalid open task status")
        if (not all(isinstance(row.get(k), str) for k in ("title", "description", "category", "details"))
                or not row["title"].strip() or not row["description"].strip()):
            raise ValueError(f"{task_id}: open task needs a title and description")
        if not positive_minutes(row["estimate_minutes"]):
            raise ValueError(f"{task_id}: invalid open task budget")
        sessions = row.get("sessions_minutes", [])
        if (not isinstance(sessions, list) or not all(positive_minutes(v) for v in sessions)
                or (row["status"] == "ready" and sessions) or (row["status"] == "paused" and not sessions)):
            raise ValueError(f"{task_id}: invalid previously recorded open-task sessions")
        if row.get("activity") and check_pdfs:
            from examples.estimation.activity_fixture import replay_history
            replay_history({"id": history_id, "tasks": [row]})
        if row.get("pdf") and check_pdfs:
            if not task_inputs(row, history_id, directory)[0]["pdf_text"].strip():
                raise ValueError(f"{task_id}: attachment has no readable text")
    graded = any("evaluation" in row for row in holdouts)
    for row in holdouts:
        if graded and "evaluation" not in row:
            raise ValueError(f"{history_id}: every holdout needs its declared evaluation")
        if "evaluation" in row:
            validate_rubric(row["evaluation"], {past["id"] for past in tasks})
    if any(not row.get("check") and not row.get("evaluation") for row in holdouts):
        raise ValueError(f"{history_id}: each holdout needs a check describing its intended test")
    if min(row["day"] for row in holdouts) <= max(row["day"] for row in tasks):
        raise ValueError(f"{history_id}: every holdout must be later than all historical tasks")


def histories(selector, directory=DATASET):
    files = sorted((directory / "histories").glob("*.json"))
    all_histories = [json.loads(path.read_text(encoding="utf-8")) for path in files]
    if not all_histories:
        raise ValueError("No histories found")
    ids = [history["id"] for history in all_histories]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate history ID")
    if selector is None:
        return all_histories
    matches = [h for h in all_histories if selector == h["id"] or selector == h["id"].split("-")[0]]
    if len(matches) != 1:
        raise ValueError(f"Unknown or ambiguous history {selector!r}. Available: {', '.join(ids)}")
    return matches


def historical_rows(history, directory):
    from examples.estimation.activity_fixture import replay_history
    reports = replay_history(history, directory=directory)
    rows = []
    for task in sorted(history["tasks"], key=lambda row: row["day"]):
        fields, _ = task_inputs(task, history["id"], directory)
        rows.append({**fields, "id": task["id"], "actual_seconds": task["actual_minutes"] * 60,
                     "status": "completed", "completion_note": task.get("completion_note", ""),
                     "report": reports.get(task["id"], {}).get("report")})
    return rows


def category_prediction(task, completed):
    category = task["category"].strip().casefold()
    ratios = [past["actual_seconds"] / (60 * past["estimate_minutes"]) for past in completed
              if category and past["category"].strip().casefold() == category]
    return max(1, round(task["estimate_minutes"] * median(ratios))) if ratios else task["estimate_minutes"]


def evaluate(history, directory, index_dir, *, offline=False, case=None):
    completed = historical_rows(history, directory)
    selected = [row for row in history["holdouts"] if case is None or case == row["id"]]
    if not selected:
        raise ValueError(f"{history['id']}: no holdout matches {case!r}")
    scores = []
    request = estimator.chat_json
    for row in selected:
        fields, _ = task_inputs(row, history["id"], directory)
        category_min = category_prediction(fields, completed)
        outcome = "not_called"
        model_seconds = 0.0
        model_error = None
        model_calls = []

        def observed_chat(*args, **kwargs):
            nonlocal outcome, model_seconds, model_error
            if offline:
                outcome = "offline"
                raise ModelUnavailable("Offline fixture run.")
            call_started = time.perf_counter()
            messages = args[0] if args else kwargs["messages"]
            prompt_bytes = json.dumps(messages, ensure_ascii=False, sort_keys=True).encode("utf-8")
            receipt = {
                "prompt_sha256": hashlib.sha256(prompt_bytes).hexdigest(),
                "max_tokens": kwargs.get("max_tokens"),
            }
            model_calls.append(receipt)
            try:
                reply = request(*args, **kwargs)
            except ModelUnavailable as exc:
                outcome = "unavailable"
                receipt["error"] = str(exc)
                model_error = str(exc)
                raise
            finally:
                duration = time.perf_counter() - call_started
                model_seconds += duration
                receipt["seconds"] = round(duration, 3)
            receipt["response"] = reply
            outcome = "ok_after_retry" if model_error is not None else "ok"
            model_error = None
            return reply

        # This scoped test hook exercises the production fallback without contacting a provider.
        # The CLI is single-threaded and restores the client after every prediction.
        started = time.perf_counter()
        with patch.object(estimator, "chat_json", observed_chat):
            prediction = estimator.predict(fields, completed, index_dir)
        latency = time.perf_counter() - started
        actual = row["actual_minutes"]
        fields_to_score = {}
        if "evaluation" in row:
            fields_to_score = {
                "fixed_duration": bool(fields.get("fixed_duration")),
                "cautions": prediction.get("cautions", []),
                "explanation": prediction.get("explanation", ""),
                "model_calls": model_calls,
                "evaluation": grade(row["evaluation"], prediction, fields["estimate_minutes"], actual,
                                    {r["id"] for r in completed}),
            }
        scores.append({
            **fields_to_score,
            "history": history["id"], "task": row["id"], "title": row["title"],
            "original": fields["estimate_minutes"], "category": category_min,
            "planner": prediction["point_min"], "actual": actual,
            "range": [prediction["low_min"], prediction["high_min"]],
            "covered": prediction["low_min"] <= actual <= prediction["high_min"],
            "confidence": prediction["confidence"], "basis": prediction["basis"],
            "model_status": outcome, "model_error": model_error, "model_seconds": round(model_seconds, 3),
            "latency_seconds": round(latency, 3),
            "evidence": prediction["evidence"], "check": row.get("check", ""),
        })
    return scores


def model_succeeded(row):
    if row.get("fixed_duration") and row["basis"] == "Scheduled duration":
        return row["model_status"] == "not_called"
    return row["model_status"] in {"ok", "ok_after_retry"}


def metrics(scores):
    return {
        "cases": len(scores),
        **rubric_summary(scores),
        "mae_minutes": {method: round(mean(abs(row[method] - row["actual"]) for row in scores), 2)
                        for method in ("original", "category", "planner")},
        "ranges_covering_actual": sum(row["covered"] for row in scores),
        "mean_range_width_minutes": round(mean(row["range"][1] - row["range"][0] for row in scores), 2),
        "basis_counts": dict(Counter(row["basis"] for row in scores)),
        "model_status_counts": dict(Counter(row["model_status"] for row in scores)),
        "mean_prediction_seconds": round(mean(row["latency_seconds"] for row in scores), 3),
    }


def populate_profile(history, directory, root):
    """Seed a new profile with real store transitions and attach synthetic PDFs."""
    if (root / "live").exists():
        raise ValueError("Choose a new, empty fixture profile")
    from dev_workspace import seed_completed
    store = PlannerStore(root / "live")
    seed_completed(store, history, directory)
    completed = store.completed_tasks()
    for row in history["holdouts"]:
        fields, files = task_inputs(row, history["id"], directory)
        # Stage each benchmark input on today's screen. Only its published description
        # and details enter the estimator; the hidden outcome remains in the JSON.
        fields["day"] = date.today().isoformat()
        prediction = estimator.predict(fields, completed, store.data_dir / "vectors")
        store.create_task(**fields, attachments=files, prediction=prediction)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("list", help="List all synthetic histories")
    commands.add_parser("validate", help="Check dates, session totals, and readable PDFs")
    score = commands.add_parser("evaluate", help="Score held-out tasks against fixed earlier history")
    score.add_argument("history", nargs="?", help="History ID or two-digit prefix; default: all")
    score.add_argument("--case", help="Score one held-out task ID in the chosen history")
    score.add_argument("--offline", action="store_true", help="Exercise the planner's no-model fallback")
    score.add_argument("--strict-model", action="store_true",
                       help="Exit nonzero after saving results if any model call was unavailable")
    score.add_argument("--json", type=Path, help="Save machine-readable case details and summary")
    launch = commands.add_parser("open", help="Launch the actual Streamlit app with an isolated synthetic profile")
    launch.add_argument("history", help="History ID or two-digit prefix")
    launch.add_argument("--port", type=int, default=8502)
    args = parser.parse_args(argv)
    if getattr(args, "strict_model", False) and args.offline:
        parser.error("--strict-model requires an online run")
    if getattr(args, "case", None) and not args.history:
        parser.error("--case requires a history")
    try:
        selected = histories(getattr(args, "history", None))
        if args.command == "list":
            for history in selected:
                print(f"{history['id']}  {history['name']}  ({len(history['tasks'])} past / {len(history['holdouts'])} held out)")
            return 0
        for history in selected:
            validate(history, DATASET, check_pdfs=args.command == "validate")
        if args.command == "validate":
            from examples.estimation.activity_fixture import replay_history
            for history in selected:
                replay_history(history, directory=DATASET)
            print(f"Validated {len(selected)} histories, activity sessions, holdout rubrics, dates, and attachments.")
            return 0
        with TemporaryDirectory(prefix="planner-synthetic-") as temp:
            root = Path(temp)
            if args.command == "open":
                history = selected[0]
                print(f"Synthetic workspace: {history['name']}. Preparing today's holdouts…", flush=True)
                populate_profile(history, DATASET, root)
                env = dict(os.environ, PLANNER_DATA_DIR=str(root))
                print("Open My planner; both held-out tasks are staged on today’s date. Stop the app to discard this trial.", flush=True)
                return subprocess.call([sys.executable, "-m", "streamlit", "run", "app.py",
                                        "--server.port", str(args.port)], cwd=REPO, env=env)
            started_at = datetime.now(timezone.utc).isoformat()
            started = time.perf_counter()
            results = []
            for history in selected:
                rows = evaluate(history, DATASET, root / history["id"], offline=args.offline, case=args.case)
                for row in rows:
                    print(f"{row['history']}/{row['task']}  original {row['original']}  category {row['category']}  "
                          f"planner {row['planner']} [{row['range'][0]}, {row['range'][1]}]  actual {row['actual']} min  "
                          f"({row['basis']}, {row['model_status']})", flush=True)
                results.extend(rows)
            summary = metrics(results)
            endpoint, model = model_settings()
            # Keep embedded endpoint userinfo and query parameters out of public results.
            address = urlsplit(endpoint)
            host = address.hostname or ""
            if ":" in host:
                host = f"[{host}]"
            if address.port:
                host = f"{host}:{address.port}"
            config = {
                "mode": "offline_fallback" if args.offline else "configured_model",
                "model": None if args.offline else model,
                "endpoint": None if args.offline else f"{address.scheme}://{host}{address.path}",
                "strict_model": args.strict_model,
                "started_at_utc": started_at, "ended_at_utc": datetime.now(timezone.utc).isoformat(),
                "elapsed_seconds": round(time.perf_counter() - started, 3),
                "model_seconds": round(sum(row["model_seconds"] for row in results), 3),
            }
            print(json.dumps(summary, indent=2))
            if args.json:
                args.json.parent.mkdir(parents=True, exist_ok=True)
                args.json.write_text(json.dumps({"configuration": config, "cases": results, "summary": summary}, indent=2,
                                                ensure_ascii=False) + "\n", encoding="utf-8")
            if args.strict_model and not all(model_succeeded(row) for row in results):
                print("Model failed on at least one case; fallback scores are marked in the results.", file=sys.stderr)
                return 1
        return 0
    except KeyboardInterrupt:
        return 130
    except (FileNotFoundError, KeyError, ValueError) as error:
        parser.exit(2, f"Fixture error: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
