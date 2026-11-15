"""Measure the five dashboard calendars against the original ten from native watcher stores."""

import argparse
from collections import Counter
from contextlib import closing, nullcontext
from datetime import date
import json
import os
from pathlib import Path
import sqlite3
from statistics import median
from tempfile import TemporaryDirectory
from unittest.mock import patch

import activity
import dev_workspace
import estimator
import retrieval
from examples.estimation import activity_fixture, run
from examples.estimation.replay import replay
from examples.estimation.scoring import pair_metrics
from model_client import ModelUnavailable
from pdf_reader import extract_pdf_text

DENSITY_FIELDS = ("timer_periods", "raw_watcher_records", "intervals", "distinct_foreground_contexts")


def distribution(values):
    return {"min": min(values), "median": median(values), "max": max(values)} if values else None


def minutes_by(rows):
    seconds = Counter()
    for key, duration in rows:
        seconds[key] += duration
    return {key: round(value / 60, 2) for key, value in sorted(seconds.items())}


def schedules(events, completed, open_ids):
    created, first_start, finished, days = {}, {}, {}, set()
    open_at_create, scheduled = set(), set()
    for item in events:
        name, action = item["task"], item["action"]
        minute = item["day_offset"] * 1440 + item["minute"]
        if action == "create":
            if name in created:
                raise ValueError(f"Multiple creations of {name}")
            if scheduled:
                open_at_create.add(name)
            created[name] = minute
            scheduled.add(name)
        else:
            if name not in scheduled or created[name] > minute:
                raise ValueError(f"{name}: timer action outside task lifetime")
            if action in ("start", "resume"):
                first_start.setdefault(name, minute)
                days.add(item["day_offset"])
            if action == "finish":
                finished[name] = minute
                scheduled.remove(name)
    if scheduled != open_ids or set(finished) != completed or not completed <= first_start.keys():
        raise ValueError("Calendar terminal states disagree with the historical/open task records")
    if [(e["day_offset"], e["minute"]) for e in events] != sorted((e["day_offset"], e["minute"]) for e in events):
        raise ValueError("Calendar is not in chronological order")
    leads = [first_start[k] - created[k] for k in completed]
    return {"event_count": len(events), "activity_days": len(days),
            "calendar_span_days": 1 + max(e["day_offset"] for e in events) - min(e["day_offset"] for e in events),
            "completed_create_to_first_start_minutes": distribution(leads),
            "completed_created_on_an_earlier_day": sum(first_start[k] // 1440 > created[k] // 1440 for k in completed),
            "created_while_other_task_unfinished": len(open_at_create)}


def targeted_evidence(history, directory):
    tasks = {row["id"]: row for row in history["tasks"]}
    targets = {}
    origins = {}
    if directory == run.DEMO_DATASET:
        origins = json.loads((directory / "sources.json").read_text())[history["id"]]
    for row in history["holdouts"]:
        rule = row.get("evaluation", {})
        ids = rule.get("evidence_ids", [])
        if origins:
            source = origins[row["id"]]
            archive = json.loads((run.DATASET / "histories" / (source["history"] + ".json")).read_text())
            original = next(t for t in archive["holdouts"] if t["id"] == source["task"])
            rule = original["evaluation"]
            ids = [key for key in tasks if origins.get(key, {}).get("history") == source["history"]
                   and origins[key]["task"] in rule["evidence_ids"]]
            if len(ids) != len(rule["evidence_ids"]):
                raise ValueError(f"{row['id']}: a labeled historical reference is missing from the demo")
        ratios = [tasks[key]["actual_minutes"] / tasks[key]["estimate_minutes"] for key in ids]
        targets[row["id"]] = {"title": row["title"], "budget": row["estimate_minutes"],
                              "fixed": bool(row.get("fixed_duration")), "expected_band": rule.get("expected_ratio_range"),
                              "historical_evidence": ids,
                              "past_actual_to_budget_median": round(median(ratios), 3) if ratios else None}
    return targets


def profile(path, directory):
    history = json.loads(path.read_text())
    run.validate(history, directory)
    events = json.loads((directory / "schedules" / path.name).read_text())
    historical = {row["id"]: row for row in history["tasks"]}
    open_tasks = history["open_tasks"]
    schedule = schedules(events, set(historical), {row["id"] for row in open_tasks})
    reports = activity_fixture.replay_history(history, directory=directory)
    database = directory / "activitywatch" / (path.stem + ".db")
    with closing(sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True)) as db:
        tables = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if not {"events", "buckets", "key_value"} <= tables or db.execute("PRAGMA user_version").fetchone() != (6,):
            raise ValueError(f"{database.name}: not an ActivityWatch v6 database")
        if db.execute("PRAGMA integrity_check").fetchone() != ("ok",) or db.execute("PRAGMA foreign_key_check").fetchone():
            raise ValueError(f"{database.name}: damaged watcher database")
        buckets = dict(db.execute("SELECT b.name, count(e.id) FROM buckets b LEFT JOIN events e ON b.id=e.bucketrow GROUP BY b.id"))
        records = list(db.execute("SELECT id, starttime, endtime FROM events"))
    tasks, labels, devices = {}, [], []
    off_task_domains, off_task_apps = Counter(), Counter()
    for key, replay in reports.items():
        row = historical[key]
        periods = [(round(a * 1e9), round(b * 1e9)) for a, b in replay["periods"]]
        hits = {ident for ident, start, end in records if any(start < b and end > a for a, b in periods)}
        intervals = replay["report"]["intervals"]
        actual_seconds = row["actual_minutes"] * 60
        if abs(sum(i["seconds"] for i in intervals) - actual_seconds) > .01:
            raise ValueError(f"{path.stem}/{key}: reported time differs from recorded periods")
        visible_secs = sum(i["seconds"] for i in intervals if activity.reviewable(i))
        labels.extend((i["label"], i["seconds"]) for i in intervals)
        devices.extend((activity.device_label(i), i["seconds"]) for i in intervals)
        off_task = [i for i in intervals if i["label"] == "off_task"]
        for event in off_task:
            if event.get("domain"):
                off_task_domains[event["domain"]] += event["seconds"] / 60
            if event.get("phone_app"):
                off_task_apps[event["phone_app"]] += event["seconds"] / 60
        contexts = {tuple(i.get(k) or "" for k in ("app", "domain", "title", "phone_app"))
                    for i in intervals if activity.reviewable(i)}
        tasks[key] = {"timer_periods": len(periods), "raw_watcher_records": len(hits), "intervals": len(intervals),
                      "distinct_foreground_contexts": len(contexts), "off_task_intervals": len(off_task),
                      "timed_minutes": row["actual_minutes"], "visible_foreground_minutes": round(visible_secs / 60, 2)}
    with_pdf = [t for t in [*history["tasks"], *open_tasks, *history["holdouts"]] if t.get("pdf")]
    documents = {run.pdf_file(t, history["id"], directory) for t in with_pdf}
    pdf_words = [len(extract_pdf_text(doc.read_bytes()).split()) for doc in sorted(documents)]
    return {"id": path.stem, "name": history["name"], "completed": len(historical),
            "open_by_status": dict(sorted(Counter(x["status"] for x in open_tasks).items())),
            "addable": len(history["holdouts"]), "schedule": schedule,
            "completed_with_note": sum(bool(r.get("completion_note", "").strip()) for r in historical.values()),
            "with_pdf": len(with_pdf), "unique_pdfs": len(documents), "words_per_pdf": distribution(pdf_words),
            "watcher_event_count": len(records), "watcher_buckets": buckets,
            "tasks_with_off_task_activity": sum(t["off_task_intervals"] > 0 for t in tasks.values()),
            "off_task_intervals": sum(t["off_task_intervals"] for t in tasks.values()),
            "off_task_domain_minutes": {k: round(v, 2) for k, v in sorted(off_task_domains.items())},
            "off_task_phone_app_minutes": {k: round(v, 2) for k, v in sorted(off_task_apps.items())},
            "all_labeled_minutes": minutes_by(labels), "all_device_minutes": minutes_by(devices),
            "tasks": tasks, "sample_add_evidence": targeted_evidence(history, directory)}


def summarize(profiles):
    tasks = [t for p in profiles for t in p["tasks"].values()]
    total = sum(t["timed_minutes"] for t in tasks)
    foreground = sum(t["visible_foreground_minutes"] for t in tasks)
    return {"calendars": len(profiles), "completed": len(tasks),
            "open": sum(sum(p["open_by_status"].values()) for p in profiles),
            "addable": sum(p["addable"] for p in profiles),
            "completed_with_note": sum(p["completed_with_note"] for p in profiles),
            "calendar_events": sum(p["schedule"]["event_count"] for p in profiles),
            "native_watcher_events": sum(p["watcher_event_count"] for p in profiles),
            "tasks_with_off_task_activity": sum(p["tasks_with_off_task_activity"] for p in profiles),
            "off_task_intervals": sum(p["off_task_intervals"] for p in profiles),
            "off_task_minutes": round(sum(p["all_labeled_minutes"].get("off_task", 0) for p in profiles), 2),
            "off_task_domains": sorted({k for p in profiles for k in p["off_task_domain_minutes"]}),
            "off_task_phone_apps": sorted({k for p in profiles for k in p["off_task_phone_app_minutes"]}),
            "per_completed_task": {key: distribution([t[key] for t in tasks]) for key in DENSITY_FIELDS},
            "without_visible_foreground": sum(t["visible_foreground_minutes"] == 0 for t in tasks),
            "timed_minutes": total, "visible_foreground_minutes": round(foreground, 2),
            "percentage_with_visible_foreground": round(100 * foreground / total, 1)}


def chronological_check(history_id, directory):
    """Execute the real event/worker loop, disabling model calls for this state and timing check."""
    with TemporaryDirectory(prefix="planner-demo-chronology-") as root:
        out = Path(root) / "run"
        result = replay(history_id, out, offline=True, anchor=date(2026, 11, 15), directory=directory)
        events = [json.loads(line) for line in (out / "events.jsonl").read_text().splitlines()]
    finished = set()
    sources = Counter()
    for event in events:
        if event["action"] == "finish":
            finished.add(event["task"])
        for job in event["forecast_jobs"]:
            if not set(job["history_available"]) <= finished:
                raise ValueError(f"{history_id}: forecast used a task before completion at event {event['number']}")
            sources[job["source"]] += 1
    if not result["schedule_complete"] or sources["model"]:
        raise ValueError(f"{history_id}: unexpected offline replay result")
    return {"actions": len(events), "completed": len(finished), "forecast_jobs": dict(sources),
            "anchor": "2026-11-15"}


def addable_retrieval(results):
    """Score production Qdrant search and Kev selections; no new model output can enter this probe."""
    labels = json.loads((run.REPO / "data/results/demo-histories/addable-labels.json").read_text())
    cases, pairs = {}, []
    model_settings = {"PLANNER_RELEVANCE_THRESHOLD": "0.50", "PLANNER_EMBEDDING_MODEL": retrieval.EMBEDDING_MODEL,
                      "PLANNER_DECISION_MODEL": retrieval.DECISION_MODEL}
    with TemporaryDirectory(prefix="planner-demo-retrieval-") as root, patch.dict(os.environ, model_settings):
        for profile in results["demo"]["profiles"]:
            name = profile["id"]
            store = dev_workspace.open_dev_workspace(Path(root), name)
            history = store.completed_tasks()
            keys = {store.get_setting("dev_task_" + k): k for k in profile["tasks"]}
            eligible = {keys[h["id"]] for h in history if not h["fixed_duration"]}
            cases[name] = {}
            for key, example in profile["sample_add_evidence"].items():
                task = dev_workspace.add_sample_task(store, name, key, "2026-11-15")
                screened = {}

                def search(*args, **kwargs):
                    ranked = retrieval.search(*args, **kwargs)
                    screened.update({keys[row["id"]]: round(score, 5) for row, score in ranked})
                    return ranked

                with (patch.object(estimator, "_search", search),
                      patch.object(estimator, "forecast_endpoint", return_value=nullcontext({"model": None, "provider": None})),
                      patch.object(estimator, "chat_json", side_effect=ModelUnavailable("Final model disabled in retrieval audit.")),
                      patch.object(retrieval, "_local_post", side_effect=AssertionError("Missing packaged model result"))):
                    prediction = estimator.predict(task, history, store.data_dir / "vectors")
                if prediction.get("retrieval_error"):
                    raise ValueError(f"{key}: {prediction['retrieval_error']}")
                selected = {keys[s["task_id"]] for s in prediction["related_tasks"]}
                expected = set(example["historical_evidence"]) | labels["additions"].get(key, {}).keys()
                neutral = set(labels["neutral"].get(key, {}))
                if not selected <= screened.keys() or not (expected | neutral) <= eligible or expected & neutral:
                    raise ValueError(f"{key}: retrieval or labels disagree with eligible candidates")
                for prior in sorted(eligible - neutral):
                    pairs.append({"expected": prior in expected, "qwen": prior in screened, "kev": prior in selected})
                cases[name][key] = {"eligible": len(eligible), "expected": sorted(expected), "neutral": sorted(neutral),
                                    "qwen_cosine": screened, "kev_selected": sorted(selected),
                                    "missed": sorted(expected - selected), "extra": sorted(selected - expected - neutral)}
    return {"settings": model_settings, "graded_pairs": len(pairs), "qwen": pair_metrics(pairs, lambda p: p["qwen"]),
            "qwen_and_kev": pair_metrics(pairs, lambda p: p["kev"]),
            "kev_given_retrieved": pair_metrics([p for p in pairs if p["qwen"]], lambda p: p["kev"]), "cases": cases}


def audit(with_replay):
    cohorts = {}
    for name, directory, expected in [("benchmark", run.DATASET, 10), ("demo", run.DEMO_DATASET, 5)]:
        files = sorted((directory / "histories").glob("*.json"))
        if len(files) != expected:
            raise ValueError(f"Expected {expected} {name} calendars, found {len(files)}")
        profiles = [profile(p, directory) for p in files]
        results = summarize(profiles)
        for p in profiles:
            p["per_completed_task"] = {key: distribution([t[key] for t in p["tasks"].values()]) for key in DENSITY_FIELDS}
            if name == "benchmark":
                p.pop("tasks")
                p.pop("sample_add_evidence")
            elif with_replay:
                p["offline_chronological_check"] = chronological_check(p["id"], directory)
        cohorts[name] = {"totals": results, "profiles": profiles}
    return cohorts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--replay", action="store_true", help="Replay all five schedules with model inference disabled.")
    parser.add_argument("--retrieval", action="store_true", help="Evaluate the 15 addable tasks with packaged Qwen/Kev.")
    args = parser.parse_args()
    output = args.output
    results = audit(args.replay)
    if args.retrieval:
        results["addable_retrieval"] = addable_retrieval(results)
    output.parent.mkdir(parents=True, exist_ok=True)
    run.write_json(output, results)
    for name in ("benchmark", "demo"):
        print(name, json.dumps(results[name]["totals"]), flush=True)
