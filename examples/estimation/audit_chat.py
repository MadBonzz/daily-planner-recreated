"""Run isolated analytics, daily-plan and document questions with the configured live model.

Run `python -m examples.estimation.audit_chat OUTPUT [--split development|holdout]`.
Each case uses the same task, log and clock snapshot for both models.
"""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import time
from unittest.mock import patch

import planner_analytics
import task_workspace
from model_client import ModelUnavailable, chat_json, model_settings
from pdf_reader import extract_pdf_text
from planner_analytics import analytics_rows, ask_analytics
from planner_chat import planning_context
from task_workspace import answer

ROOT = Path(__file__).resolve().parents[2]
PLAN_DAY = datetime(2026, 11, 13, 18, 0).astimezone()
PDF = ROOT / "data/tasks/pdfs"


def snapshot():
    """Three high-information rows plus one explicitly uncertain interval and an unrelated category."""
    rows = [
        ("g1", "Debug CUDA tail", "GPU", 35, 49, 11, "Recap A", "off_task", "sports.test", 180),
        ("g2", "Profile a second warp", "GPU", 20, 30, 12, "Recap B", "off_task", "sports.test", 240),
        ("g3", "Read occupancy notes", "GPU", 30, 27, 12, "Reference", "uncertain", "wiki.test", 480),
        ("h1", "Practice fugue", "Music", 30, 60, 12, "Interview", "off_task", "shows.test", 720),
    ]
    complete = []
    for tid, title, category, budget, actual, day, window, label, domain, seconds in rows:
        stamp = datetime(2026, 11, day, 13).timestamp()
        complete.append({"id": tid, "title": title, "description": title, "category": category,
                         "estimate_minutes": budget, "actual_seconds": 60*actual, "completed_at": stamp,
                         "report": {"intervals": [{"start": stamp, "seconds": seconds, "app": "Browser",
                                                   "domain": domain, "title": window, "label": label}]}})
    completed, logs = analytics_rows(complete, [])
    active = []
    for tid, title, budget, spent, status, day in [
        ("k", "Kernel lab", 60, 0, "ready", 13),
        ("f", "File receipts", 20, 5, "paused", 13),
        ("m", "Music journal", 40, 0, "ready", 14),
        ("x", "Completed poem", 5, 5, "completed", 13),
    ]:
        active.append({"id": tid, "title": title, "day": f"2026-11-{day:02}", "status": status,
                       "estimate_minutes": budget, "actual_seconds": spent * 60, "prediction": {"point_min": budget}})
    return completed, logs, planning_context(active, PLAN_DAY.date(), PLAN_DAY)


CASES = [
    # Gold comes from the concrete records and text above, with no synthetic "ideal time" labels.
    ("development", "gpu-distraction", "distraction", "What regularly distracts me during GPU tasks?", "sports.test"),
    ("development", "fit-20", "plan", "What can I finish today? I have 20 minutes.", (20, {"File receipts"})),
    ("development", "gpu-accuracy", "accuracy", "On GPU tasks, when I underestimate, by how many minutes on average?", 12),
    ("development", "ring-header", "pdf", "How many bytes is the ring header?", ("02-migration-lab/ring-port.pdf", r"\b(?:4|four)\s*[- ]?bytes?\b", True)),
    ("development", "ring-absent", "pdf", "What telephone number does the specification give for emergency support?", ("02-migration-lab/ring-port.pdf", "", False)),
    ("holdout", "time-yesterday", "count", "How many tasks did I complete yesterday?", 3),
    ("holdout", "music-distraction", "distraction", "What distracts me during Music tasks?", "shows.test"),
    ("holdout", "budget-bare", "plan", "I have 25 minutes. Which task fits?", (25, {"File receipts"})),
    ("holdout", "bare-half", "plan", "What can I do in half an hour?", (30, {"File receipts"})),
    ("holdout", "tile-count", "pdf", "How many consecutive lanes form one tile in this handout?", ("04-kernel-workbench/warp-tail.pdf", r"\b(?:32|thirty[- ]two)\b", True)),
    ("holdout", "tile-price", "pdf", "How much does the handout say the GPU costs in dollars?", ("04-kernel-workbench/warp-tail.pdf", "", False)),
    ("holdout", "journal-acks", "pdf", "How many signed acknowledgments does the front desk wait for?", ("01-replica-proofs/clock-two.pdf", r"\b(?:3|three)\b", True)),
    ("holdout", "wrong-page", "pdf", "According to page 2, what byte order is used for the event header?", ("02-migration-lab/ring-port.pdf", "", False)),
    ("holdout", "journal-timeout", "pdf", "What is the specified customer support email address for late tickets?", ("01-replica-proofs/clock-two.pdf", "", False)),
]


def check(case, data):
    split, key, kind, question, expected = case
    completed, logs, plan = data
    if kind == "pdf":
        filename, pattern, positive = expected
        document = PDF / filename
        text = extract_pdf_text(document.read_bytes())
        result = answer({"title": "Read " + document.stem}, question, document=document.name, text=text)
        has_answer = bool(result.get("sources"))
        gold_fact = not positive or bool(re.search(pattern, result["answer"], re.I))
        return {"pass": has_answer == positive and gold_fact, "gold_positive": positive,
                "predicted_positive": has_answer, "content_correct": gold_fact, "result": result}
    result = ask_analytics(question, completed, logs, plan=plan)
    if kind == "plan":
        minutes, gold = expected
        predicted = {r["Task"] for r in result.get("plan_rows", []) if r["Suggestion"].startswith("Fits")}
        actual = result.get("plan_window", {}).get("available_minutes")
        valid = actual == minutes and predicted == gold
    elif kind == "distraction":
        observed = {r.get("domain") for r in result.get("rows", [])}
        valid = observed == {expected} and expected in result["answer"]
        if key == "gpu-distraction":
            valid = valid and any(r.get("tasks") == 2 and r.get("minutes") == 7 for r in result["rows"])
    elif kind == "count":
        observed = [v for r in result.get("rows", []) for v in r.values()]
        valid = expected in observed and bool(re.search(r"\b(?:3|three)\b",result["answer"],re.I))
    else:
        valid = (any(expected in r.values() for r in result.get("rows", []))
                 and bool(re.search(r"\b(?:12|twelve)\b",result["answer"],re.I)))
    return {"pass": bool(valid), "result": result}


def score(records):
    """Use set membership for selected tasks/sites and a separate abstention test for PDF QA."""
    totals = {}
    labels = {case[1]: case for case in CASES}
    for key, row in records.items():
        split, _, kind, _, expected = labels[key]
        bucket = totals.setdefault(split, {})
        summary = bucket.setdefault(kind, {"cases": 0, "exact_passes": 0})
        summary["cases"] += 1
        summary["exact_passes"] += int(row["pass"])
        result = row.get("result", {})
        if kind == "pdf":
            if "predicted_positive" not in row:
                summary["unavailable"] = summary.get("unavailable", 0) + 1
                continue
            gold = {"answer"} if expected[-1] else set()
            predicted = {"answer"} if row["predicted_positive"] else set()
            candidates = {"answer"}
        elif kind == "plan":
            _, gold = expected
            predicted = {r["Task"] for r in result.get("plan_rows", []) if r["Suggestion"].startswith("Fits")}
            candidates = {"Kernel lab", "File receipts"} | predicted
        elif kind == "distraction":
            gold = {expected}
            predicted = set(re.findall(r"[a-z0-9.-]+\.test\b", result.get("answer", "").lower()))
            candidates = {"sports.test", "wiki.test", "shows.test"} | predicted
        else:
            continue
        counts = {"tp": len(gold & predicted), "fp": len(predicted - gold),
                  "fn": len(gold - predicted), "tn": len(candidates - (predicted | gold))}
        for field, n in counts.items():
            summary[field] = summary.get(field, 0) + n
    for bucket in totals.values():
        for summary in bucket.values():
            if "tp" in summary:
                for field, miss in (("precision", "fp"), ("recall", "fn")):
                    denominator = summary["tp"] + summary[miss]
                    summary[field] = round(summary["tp"] / denominator, 3) if denominator else None
    return totals


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--split", choices=("development", "holdout"))
    parser.add_argument("--score", action="store_true", help="Score existing records without calling any model")
    parser.add_argument("--case", action="append", help="Only run these case IDs")
    parser.add_argument("--cache", type=Path, help="Replay only identical successful requests to the same model")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    records = json.loads(args.output.read_text()) if args.output.exists() else {}
    if args.score:
        print(json.dumps(score(records), indent=2))
        return
    endpoint, model = model_settings()
    if args.cache and args.output.name.startswith("after-"):
        before = args.output.with_name(args.output.name.replace("after-", "before-", 1))
        if before.is_file():
            args.cache.mkdir(parents=True, exist_ok=True)
            for row in json.loads(before.read_text()).values():
                for call in row.get("model_calls", []):
                    if "response" in call:
                        (args.cache / (call["request_sha256"] + ".json")).write_text(json.dumps(call["response"]))
    data = snapshot()
    for case in CASES:
        split, key, kind, question, _ = case
        if key in records or (args.split and split != args.split) or (args.case and key not in args.case):
            continue
        start = time.monotonic()
        calls = []
        def observed(messages, schema, **options):
            payload = [endpoint, model, messages, schema, options]
            digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
            cached = (args.cache / f"{digest}.json") if args.cache else None
            hit = cached is not None and cached.exists()
            record = {"request_sha256": digest, "origin": "identical prior call" if hit else "live"}
            calls.append(record)
            try:
                response = json.loads(cached.read_text()) if hit else chat_json(messages, schema, **options)
            except ModelUnavailable as exc:
                record["error"] = str(exc)
                raise
            if cached is not None and not hit:
                cached.parent.mkdir(parents=True, exist_ok=True)
                cached.write_text(json.dumps(response))
            record["response"] = response
            return response
        with patch.object(planner_analytics, "chat_json", observed), patch.object(task_workspace, "chat_json", observed):
            try:
                row = check(case, data)
            except ModelUnavailable as exc:
                row = {"pass": False, "error": str(exc)}
        records[key] = {**row, "model": model, "model_calls": calls, "endpoint": endpoint, "split": split, "kind": kind,
                        "question": question, "seconds": round(time.monotonic() - start, 2)}
        temporary = args.output.with_suffix(".tmp")
        temporary.write_text(json.dumps(records, indent=2, ensure_ascii=False) + "\n")
        temporary.replace(args.output)
        print(key, row["pass"], flush=True)


if __name__ == "__main__":
    main()
