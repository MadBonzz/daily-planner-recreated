"""Opt-in end-to-end checks against the configured, running forecast endpoint."""
from datetime import date, datetime, timedelta
import os

import pytest

import chat_tasks
import dev_workspace
import estimate_worker
from planner_analytics import ask_analytics
from planner_chat import planning_context
from planner_store import PlannerStore


pytestmark = pytest.mark.skipif(os.getenv("PLANNER_TEST_LIVE_HISTORY") != "1",
                                reason="Requires a running forecast endpoint")


def test_nora_forecasts_and_sequential_completions(tmp_path, monkeypatch):
    monkeypatch.setenv("PLANNER_PREFER_PRECOMPUTED", "1")
    store = dev_workspace.open_dev_workspace(tmp_path, "d1-systems-student")
    day = store.get_setting("dev_calendar", {})["day"]
    cards = {task["title"]: task["id"] for task in store.list_tasks(day)}

    def process():
        assert estimate_worker.run_one_job(store, store.data_dir / "vectors")

    def model_minutes(task_id):
        prediction = store.get_task(task_id)["prediction"]
        assert prediction.get("provenance", {}).get("minutes_by") == "model", prediction.get("model_error")
        assert 0 < prediction["low_min"] <= prediction["point_min"] <= prediction["high_min"]
        return prediction

    while store.daily_forecast_totals(day)["pending"]:
        process()
    assert len(cards) == 6
    for title, task_id in cards.items():
        print(title, model_minutes(task_id)["point_min"], flush=True)
    target = cards["Attach the minimized trace to the lab issue"]
    for title, minutes, related in [("Pick up dish soap", 20, False), ("Minimize the marker-word failure", 110, True)]:
        before = model_minutes(target)
        source = cards[title]
        if store.get_task(source)["status"] == "ready":
            store.transition(source, "start")
            store.transition(source, "pause")
        store.transition(source, "finish", spent_seconds=minutes * 60)
        store = dev_workspace.open_dev_workspace(tmp_path, "d1-systems-student")
        while any(task["awaiting_related_update"] or task["estimate_updating"]
                  for task in store.list_tasks(day) if task["status"] != "completed"):
            process()
        after = model_minutes(target)
        if related:
            assert source in {item["task_id"] for item in after["related_tasks"]}
        else:
            assert after == before
        print(f"After completing {title}: {before['point_min']} -> {after['point_min']} min", flush=True)
    assert store.daily_forecast_totals(day)["done"] == 2


def test_chat_create_attach_edit_and_postpone(tmp_path):
    store = PlannerStore(tmp_path)
    history = []

    def submit(question, files=None):
        tasks = store.list_tasks()
        plan = planning_context(store.list_tasks(date.today()), date.today(), datetime.now())
        reply = ask_analytics(question, [], [], history, plan=plan, files=files or {},
                              create_categories=["Study"], task_catalog=tasks)
        assert "new_task" in reply or "task_change" in reply, reply
        result = chat_tasks.apply_action(store, reply, files or {}, question)
        assert result.get("affected_task_id"), result
        history.append({**result, "question": question})
        return result["affected_task_id"]

    scope = "it has theoretical questions about consensus, Paxos and broadcast"
    upload = {"name": "assignment.txt", "kind": "text", "data": b"Prove agreement for Paxos.", "current": True}
    task_id = submit("Can you add a task for my distributed computing assignment, I expect it will take 4 hours, "
                     + scope + ". Attach assignment.txt to this task.", {"1": upload})
    task = store.get_task(task_id)
    assert task["estimate_minutes"] == 240 and task["description"] == scope
    assert store.attachment_bytes(task["attachments"][0]["id"]) == upload["data"]
    assert submit("Edit that task and change its description to Prove agreement under a crash failure") == task_id
    assert store.get_task(task_id)["description"] == "Prove agreement under a crash failure"
    assert submit("Postpone that task to tomorrow") == task_id
    assert store.get_task(task_id)["day"] == (date.today() + timedelta(days=1)).isoformat()
