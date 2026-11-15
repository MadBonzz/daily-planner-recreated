from datetime import date

import pytest
from streamlit.testing.v1 import AppTest

from planner_store import PlannerStore


def create(store, title, **kwargs):
    return store.create_task(title=title, description="Task scope", estimate_minutes=20,
                             prediction={}, day=date.today(), **kwargs)


def test_archiving_an_active_task_stops_capture_and_preserves_its_data(tmp_path):
    stamp = 1000
    store = PlannerStore(tmp_path, clock=lambda: stamp)
    task = create(store, "Repair", attachments=[("instructions.pdf", b"%PDF-1.7 example")])
    store.transition(task["id"], "start")
    stamp += 40
    store.archive_task(task["id"])
    stamp += 60
    assert store.get_task(task["id"])["actual_seconds"] == 40
    assert store.periods(task["id"]) == store.capture_periods(task["id"]) == [(1000, 1040)]
    assert store.list_tasks() == store.list_tasks(date.today()) == []
    assert store.running_task() is store.pending_task() is None
    assert store.attachment_bytes(task["attachments"][0]["id"]).startswith(b"%PDF")
    assert store.archived_tasks()[0]["id"] == task["id"]
    assert not store.retry_estimate(task["id"])
    with pytest.raises(ValueError):
        store.transition(task["id"], "resume")


def test_soft_delete_preserves_learning_then_forget_cancels_jobs_and_removes_evidence(tmp_path):
    stamp = 1000
    store = PlannerStore(tmp_path, clock=lambda: stamp)
    past = create(store, "Secret test handout", attachments=[("private.pdf", b"%PDF-1.7 private")])
    source_id = past["id"]
    store.set_setting("dev_activity_" + source_id, {"task": "test"})
    store.transition(source_id, "start")
    stamp += 90
    store.transition(source_id, "stop")
    store.transition(source_id, "finish", completion_note="Private note", report={"intervals": [], "summary": "private"})
    store.archive_task(source_id)
    assert [t["id"] for t in store.completed_tasks()] == [source_id]
    prediction = {"point_min": 22, "confidence": "medium", "evidence": [],
                  "related_tasks": [{"task_id": source_id, "title": past["title"]}],
                  "model_rationale": "Private note"}
    next_task = store.create_task(title="Next task", description="Next", estimate_minutes=20,
                                  day=date.today(), prediction=prediction)
    assert store.retry_estimate(next_task["id"])
    leased = store.claim_prediction()
    assert [row["id"] for row in leased["history"]] == [source_id]
    store.delete_task(source_id)
    assert not store.resolve_prediction(next_task["id"], leased["token"], prediction)
    assert not store.claim_prediction()["history"]
    assert not store.archived_tasks() and not store.completed_tasks()
    assert not store.capture_periods(source_id) and not store.periods(source_id)
    assert store.get_setting("dev_activity_" + source_id) is None
    assert store.get_setting("memory_revision") == 1
    newer = store.get_task(next_task["id"])
    assert all('Private' not in str(newer[k]) and source_id not in str(newer[k])
               for k in ('prediction', 'creation_prediction'))
    with pytest.raises(ValueError):
        store.attachment_bytes(past["attachments"][0]["id"])
    assert store.pending_vector_purges() == [source_id]
    store.confirm_vector_purge(source_id)
    assert not store.pending_vector_purges()


def test_forgetting_a_comparison_refreshes_only_the_affected_open_forecast(tmp_path):
    store = PlannerStore(tmp_path)
    source = create(store, "Old draft")
    pred = {"point_min": 24, "evidence": [], "related_tasks": [{"task_id": source["id"], "title": source["title"]}]}
    target = store.create_task(title="Revise draft", description="Revise the document", estimate_minutes=20,
                               day=date.today(), prediction=pred)
    unrelated = create(store, "Sweep floor")
    store.delete_task(source["id"])
    assert store.get_task(target["id"])["estimate_updating"]
    assert not store.get_task(unrelated["id"])["estimate_updating"]
    job = store.claim_prediction()
    assert job["task"]["id"] == target["id"] and job["history"] == []


def test_card_trash_cancel_and_default_confirm(tmp_path):
    store = PlannerStore(tmp_path)
    task = create(store, "Inspect switch")
    source = f"""
from app import render_task
from planner_store import PlannerStore
from pathlib import Path
store=PlannerStore({str(tmp_path)!r})
for item in store.list_tasks():
    render_task(store,Path({str(tmp_path)!r})/'vectors',item,False)
"""
    app = AppTest.from_string(source).run()
    app.button(key="delete-" + task["id"]).click().run()
    assert app.checkbox(key="forget-" + task["id"]).value is False
    next(button for button in app.button if button.label == "No").click().run()
    assert store.list_tasks()
    app.button(key="delete-" + task["id"]).click().run()
    next(button for button in app.button if button.label == "Yes, delete").click().run()
    assert not app.exception and not store.list_tasks() and store.archived_tasks()
