"""User-facing scenarios for task creation, attachment ownership and concurrent editing."""
from datetime import date
import json

import pytest
from streamlit.testing.v1 import AppTest

import app
import planner_analytics as analytics
from planner_store import PlannerStore
import task_workspace as desk


def edit(store, task, **changes):
    fields = {key: task[key] for key in ("title", "description", "category", "details", "fixed_duration", "estimate_minutes")}
    return store.edit_task(task["id"], expected_revision=task["edit_revision"], **{**fields, **changes})


def create(store, title, **changes):
    return store.create_task(title=title, description="Initial scope", estimate_minutes=30,
                             day=date.today(), **changes)


def test_edit_while_estimating_rejects_stale_workers_and_preserves_timer(tmp_path):
    store = PlannerStore(tmp_path)
    task = create(store, "Write proof", attachments=[("old.pdf", b"%PDF-old")])
    original = store.claim_prediction()
    store.transition(task["id"], "start")
    new = edit(store, task, description="Analyze the traces instead", remove_ids=[task["attachments"][0]["id"]],
               attachments=[("traces.csv", b"duration\n10")])
    assert new["id"] == task["id"] and new["status"] == "running"
    assert [item["name"] for item in new["attachments"]] == ["traces.csv"]
    store.cache_document_text(task["id"], "old content", expected_revision=task["edit_revision"])
    assert not store.resolve_prediction(task["id"], original["token"], {"point_min": 99}, pdf_text="old content")
    assert store.get_task(task["id"])["pdf_text"] == ""
    with pytest.raises(ValueError, match="changed"):
        edit(store, task, title="Stale tab")
    with pytest.raises(ValueError):
        edit(store, new, title="should roll back", attachments=[("invalid.pdf", b"garbage")])
    job = store.claim_prediction()
    assert job["task"]["description"] == "Analyze the traces instead"
    assert store.resolve_prediction(task["id"], job["token"], {"point_min": 35})
    assert store.get_task(task["id"])["creation_prediction"] == {"point_min": 35}


def test_edit_rejudges_only_affected_pairs_and_preserves_initial_estimate(tmp_path):
    store = PlannerStore(tmp_path)
    first, second, third = [create(store, title, prediction={"point_min": 25}) for title in ("Read A", "Read B", "Walk")]
    while pair := store.claim_relation():
        store.resolve_relation(pair["pair"], pair["token"], True)
    updated = edit(store, first, description="Walk to the library", fixed_duration=True, estimate_minutes=15)
    assert updated["prediction"]["point_min"] == 15
    assert updated["creation_prediction"]["point_min"] == 25
    with store._db() as db:
        after = {(r["left_id"], r["right_id"]): r["related"] for r in db.execute("SELECT * FROM task_relations")}
    assert after[tuple(sorted((second["id"], third["id"])))] == 1
    assert all(val is None for pair, val in after.items() if first["id"] in pair)
    claimed = store.claim_relation()
    edit(store, updated, details="Include the return trip")
    assert not store.resolve_relation(claimed["pair"], claimed["token"], True)


def test_shared_chat_can_attach_selected_file_across_pages_and_edit_card(tmp_path, monkeypatch):
    monkeypatch.setattr(app, "DATA_ROOT", tmp_path)
    monkeypatch.setattr(app, "schedule_pending", lambda *_: None)
    monkeypatch.setattr(app, "schedule_reports", lambda *_: None)
    monkeypatch.setattr(desk, "availability", lambda *_: {"eligible": False})
    def respond(messages, schema, **kwargs):
        payload = json.loads(messages[-1]["content"])
        fields = schema["properties"]
        if "intent" in fields:
            return {"intent": "create_task"}
        if "action" not in fields:
            assert "momentum" in payload["files"][0]["excerpt"]
            return {"answer": "The reading notes discuss momentum."}
        assert [f["name"] for f in payload["files"]] == ["notes.md", "errands.txt"]
        assert not any(f["uploaded_this_turn"] for f in payload["files"])
        return {"action": "create", "title": "Read the notes", "description": "Review momentum.",
                "details": "", "category": "Study", "day": payload["selected_day"], "fixed_duration": False,
                "estimate_minutes": 30, "when": "", "estimate_phrase": "half-hour", "file_ids": ["1"], "reply": ""}
    monkeypatch.setattr(analytics, "chat_json", respond)
    page = AppTest.from_string("from app import main\nmain()", default_timeout=15).run()
    widget = page.chat_input[0].set_value("Please summarize notes.md")
    widget.upload("notes.md", b"Study momentum", "text/markdown").upload("errands.txt", b"Buy apples", "text/plain")
    page.run()
    assert not page.exception and not PlannerStore(tmp_path / "live").list_tasks()
    next(v for v in page.get("segmented_control") if v.key == "planner_view").set_value("Analytics & logs").run()
    page.chat_input[0].set_value("Add a half-hour reading task using notes.md only.").run()
    assert not page.exception
    store = PlannerStore(tmp_path / "live")
    task, = store.list_tasks()
    doc, = task["attachments"]
    assert task["estimate_minutes"] == 30 and store.attachment_bytes(doc["id"]) == b"Study momentum"
    next(v for v in page.get("segmented_control") if v.key == "planner_view").set_value("Daily plan").run()
    assert not any(b.label == "Move task" for b in page.button)
    page.button(key=f"edit-task-{task['id']}").click().run()
    assert not page.exception
    next(w for w in page.text_area if w.label == "Description").set_value("Read the derivation.")
    page.get("file_uploader")[0].set_value([("appendix.txt", b"Extra reading", "text/plain")])
    next(b for b in page.button if b.label == "Save changes").click().run()
    assert not page.exception
    saved, = store.list_tasks()
    assert saved["id"] == task["id"] and saved["description"] == "Read the derivation."
    assert [d["name"] for d in saved["attachments"]] == ["notes.md", "appendix.txt"]


def test_chat_edits_and_postpones_the_chosen_task_in_streamlit(tmp_path, monkeypatch):
    from datetime import timedelta
    monkeypatch.setattr(app, "DATA_ROOT", tmp_path)
    monkeypatch.setattr(app, "schedule_pending", lambda *_: None)
    monkeypatch.setattr(app, "schedule_reports", lambda *_: None)
    monkeypatch.setattr(desk, "availability", lambda *_: {"eligible": False})
    store = PlannerStore(tmp_path / "live")
    first = create(store, "Lab setup")
    second = create(store, "Lab analysis")
    tomorrow = (date.today() + timedelta(days=1)).isoformat()

    def respond(messages, schema):
        payload = json.loads(messages[-1]["content"])
        if "intent" in schema["properties"]:
            return {"intent": "edit_task"}
        if "edits" in schema["properties"]:
            assert "results.csv" in payload["uploads"].values()
            return {"target": "Lab analysis", "when": "", "day": "",
                    "edits": [{"field": "description", "value": "Compare timings"},
                              {"field": "estimate_minutes", "value": "45"}],
                    "file_ids": ["1"], "remove_names": []}
        when = "tomorrow" if "tomorrow" in payload["request"] else ""
        return {"target": "Lab", "when": when, "day": date.today().isoformat()}

    monkeypatch.setattr(analytics, "chat_json", respond)
    page = AppTest.from_string("from app import main\nmain()", default_timeout=15).run()
    message = page.chat_input[0].set_value("Edit Lab analysis: set the description to Compare timings, "
                                           "allow 45 minutes and attach results.csv.")
    message.upload("results.csv", b"ms\n41", "text/csv")
    page.run()
    assert not page.exception
    saved = store.get_task(second["id"])
    assert (saved["description"], saved["estimate_minutes"], saved["fixed_duration"]) == ("Compare timings", 45, False)
    assert [d["name"] for d in saved["attachments"]] == ["results.csv"]
    assert store.get_task(first["id"])["description"] == "Initial scope"
    store.transition(second["id"], "start")
    next(v for v in page.get("segmented_control") if v.key == "planner_view").set_value("Analytics & logs").run()
    page.chat_input[0].set_value("Postpone Lab to tomorrow").run()
    turn = page.session_state["planner_chat"]["turns"][-1]
    assert set(turn["choices"]) == {first["id"], second["id"]} and not turn.get("affected_task_id")
    assert store.get_task(second["id"])["status"] == "running"
    page.chat_input[0].set_value(str(turn["choices"].index(second["id"]) + 1)).run()
    assert not page.exception
    assert store.get_task(first["id"])["day"] == date.today().isoformat()
    saved = store.get_task(second["id"])
    assert saved["day"] == tomorrow and saved["status"] == "paused"
    assert saved["description"] == "Compare timings"
    assert page.session_state["planner_chat"]["turns"][-1]["affected_task_id"] == second["id"]


def test_underspecified_postpone_and_repeated_or_stale_chat_actions(tmp_path):
    from datetime import datetime, timedelta
    from chat_tasks import apply_action, handle_request
    store = PlannerStore(tmp_path)
    task = create(store, "Review")
    plan = {"now": datetime.now(), "selected_day": task["day"]}
    day = (date.today() + timedelta(days=3)).isoformat()
    pending = handle_request("postpone_task", "Postpone Review", [], {}, plan, [],
                             lambda *_: {"target": "Review", "when": "tomorrow", "day": day}, [task])
    assert "task_change" not in pending and "What date" in pending["answer"]
    extracted = {"target": "Review", "when": "", "day": "", "edits": [], "file_ids": ["1"], "remove_names": []}
    files = {"1": {"name": "report.txt", "data": b"notes"}}
    action = handle_request("edit_task", "Edit Review and attach report.txt", [], files, plan, [],
                            lambda *_: extracted, [task])
    first = apply_action(store, action, files, "edit-1")
    assert apply_action(store, action, files, "edit-1") == first
    assert len(store.get_task(task["id"])["attachments"]) == 1
    with pytest.raises(ValueError, match="changed"):
        apply_action(store, action, files, "another")
    with pytest.raises(ValueError, match="changed"):
        store.postpone(task["id"], day, previous_day=task["day"], expected_revision=0)


def test_chat_grounds_tomorrow_and_explicit_pronouns_even_if_model_misreads_them(tmp_path):
    from datetime import datetime, timedelta
    from chat_tasks import handle_request
    store = PlannerStore(tmp_path)
    task = create(store, "Optimization notes")
    plan = {"now": datetime.now(), "selected_day": date.today().isoformat()}
    extracted = {"title": "Optimization notes", "description": "Compare steps", "details": "",
                 "estimate_minutes": 30, "estimate_phrase": "half an hour", "category": "", "file_ids": [],
                 "fixed_duration": False, "action": "create", "when": "", "day": task["day"], "reply": ""}
    proposed = handle_request("create_task", "Add a task for tomorrow to compare steps, half an hour",
                              [], {}, plan, [], lambda *_: extracted)
    assert proposed["new_task"]["day"] == (date.today() + timedelta(days=1)).isoformat()
    edit = {"target": "Compare two step sizes", "when": "", "day": "", "file_ids": ["1"], "remove_names": ["grocery.txt"],
            "edits": [{"field": "description", "value": "Compare two step sizes"}]}
    previous = [{"question": "Add a task", "answer": "Added", "tool": "create_task", "affected_task_id": task["id"]}]
    result = handle_request("edit_task", "Edit that task and change its description to Compare two step sizes",
                            previous, {"1": {"name": "grocery.txt"}}, plan, [], lambda *_: edit, [task])
    assert result["task_change"]["task_id"] == task["id"]
    assert result["task_change"]["fields"]["description"] == "Compare two step sizes"
    assert result["file_ids"] == result["task_change"]["remove_ids"] == []
