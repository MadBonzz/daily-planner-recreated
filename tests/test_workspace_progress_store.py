"""Workspace chat isolation and timer changes that bypass the reader's Pause button."""
from datetime import date, timedelta

import pytest

from planner_store import PlannerStore
import task_workspace as desk


def desk_task(store, name, documents=()):
    task = store.create_task(title=name, description="Revise the draft.", day=date.today(), estimate_minutes=30,
                             attachments=documents, prediction={"point_min": 30})
    desk._record(store, task["id"], "test-eligibility", True, "test")
    with store._db() as db:
        token = db.execute("SELECT token FROM task_workspaces WHERE task_id=?", (task["id"],)).fetchone()[0]
    return task, token


def test_concurrent_turns_are_scoped_to_the_original_doc_and_workspace_period(tmp_path):
    now = [100.]
    store = PlannerStore(tmp_path, clock=lambda: now[0])
    a, token = desk_task(store, "Edit A", [("first.pdf", b"%PDF-test"), ("second.pdf", b"%PDF-test")])
    b, _ = desk_task(store, "Edit B")
    first, second = [d["id"] for d in a["attachments"]]
    desk.start_workspace(store, token)
    q = desk.begin_question(store, token, first, "The first section is ready.", turn_id="browser-request")
    assert desk.begin_question(store, token, first, "The first section is ready.", turn_id=q) == q
    now[0] += 300
    store.transition(b["id"], "start")
    old = store.claim_prediction()
    assert old["workspace_progress"]["elapsed_seconds"] == 300
    desk.start_workspace(store, token)
    with pytest.raises(ValueError):
        desk.finish_question(store, token, second, q, {"answer": "cross-document response"})
    desk.finish_question(store, token, first, q, {"answer": "Delayed suggestion."})
    desk.finish_question(store, token, first, q, {"answer": "Repeated reply."})
    assert [row["content"] for row in desk.chat_history(store, token, first)] == [
        "The first section is ready.", "Delayed suggestion."]
    desk.begin_question(store, token, second, "The appendix still needs revision.")
    now[0] += 120
    store.transition(a["id"], "pause")
    assert not store.resolve_prediction(a["id"], old["token"], {"point_min": 10})
    latest = store.claim_prediction()
    progress = latest["workspace_progress"]
    assert [(m["document"], m["content"]) for m in progress["messages"]] == [
        ("first.pdf", "The first section is ready."), ("second.pdf", "The appendix still needs revision.")]
    assert progress["new_message_start"] == 0
    assert progress["elapsed_seconds"] == 420
    store.archive_task(a["id"])
    assert not store.resolve_prediction(a["id"], latest["token"], {"point_min": 10})
    with pytest.raises(ValueError):
        desk.chat_history(store, token, first)
    store.delete_task(a["id"])
    with store._db() as db:
        assert not db.execute("SELECT id FROM workspace_messages WHERE task_id=?", (a["id"],)).fetchall()
        assert not db.execute("SELECT 1 FROM workspace_progress WHERE task_id=?", (a["id"],)).fetchall()


def test_stop_continue_postpone_and_corrected_duration_preserve_progress(tmp_path):
    today = date.today()
    now = [100.]
    store = PlannerStore(tmp_path, clock=lambda: now[0])
    task, token = desk_task(store, "Check argument")
    tid = task["id"]
    desk.start_workspace(store, token)
    desk.begin_question(store, token, "task", "I have corrected the argument.")
    now[0] += 1200
    store.transition(tid, "stop")
    assert store.claim_prediction() is None
    store.transition(tid, "continue_later", spent_seconds=600)
    first = store.claim_prediction()
    assert first["workspace_progress"]["elapsed_seconds"] == 600
    assert store.resolve_prediction(tid, first["token"], {"point_min": 22, "workspace_update": {"status": "adjusted"}})
    desk.start_workspace(store, token)
    desk.begin_question(store, token, "task", "The last figure needs labels.")
    now[0] += 60
    store.postpone(tid, today + timedelta(days=1), previous_day=today)
    second = store.claim_prediction()
    assert second["workspace_progress"]["new_message_start"] == 1
    assert second["workspace_progress"]["elapsed_seconds"] == 660
    store.transition(tid, "adjust_time", spent_seconds=900)
    assert not store.renew_prediction(tid, second["token"])
    corrected = store.claim_prediction()
    assert corrected["workspace_progress"]["elapsed_seconds"] == 900
    assert corrected["previous_prediction"]["point_min"] == 22


def test_backdated_stop_retains_later_chat_for_reading_only(tmp_path):
    now = [100.]
    store = PlannerStore(tmp_path, clock=lambda: now[0])
    task, token = desk_task(store, "Review protocol")
    tid = task["id"]
    desk.start_workspace(store, token)
    now[0] += 600
    desk.begin_question(store, token, "task", "I wrote this after the work was over.")
    store.transition(tid, "stop", ended_at=now[0] - 300)
    store.transition(tid, "continue_later")
    assert store.claim_prediction() is None
    desk.start_workspace(store, token)
    desk.begin_question(store, token, "task", "I just reduced the failing case.")
    now[0] += 120
    store.transition(tid, "pause")
    assert len(desk.chat_history(store, token, "task")) == 2
    snapshot = store.claim_prediction()["workspace_progress"]
    assert [m["content"] for m in snapshot["messages"]] == ["I just reduced the failing case."]
    assert snapshot["elapsed_seconds"] == 420
