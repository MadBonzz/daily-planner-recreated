"""Workspace conversations used for a pause forecast across independent planner tabs."""

from datetime import date
import json
from pathlib import Path
import time

import pytest
from streamlit.testing.v1 import AppTest

import app
import estimate_worker as worker
from model_client import ModelUnavailable
from planner_store import PlannerStore
import task_workspace as desk
import workspace_eligibility as eligibility
import workspace_ui


INITIAL = {"point_min": 60, "low_min": 45, "high_min": 75, "confidence": "low", "evidence": []}
PROGRESS = "I finished the introduction and both diagrams. Only the last appendix needs editing."
QUESTION = "Which diagrams would be clearest in the introduction?"
PRIVATE = "The budget amendment for the other project is classified as emerald."
PDF = Path(__file__).parents[1] / "data/tasks/pdfs/09-rover-night/frame-drawing.pdf"


def make_task(store, title="Edit research handout", **options):
    return store.create_task(title=title, day=date.today(), estimate_minutes=60,
                             description="Edit the introduction, both diagrams and the appendix.",
                             **{"prediction": INITIAL, **options})


def workspace_token(store, task, monkeypatch):
    request = eligibility.payload(task, desk.planner_settings())
    monkeypatch.setattr(eligibility, "package", lambda: {desk.decision_hash(request): True})
    return desk.availability(store, task)["token"]


def open_page(token=None):
    page = AppTest.from_string("from app import main\nmain()", default_timeout=10)
    if token:
        page.query_params["workspace"] = token
    page.run()
    assert not page.exception
    return page


def click(page, label):
    button = next(item for item in page.button if item.label == label)
    button.click().run()
    assert not page.exception


def supported_reply():
    return {"assessment": "supported", "remaining_min": 20, "remaining_low_min": 15, "remaining_high_min": 30,
            "explanation": "Only the appendix edit remains.", "confidence": "medium",
            "confidence_reason": "The writer identified the completed sections.",
            "evidence": [{"turn": 1, "quote": "I finished the introduction and both diagrams.", "kind": "completed"}]}


@pytest.fixture
def planner(tmp_path, monkeypatch):
    now = [time.time()]

    def open_store(path):
        return PlannerStore(path, clock=lambda: now[0])

    monkeypatch.setattr(app, "DATA_ROOT", tmp_path)
    monkeypatch.setattr(app, "PlannerStore", open_store)
    monkeypatch.setattr(workspace_ui, "PlannerStore", open_store)
    # Drive the durable worker synchronously after inspecting the paused timer.
    wakeups = []

    def schedule(store, index_dir):
        if store.has_pending_predictions():
            wakeups.append(store.path)

    monkeypatch.setattr(worker, "schedule_pending", schedule)
    monkeypatch.setattr(app, "schedule_pending", schedule)
    monkeypatch.setattr(workspace_ui, "schedule_pending", schedule)
    monkeypatch.setattr(worker.estimator, "predict", lambda *args: INITIAL.copy())
    monkeypatch.setattr(worker.estimator, "index_current_task", lambda *args: None)

    def answer(messages, *args, **kwargs):
        body = json.loads(messages[1]["content"])
        reply = {"answer": "Check the appendix references."}
        if "passages" in body:
            first = body["passages"][0]
            reply.update(supported=True, sources=[{"page": first["page"], "quote": first["text"][:35]}])
        return reply

    monkeypatch.setattr(desk, "chat_json", answer)
    return open_store(tmp_path / "live"), now, wakeups


@pytest.mark.parametrize("pause_from", ["planner", "reader"])
def test_workspace_chat_can_drive_pause_forecast_from_either_tab(planner, monkeypatch, pause_from):
    import workspace_forecast

    store, now, wakeups = planner
    files = [(PDF.name, PDF.read_bytes())] if pause_from == "reader" else []
    task = make_task(store, attachments=files)
    token = workspace_token(store, task, monkeypatch)
    reader = open_page(token)
    click(reader, "Start timer")
    now[0] += 15 * 60
    reader.chat_input[0].set_value(PROGRESS).run()
    assert not reader.exception
    assert not store.has_pending_predictions()

    # A new browser session cannot get the reader's chat from Streamlit session state.
    page = open_page() if pause_from == "planner" else open_page(token)
    click(page, "Pause" if pause_from == "planner" else "Pause timer")
    paused = store.get_task(task["id"])
    assert paused["status"] == "paused" and paused["actual_seconds"] == 15 * 60
    assert paused["prediction"] == INITIAL and paused["estimate_updating"]
    assert store.path in wakeups

    requests = []

    def assess(messages, schema, **kwargs):
        requests.append(json.loads(messages[1]["content"]))
        return supported_reply()

    monkeypatch.setattr(workspace_forecast, "chat_json", assess)
    assert worker.run_one_job(store, store.data_dir / "vectors")
    assert len(requests) == 1 and PROGRESS in json.dumps(requests[0])
    assert "Check the appendix references." in json.dumps(requests[0])
    result = store.get_task(task["id"])
    assert result["creation_prediction"] == INITIAL and result["status"] == "paused"
    assert not result["estimate_updating"]
    forecast = result["prediction"]
    assert (forecast["low_min"], forecast["point_min"], forecast["high_min"]) == (30, 35, 45)
    update = forecast["workspace_update"]
    assert (update["status"], update["elapsed_minutes"], update["remaining_min"]) == ("adjusted", 15, 20)
    if files:
        assert update["evidence"][0]["document"] == PDF.name
    page.run()
    assert not page.exception
    captions = "\n".join(item.value for item in page.caption)
    assert "20 min remaining" in captions and "35 min total" in captions


@pytest.mark.parametrize("workspace_start", [False, True])
def test_pause_without_new_workspace_evidence_keeps_the_estimate(planner, monkeypatch, workspace_start):
    store, now, _ = planner
    task = make_task(store)
    token = workspace_token(store, task, monkeypatch)
    if workspace_start:
        desk.start_workspace(store, token)
    else:
        # Chat from a previously opened reader, outside a workspace timer, is not pause evidence.
        desk.begin_question(store, token, "task", PROGRESS)
        store.transition(task["id"], "start")
    now[0] += 9 * 60
    paused = store.transition(task["id"], "pause")
    assert paused["status"] == "paused" and paused["actual_seconds"] == 9 * 60
    assert paused["prediction"] == INITIAL and not paused["estimate_updating"]
    assert store.claim_prediction() is None


def test_pause_while_initial_estimate_is_in_flight_preserves_creation_snapshot(planner, monkeypatch):
    store, now, _ = planner
    task = make_task(store, prediction=None)
    initial_job = store.claim_prediction(lease_seconds=3600)
    token = workspace_token(store, task, monkeypatch)
    desk.start_workspace(store, token)
    desk.begin_question(store, token, "task", PROGRESS)
    now[0] += 6 * 60
    store.transition(task["id"], "pause")

    # A crash at this point must still recover the original, pre-chat request.
    now[0] += 3600
    replacement = store.claim_prediction()
    assert replacement["token"] != initial_job["token"]
    assert "workspace_progress" not in replacement and PROGRESS not in json.dumps(replacement)
    assert store.resolve_prediction(task["id"], replacement["token"], INITIAL)
    progress_job = store.claim_prediction()
    assert progress_job["workspace_only"] and progress_job["previous_prediction"] == INITIAL
    assert progress_job["workspace_progress"]["elapsed_seconds"] == 6 * 60
    assert PROGRESS in json.dumps(progress_job["workspace_progress"])
    assert store.get_task(task["id"])["creation_prediction"] == INITIAL


def test_new_pause_supersedes_a_leased_forecast_and_scopes_its_snapshot(planner, monkeypatch):
    store, now, _ = planner
    other = make_task(store, "Other project handout", fixed_duration=True)
    other_token = workspace_token(store, other, monkeypatch)
    desk.start_workspace(store, other_token)
    desk.begin_question(store, other_token, "task", PRIVATE)
    store.transition(other["id"], "pause")

    task = make_task(store)
    token = workspace_token(store, task, monkeypatch)
    desk.start_workspace(store, token)
    desk.begin_question(store, token, "task", QUESTION)
    now[0] += 5 * 60
    store.transition(task["id"], "pause")
    old = store.claim_prediction(lease_seconds=3600)
    assert old["task"]["id"] == task["id"]
    assert PRIVATE not in json.dumps(old["workspace_progress"])

    desk.start_workspace(store, token)
    last_question = desk.begin_question(store, token, "task", PROGRESS)
    now[0] += 10 * 60
    store.transition(task["id"], "pause")
    # The model finished a chat request after Pause. That response did not exist at pause time.
    late_reply = "Private post-pause assistant text."
    desk.finish_question(store, token, "task", last_question,
                         {"answer": late_reply, "sources": [], "provider": "Test", "model": "Chat"})
    assert not store.resolve_prediction(task["id"], old["token"], {**INITIAL, "point_min": 99})
    fresh = store.claim_prediction()
    assert fresh["token"] != old["token"]
    snapshot = fresh["workspace_progress"]
    assert snapshot["elapsed_seconds"] == 15 * 60
    assert PROGRESS in [turn["content"] for turn in snapshot["messages"] if turn["role"] == "user"]
    assert late_reply not in json.dumps(snapshot) and PRIVATE not in json.dumps(snapshot)
    assert store.get_task(task["id"])["creation_prediction"] == INITIAL


def test_retry_reuses_the_failed_pause_without_restarting_the_timer(planner, monkeypatch):
    import workspace_forecast

    store, now, _ = planner
    task = make_task(store)
    token = workspace_token(store, task, monkeypatch)
    desk.start_workspace(store, token)
    desk.begin_question(store, token, "task", PROGRESS)
    now[0] += 15 * 60
    store.transition(task["id"], "pause")
    requests = []

    def offline_then_recover(messages, schema, **kwargs):
        requests.append(json.loads(messages[1]["content"]))
        if len(requests) == 1:
            raise ModelUnavailable("Offline")
        return supported_reply()

    monkeypatch.setattr(workspace_forecast, "chat_json", offline_then_recover)
    index_dir = store.data_dir / "vectors"
    assert worker.run_one_job(store, index_dir)
    failed = store.get_task(task["id"])
    assert failed["status"] == "paused" and failed["prediction"]["point_min"] == 60
    assert failed["prediction"]["workspace_update"]["status"] == "unavailable"
    now[0] += 20 * 60
    reader = open_page(token)
    click(reader, "Retry workspace estimate")
    assert worker.run_one_job(store, index_dir)
    retried = store.get_task(task["id"])
    assert len(requests) == 2
    assert retried["status"] == "paused" and retried["actual_seconds"] == 15 * 60
    assert retried["creation_prediction"] == INITIAL and retried["prediction"]["point_min"] == 35
    assert retried["prediction"]["workspace_update"]["elapsed_minutes"] == 15

    # Pausing later without another conversation cannot spend this evidence twice.
    desk.start_workspace(store, token)
    now[0] += 3 * 60
    store.transition(task["id"], "pause")
    assert store.claim_prediction() is None
