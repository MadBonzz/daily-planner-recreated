"""Task mutations through the Streamlit composer and Bionic's local completion protocol."""
import json
from datetime import date, timedelta
from io import BytesIO
from pathlib import Path
from unittest.mock import Mock

import pytest
from streamlit.testing.v1 import AppTest

import app
import dev_workspace
import local_model_runtime
import model_client
import task_workspace
from planner_analytics import analytics_rows
from planner_store import PlannerStore


QUESTION = ("can you add a task for my distributed computing assignment, I expect it will take 4 hours, "
            "it has theoretical questions to prove various properties in algorithms related to consensus, "
            "paxos, broadcast etc")
MODEL = "ternary-bonsai-2-4b-mlx"
INSTANCE = "local-bonsai-instance"
ORIGIN = "http://127.0.0.1:1235"
TITLE = "Distributed computing assignment"
DRAFT = {"content": "", "reasoning_content": '{"action":"create","title":"Unrequested task","estimate_minutes":999}'}


@pytest.fixture
def chat(tmp_path, monkeypatch):
    monkeypatch.setattr(app, "DATA_ROOT", tmp_path)
    monkeypatch.setattr(app, "schedule_pending", lambda *_: None)
    monkeypatch.setattr(app, "schedule_reports", lambda *_: None)
    monkeypatch.setattr(task_workspace, "availability", lambda *_: {"eligible": False})
    for key, value in {"PLANNER_SINGLE_MODEL": "1", "PLANNER_MODEL": MODEL, "PLANNER_MODEL_KEY": MODEL,
                       "PLANNER_MODEL_ENDPOINT": ORIGIN + "/v1/chat/completions"}.items():
        monkeypatch.setenv(key, value)

    requests, replies = [], []

    def response(request, *, timeout):
        if request.full_url == ORIGIN + "/api/v1/models":
            value = {"models": [{"key": MODEL, "type": "llm", "loaded_instances": [
                {"id": INSTANCE, "config": {"context_length": 8192, "parallel": 1}}]}]}
        else:
            assert request.full_url == ORIGIN + "/v1/chat/completions"
            body = json.loads(request.data)
            assert body["model"] == INSTANCE
            requests.append(body)
            message = replies.pop(0)
            if callable(message):
                message = message(json.loads(body["messages"][-1]["content"]))
            value = {"choices": [{"finish_reason": "stop", "message": message}]}
        return BytesIO(json.dumps(value).encode())

    transport = Mock()
    transport.open.side_effect = response
    monkeypatch.setattr(local_model_runtime, "build_opener", lambda *_: transport)
    monkeypatch.setattr(model_client, "build_opener", lambda *_: transport)
    page = AppTest.from_string("from app import main\nmain()", default_timeout=15).run()
    assert not page.exception
    return page, PlannerStore(tmp_path / "live"), requests, replies


@pytest.mark.parametrize("second_reply,attach_pdf", [({"content": '{"category":"Study"}'}, False), (DRAFT, False), (DRAFT, True)],
                         ids=["valid-repair", "literal-request", "literal-request-with-related-pdf"])
def test_add_four_hour_assignment_once_after_reasoning_only_reply(chat, second_reply, attach_pdf):
    page, store, requests, replies = chat
    replies.extend([DRAFT, second_reply])
    question = QUESTION
    if attach_pdf:
        question += ". Attach assignment.pdf to this task."
    message = page.chat_input[0].set_value(question)
    if attach_pdf:
        pdf = (Path(__file__).resolve().parents[1] / "data/tasks/pdfs/01-replica-proofs/bridge-proof.pdf").read_bytes()
        message.upload("assignment.pdf", pdf, "application/pdf")
    page.run()
    assert not page.exception
    page.run()
    assert not page.exception
    turn = page.session_state["planner_chat"]["turns"][-1]
    tasks = store.list_tasks()
    assert len(tasks) == 1, turn["answer"]
    task, = tasks
    assert turn["affected_task_id"] == task["id"]
    assert TITLE.casefold() in task["title"].casefold() and task["estimate_minutes"] == 240
    scope = (task["description"] + " " + task["details"]).casefold()
    assert all(term in scope for term in ("theoretical", "consensus", "paxos", "broadcast"))
    if attach_pdf:
        file, = task["attachments"]
        assert file["name"] == "assignment.pdf" and store.attachment_bytes(file["id"]) == pdf
    else:
        assert not task["attachments"]
    first, retry = requests
    assert all("response_format" not in body and "category" in body["messages"][0]["content"]
               for body in (first, retry))
    request = json.loads(retry["messages"][-1]["content"])
    assert set(request) == {"title", "description", "categories"}
    assert request["title"] == TITLE and "broadcast" in request["description"]
    if second_reply != DRAFT:
        assert task["category"] == "Study"


def test_explicit_postponement_updates_the_existing_task_without_inference(chat):
    page, store, requests, _ = chat
    task = store.create_task(title=TITLE, description="Prove consensus properties.",
                             day=date.today(), estimate_minutes=240)
    tomorrow = (date.today() + timedelta(days=1)).isoformat()
    page.chat_input[0].set_value("Postpone distributed computing assignment to tomorrow.").run()
    assert not page.exception
    page.run()
    assert not page.exception
    saved, = store.list_tasks()
    assert saved["id"] == task["id"] and saved["day"] == tomorrow and saved["estimate_minutes"] == 240
    assert not requests


def test_nora_distractions_answer_comes_from_real_logs_when_bonsai_only_reasons(chat, tmp_path):
    page, _, _, replies = chat
    next(control for control in page.get("segmented_control") if control.key == "planner_mode").set_value("Dev").run()
    page.selectbox(key="dev_history").set_value("d1-systems-student").run()
    next(control for control in page.get("segmented_control") if control.key == "planner_view").set_value("Analytics & logs").run()
    assert not page.exception
    # Every inference attempt has an empty final answer; the only evidence is the actual saved history.
    replies.extend([DRAFT] * 8)
    page.chat_input[0].set_value("what are the biggest distractions I need to avoid while working?").run()
    assert not page.exception
    turn = page.session_state["planner_chat"]["turns"][-1]
    assert turn["tool"] == "distraction_sources", turn["answer"]
    assert turn["query_source"] == turn["answer_source"] == "local records"
    assert not turn.get("affected_task_id")
    top = turn["rows"][0]
    store = dev_workspace.open_dev_workspace(tmp_path, "d1-systems-student")
    _, logs = analytics_rows(store.completed_tasks(), store.get_setting("exclusions", []))
    matched = [row for row in logs if row["classification"] == "off_task"
               and row["device"] == top["device"] and row["phone_app"] == top["phone_app"]
               and row["domain"].casefold().removeprefix("www.") == top["domain"]
               and (top["domain"] or (row["app"], row["window"]) == (top["app"], top["window"]))]
    assert top["minutes"] == round(sum(row["seconds"] for row in matched) / 60, 1) > 0
    source = top["domain"] or top["phone_app"] or top["app"]
    assert source and source in turn["answer"] and "reasoning without a final" not in turn["answer"]
