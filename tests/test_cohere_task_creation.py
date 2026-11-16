"""Exercise assignment creation through the actual Cohere request encoder and chat UI."""
from io import BytesIO
import json
from pathlib import Path
from unittest.mock import Mock
from urllib.error import HTTPError

import pytest
from streamlit.testing.v1 import AppTest

import app
import model_client
import task_workspace
from planner_store import PlannerStore


ASSIGNMENT_REQUEST = (
    "This is my assignment pdf. can you create a new distributed computing assignment, "
    "I would estimate it at aroun 200 minutes, the task is around solving questions "
    "related to Paxos, Consensus protocols etc"
)
COHERE_URL = "https://api.cohere.ai/compatibility/v1/chat/completions"


@pytest.mark.parametrize("with_pdf, grounded", [(True, True), (False, True), (True, False)],
                         ids=["uploaded-pdf", "no-upload", "conversational-request"])
def test_cohere_accepts_assignment_creation_with_and_without_pdf(tmp_path, monkeypatch, with_pdf, grounded):
    question = (ASSIGNMENT_REQUEST if grounded else
                "Add a task for my distributed computing assignment. Use the attached PDF for it. "
                "I expect 200 minutes of work on Paxos and consensus.")
    monkeypatch.setenv("PLANNER_MODEL", "command-r7b-12-2024")
    monkeypatch.setenv("PLANNER_MODEL_ENDPOINT", COHERE_URL)
    monkeypatch.setenv("COHERE_API", "test-token")
    monkeypatch.setattr(app, "DATA_ROOT", tmp_path)
    monkeypatch.setattr(app, "schedule_pending", lambda *_: None)
    monkeypatch.setattr(app, "schedule_reports", lambda *_: None)
    monkeypatch.setattr(task_workspace, "availability", lambda *_: {"eligible": False})

    def cohere_response(request, *, timeout):
        assert request.full_url == COHERE_URL
        body = json.loads(request.data)
        assert body["model"] == "command-r7b-12-2024"
        props = body["response_format"]["json_schema"]["schema"]["properties"]
        data = json.loads(body["messages"][-1]["content"])
        if grounded:
            assert set(props) == {"category"}
            assert data["title"] == "Distributed computing assignment" and "Paxos" in data["description"]
            assert "files" not in data
            reply = {"category": "Study"}
        elif "intent" in props:
            reply = {"intent": "create_task"}
        else:
            assert data["message"] == question
            # Cohere rejects numerical range constraints in structured output schemas.
            if "minimum" in props["estimate_minutes"]:
                error = {"message": "Numeric minimum is unsupported in response_format"}
                raise HTTPError(COHERE_URL, 400, "Bad Request", {}, BytesIO(json.dumps(error).encode()))
            assert [item["name"] for item in data["files"]] == (["HW_3_1.pdf"] if with_pdf else [])
            reply = {
                "action": "create", "title": "Distributed computing assignment",
                "description": "Solve questions about Paxos and consensus protocols.",
                "details": "", "estimate_minutes": 200, "estimate_phrase": "200 minutes",
                "fixed_duration": False, "day": "", "when": "", "category": "Study",
                "file_ids": [item["id"] for item in data["files"]], "reply": "",
            }
        completion = {"choices": [{"finish_reason": "stop", "message": {"content": json.dumps(reply)}}]}
        return BytesIO(json.dumps(completion).encode())

    transport = Mock()
    transport.open.side_effect = cohere_response
    monkeypatch.setattr(model_client, "build_opener", lambda *_: transport)

    page = AppTest.from_string("from app import main\nmain()", default_timeout=15).run()
    message = page.chat_input[0].set_value(question)
    if with_pdf:
        pdf = (Path(__file__).resolve().parents[1] / "data/tasks/pdfs/01-replica-proofs/bridge-proof.pdf").read_bytes()
        message.upload("HW_3_1.pdf", pdf, "application/pdf")
    page.run()
    assert not page.exception
    page.run()  # A normal Streamlit rerun must not repeat the write.
    assert not page.exception
    assert transport.open.call_count == 1  # Explicit assignment and budget already resolve the intent.
    store = PlannerStore(tmp_path / "live")
    tasks = store.list_tasks()
    assert len(tasks) == 1, page.session_state["planner_chat"]["turns"][-1]["answer"]
    task = tasks[0]
    assert task["title"] == "Distributed computing assignment"
    assert task["estimate_minutes"] == 200
    if with_pdf:
        attachment, = task["attachments"]
        assert attachment["name"] == "HW_3_1.pdf"
        assert store.attachment_bytes(attachment["id"]) == pdf
    else:
        assert task["attachments"] == []


def test_explicit_assignment_survives_rate_limit_and_keeps_its_uploaded_pdf(tmp_path, monkeypatch):
    from datetime import date, datetime
    from chat_tasks import apply_action
    from planner_analytics import ask_analytics

    monkeypatch.setenv("PLANNER_MODEL_ENDPOINT", COHERE_URL)
    monkeypatch.setenv("COHERE_API", "test-token")
    transport = Mock()
    transport.open.side_effect = HTTPError(COHERE_URL, 429, "Too Many Requests", {}, BytesIO(b"{}"))
    monkeypatch.setattr(model_client, "build_opener", lambda *_: transport)
    pdf = (Path(__file__).resolve().parents[1] / "data/tasks/pdfs/01-replica-proofs/bridge-proof.pdf").read_bytes()
    files = {"1": {"name": "HW_3_1.pdf", "current": True, "kind": "pdf", "data": pdf}}
    plan = {"now": datetime.now(), "selected_day": date.today().isoformat()}
    result = ask_analytics(ASSIGNMENT_REQUEST, [], [], plan=plan, files=files, create_categories=["Study"])
    store = PlannerStore(tmp_path)
    applied = apply_action(store, result, files, "submission")
    assert applied.get("affected_task_id"), applied.get("answer")
    task, = store.list_tasks()
    assert task["estimate_minutes"] == 200 and "paxos" in task["description"].lower()
    assert store.attachment_bytes(task["attachments"][0]["id"]) == pdf


def test_assignment_prefix_cannot_override_ambiguous_files_or_hypothetical_request():
    from chat_tasks import literal_task_creation
    files = {key: {"name": key, "kind": "pdf", "current": True} for key in ("exam.pdf", "notes.pdf")}
    plan = {"selected_day": "2026-11-16"}
    assert literal_task_creation(ASSIGNMENT_REQUEST, files, plan) is None
    hypothetical = ASSIGNMENT_REQUEST.replace("can you create", "what if I create")
    assert literal_task_creation(hypothetical, {"exam": files["exam.pdf"]}, plan) is None
