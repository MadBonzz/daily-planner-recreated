from datetime import date, datetime
from zoneinfo import ZoneInfo

import pytest

import planner_analytics as analytics
from planner_chat import plan_window


def test_recurring_domain_accumulates_across_tabs_browsers_and_days():
    source = {"device": "Laptop", "app": "Edge", "window": "Recap", "domain": "sport.test", "phone_app": "",
              "started": "2026-11-11T18:00:00", "seconds": 180, "task_id": "g1",
              "source": "reviewed", "reason": "", "classification": "off_task"}
    logs = [source, dict(source, app="Chrome", domain="WWW.Sport.test", window="Live results", task_id="g2",
                         started="2026-11-12T18:00:00", seconds=240),
            dict(source, device="Phone", domain="", phone_app="Scores", app="", window="", seconds=30),
            dict(source, classification="uncertain", seconds=600),
            dict(source, classification="on_task", seconds=700)]
    rows, _ = analytics.run_query(analytics._DISTRACTION_SQL, [], logs)
    assert [(r["device"], r["domain"], r["tasks"], r["days"], r["minutes"]) for r in rows] == [
        ("Laptop", "sport.test", 2, 2, 7), ("Phone", "", 1, 1, .5)]
    assert rows[0]["app"] == rows[0]["window"] == ""


@pytest.mark.parametrize("question, minutes", [
    ("What fits by midnight?", 360),
    ("What fits by 12:00 am?", 360),
    ("What fits before 7 PM?", 60),
    ("What fits before 19:30?", 90),
])
def test_window_uses_supplied_local_clock(question, minutes):
    now = datetime(2026, 11, 13, 18, 0, tzinfo=ZoneInfo("Asia/Kolkata"))
    plan = {"selected_day": now.date().isoformat(), "now": now, "tasks": []}
    assert plan_window(question, plan)["available_minutes"] == minutes


def test_calendar_window_respects_dst_on_future_day():
    now = datetime(2026, 10, 31, 18, 0, tzinfo=ZoneInfo("America/New_York"))
    plan = {"selected_day": date(2026, 11, 1).isoformat(), "now": now, "tasks": []}
    assert plan_window("Which tasks fit?", plan)["available_minutes"] == 25 * 60


def test_workspace_scopes_current_question_to_requested_pages(monkeypatch):
    import json
    import task_workspace

    text = "Page 1: There are seven signed messages.\nPage 2: The clerk uses red ink.\nPage 3: A bell rings."
    questions = []
    def reply(messages, schema, **kwargs):
        questions.append(json.loads(messages[-1]["content"])["passages"])
        return {"supported": False, "answer": "", "sources": []}
    monkeypatch.setattr(task_workspace, "chat_json", reply)
    task_workspace.answer({}, "According to page 2, how many messages are there?", document="Tickets", text=text,
                          dialogue=[{"role": "user", "content": "Explain page 1"}])
    assert [row["page"] for row in questions[0]] == [2]
    task_workspace.answer({}, "Compare pages 1 and 3.", document="Tickets", text=text)
    assert [row["page"] for row in questions[1]] == [1, 3]
    task_workspace.answer({}, "On pages 2 through 3, what happens?", document="Tickets", text=text)
    assert [row["page"] for row in questions[2]] == [2, 3]
    result = task_workspace.answer({}, "On page 7 what happens?", document="Tickets", text=text)
    assert result["sources"] == [] and len(questions) == 3


def test_remaining_plan_never_adds_a_minute_to_workspace_progress():
    from planner_chat import planning_context

    now = datetime(2026, 11, 13, 18)
    pause = {"status": "adjusted", "elapsed_minutes": 12.5, "remaining_min": 20}
    prediction = {"basis": "Workspace progress forecast", "workspace_update": pause,
                  "point_min": 33, "low_min": 28, "high_min": 38}
    task = {"id": "t", "title": "Draft", "status": "paused", "day": now.date().isoformat(),
            "actual_seconds": 750, "estimate_minutes": 45, "prediction": prediction}
    def remaining():
        row, = planning_context([task], now.date(), now)["tasks"]
        return row["remaining_minutes"], row["remaining_low"], row["remaining_high"]
    assert remaining() == (20, 15, 25)
    task["actual_seconds"] += 120
    prediction["workspace_update"] = {"status": "insufficient_evidence", "last_adjusted": pause}
    assert remaining() == (18, 13, 23)
    task["actual_seconds"] = 750
    prediction.update(point_min=13, low_min=13, high_min=13)
    assert remaining() == (None, 0, 0)  # The user can confirm completion.


@pytest.mark.parametrize("date_phrase", ["2026-11-12", "November 12", "12th Nov"])
def test_date_specific_accuracy_cannot_run_unfiltered_summary(monkeypatch, date_phrase):
    from unittest.mock import Mock

    replies = Mock(side_effect=[
        {"scope": "all", "tool": "custom_sql", "task_ids": [], "sql": "SELECT COUNT(*) AS n FROM tasks"},
        {"answer": "There were zero tasks."},
    ])
    monkeypatch.setattr(analytics, "chat_json", replies)
    analytics.ask_analytics(f"How accurate were my estimates on {date_phrase}?", [], [])
    schema = replies.call_args_list[0].args[1]
    assert schema["properties"]["tool"]["enum"] == ["custom_sql"]


def test_uninformative_quote_is_repaired_before_showing_document_answer(monkeypatch):
    import task_workspace

    replies = iter([
        {"supported": True, "answer": "The deadline is March 3.", "sources": [{"page": 1, "quote": "the"}]},
        {"supported": False, "answer": "", "sources": []},
    ])
    monkeypatch.setattr(task_workspace, "chat_json", lambda *a, **kw: next(replies))
    result = task_workspace.answer({}, "What is the deadline?", document="Protocol",
                                   text="Page 1: Submit the signed laboratory worksheet.")
    assert result["sources"] == [] and "March" not in result["answer"]


def test_schema_valid_integral_float_ids_work_for_priorities_and_distractions():
    from unittest.mock import patch
    from jsonschema import validate
    from planner_chat import answer_plan, planning_context

    now = datetime(2026, 11, 13, 14)
    day = now.date().isoformat()
    task = {"id": "a", "title": "Write", "status": "ready", "estimate_minutes": 15, "day": day}
    def rank(messages, schema, **kwargs):
        if "priorities" in schema["properties"]:
            reply = {"priorities": {"1": {"rank": 1.0, "reason": "It fits."}}}
        else:
            reply = {"order": [{"id": 1.0, "reason": "It fits."}, {"id": 999.0, "reason": "Invalid ID"}]}
        validate(reply, schema)
        return reply
    result = answer_plan("What fits in 20 minutes?", planning_context([task], now.date(), now), [], [], rank)
    assert [row["Task"] for row in result["plan_rows"]] == ["Write"]
    evidence = {"question": "Which sites?", "scope": {}, "result_rows": [
        {"device": "Laptop", "app": "Browser", "domain": "sport.test", "window": "", "phone_app": "",
         "minutes": 4, "tasks": 1, "days": 1}]}
    with patch.object(analytics, "chat_json", return_value={"finding_ids": [1.0, 1, 999.0]}):
        result = analytics._explain_distractions({}, evidence)
    assert result["answer"].count("sport.test") == 1


def test_every_open_task_needs_a_model_rank_and_actual_reason_is_visible():
    import json
    from jsonschema import ValidationError, validate
    from planner_chat import answer_plan, planning_context

    now = datetime(2026, 11, 13, 14)
    def task(tid, title, budget):
        return {"id": tid, "title": title, "status": "ready", "estimate_minutes": budget, "day": now.date().isoformat()}
    tasks = [task("slow", "Read report", 60), task("quick", "Send receipt", 15)]
    def rank(messages, schema, **kwargs):
        if "order" in schema["properties"]:
            return {"order": []}
        with pytest.raises(ValidationError):
            validate({"order": []}, schema)
        with pytest.raises(ValidationError):
            validate({"priorities": {"1": {"rank": 1, "reason": "One item."}}}, schema)
        assert len(json.loads(messages[-1]["content"])["unfinished_tasks"]) == 2
        reply = {"priorities": {"1": {"rank": 2, "reason": "The report needs 60 minutes."},
                                "2": {"rank": 1, "reason": "The receipt fits the 20-minute budget."}}}
        validate(reply, schema)
        return reply
    result = answer_plan("What fits in 20 minutes?", planning_context(tasks, now.date(), now), [], [], rank)
    assert result["answer_source"] == "model"
    assert [(r["Task"], r["Model rank"]) for r in result["plan_rows"]] == [("Send receipt", 1), ("Read report", 2)]
    assert all(r["Model reason"] in result["answer"] for r in result["plan_rows"])
