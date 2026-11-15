from datetime import date, datetime
from unittest.mock import Mock, patch
import json

from planner_analytics import _filter_history, ask_analytics
from planner_chat import answer_plan, plan_window, planning_context


def fixture():
    now = datetime(2026, 11, 14, 22, 30).astimezone()
    entries = []
    for i, (name, estimate, spent, status) in enumerate([
        ("GPU write-up", 50, 20, "paused"), ("Laundry", 20, 0, "ready"),
        ("Robot calibration", 15, 20, "running"), ("Review", 10, 8, "awaiting_finish"),
        ("Already done", 30, 30, "completed"), ("Archived", 60, 0, "ready")]):
        entries.append({"id": str(i), "title": name, "category": "Study", "day": "2026-11-14",
                        "actual_seconds": spent*60, "status": status, "estimate_minutes": estimate,
                        "archived": name == "Archived", "description": "Due before midnight" if i == 0 else "",
                        "prediction": {"point_min": estimate, "low_min": estimate-5, "high_min": estimate+20}
                                      if name != "Laundry" else {"status": "pending"}})
    return planning_context(entries, now.date(), now)


def test_planning_window_and_clock_boundaries():
    ctx = fixture()
    assert plan_window("What can I do?", ctx)["calendar_minutes"] == 90
    assert plan_window("I can work for 2 hours before midnight",ctx)["available_minutes"] == 90
    assert plan_window("I have 1 hour 10 minutes",ctx)["available_minutes"] == 70
    assert plan_window("Can I finish this in 35 minutes?",ctx)["budget_minutes"] == 35
    assert plan_window("I have an hour and a half",ctx)["budget_minutes"] == 90
    assert plan_window("I have 1h30m available",ctx)["budget_minutes"] == 90
    assert plan_window("30 minutes left, what should I tackle?",ctx)["budget_minutes"] == 30
    assert plan_window("What fits before 11 PM?", ctx)["available_minutes"] == 30
    assert plan_window("Which can I postpone to tomorrow?",ctx)["day"] == ctx["selected_day"]
    assert plan_window("Tomorrow's plan?",ctx)["future"]
    assert plan_window("Yesterday?",ctx)["available_minutes"] == 0
    assert "error" in plan_window("What can I do by 7?",ctx)
    assert "error" in plan_window("What fits before dinner?",ctx)
    assert plan_window("I have 25 minutes until dinner",ctx)["available_minutes"] == 25


def test_running_and_manual_progress_plan_cannot_overspend_or_treat_overrun_as_finished():
    snap = fixture()
    decide = Mock(return_value={"order": [{"id": i, "reason": "Continue the task."} for i in [1,2,3,4]]})
    r = answer_plan("What if I only have 40 minutes?",snap,[],[],decide)
    assert r["plan_window"]["available_minutes"] == 40 and r["answer_source"] == "model"
    by_name = {row["Task"]: row for row in r["plan_rows"]}
    assert set(by_name) == {"GPU write-up","Laundry","Robot calibration","Review"}
    assert by_name["GPU write-up"]["Remaining (min)"] == 30
    assert by_name["GPU write-up"]["Suggestion"] == "Fits by point estimate"
    assert by_name["Laundry"]["Suggestion"].startswith("Consider deferring")
    assert by_name["Laundry"]["Estimate source"] == "provisional user estimate"
    assert by_name["Robot calibration"]["Remaining (min)"] is None
    assert by_name["Robot calibration"]["Suggestion"] == by_name["Review"]["Suggestion"] == "Confirm remaining work"
    assert "can overrun" in r["answer"]
    next_time=fixture();next_time["now"]=next_time["now"].replace(minute=40)
    assert plan_window("Does the order change?", next_time,r["plan_window"])["available_minutes"] == 30


def test_follow_up_uses_same_router_and_prior_question_but_only_visible_logs():
    snap=fixture()
    before={"question":"What distracted me?", "answer":"Video highlights.", "tool":"distraction_sources"}
    route={"tool":"plan_remaining","scope":"all","task_ids":[],"sql":""}
    with patch("planner_analytics.chat_json",side_effect=[route, {"order": [{"id": i, "reason": "A due time is stated."} for i in [1,2,3,4]]}]) as model:
        r=ask_analytics("Given that, what should I work on by 11 pm?",[],[],[before],plan=snap)
    assert r["plan_window"]["available_minutes"] == 30
    payload=json.loads(model.call_args_list[-1].args[0][-1]["content"])
    assert payload["previous_turns"][0]["answer"] == "Video highlights."
    assert all(t["status"] != "completed" for t in payload["unfinished_tasks"])


def test_log_filters_remain_applicable_on_the_daily_page():
    tasks=[{"task_id":"a","title":"Read","category":"Study","completed_day":"2026-11-13"}]
    logs=[{"task_id":"a","device":d,"started":"2026-11-13T12:00","app":"Browser","phone_app":"","domain":"","window":"","classification":"off_task"} for d in ("Phone","Laptop")]
    t, visible=_filter_history(tasks,logs,{"categories":["Study"],"device":"Phone"})
    assert t==tasks and len(visible)==1 and visible[0]["device"]=="Phone"
