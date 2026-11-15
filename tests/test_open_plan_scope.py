from datetime import date, datetime
from unittest.mock import patch

import pytest

from planner_analytics import ask_analytics
from planner_chat import asks_what_fits, planning_context


@pytest.mark.parametrize("question", [
    "I have 25 minutes. Which task fits?",
    "What can I finish in 20 minutes?",
    "Which open tasks can I complete before 7 PM?",
])
def test_explicit_forward_looking_fit_question_uses_unfinished_plan(question):
    assert asks_what_fits(question, date(2026, 11, 13))


@pytest.mark.parametrize("question", [
    "Which tasks fit in 25 minutes yesterday?",
    "Which tasks fit in 25 minutes last Tuesday?",
    "Which tasks fit in 25 minutes on November 11?",
    "Which tasks fit in 25 minutes in the archives?",
    "Which tasks fit in 25 minutes on 2026-11-11?",
    "I have 30 minutes to review my logs. What distracted me?",
    "Which past tasks could I finish in 25 minutes?",
    "How many GPU tasks took under 25 minutes?",
    "How much did I underestimate the work I did in 20 minutes?",
    "Do I have anything left?",
])
def test_history_and_ambiguous_questions_still_use_the_model_router(question):
    assert not asks_what_fits(question, date(2026, 11, 13))


def test_wrong_historical_route_never_uses_completed_work_as_today_recommendation():
    now = datetime(2026, 11, 13, 18)
    t = {"id": "a", "title": "File receipts", "estimate_minutes": 15, "day": now.date().isoformat(),
         "status": "paused"}
    plan = planning_context([t], now.date(), now)
    bad_route = {"scope": "all", "tool": "custom_sql", "task_ids": [],
                 "sql": "SELECT title FROM tasks WHERE user_minutes <= 25"}
    ranks = {"order": [{"id": 1, "reason": "The receipts fit."}]}
    with patch("planner_analytics.chat_json", side_effect=[bad_route, ranks]) as chat:
        with patch("planner_analytics.run_query", side_effect=AssertionError("Read the wrong snapshot")):
            reply = ask_analytics("I have 25 minutes. Which task fits?", [], [], plan=plan)
    assert chat.call_count == 2
    assert reply["query_source"] == "open-plan scope check" and reply["model_proposed_tool"] == "custom_sql"
    assert reply["answer_source"] == "model" and reply["tool"] == "plan_remaining"
    assert reply["plan_rows"][0]["Task"] == "File receipts"
    assert reply["plan_rows"][0]["Model reason"] == "The receipts fit."
