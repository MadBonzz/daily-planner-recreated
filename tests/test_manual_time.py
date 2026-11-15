"""Missed timers, corrections and direct completion through durable planner state."""

import pytest
from streamlit.testing.v1 import AppTest

from planner_store import PlannerStore
from report_worker import run_one_report
from planner_analytics import accuracy, analytics_rows


def make_task(store, title="Prepare handout", minutes=120):
    return store.create_task(title=title, description="Check and print a handout", category="Work",
                             day="2026-11-14", estimate_minutes=minutes,
                             prediction={"point_min": minutes, "low_min": minutes/2, "high_min": minutes*2,
                                         "confidence": "low", "evidence": []})["id"]


def test_partial_work_without_timer_then_forgotten_stop(tmp_path):
    now = [3600.]
    store = PlannerStore(tmp_path, clock=lambda: now[0])
    store.set_capture(False)
    task = make_task(store)
    edited = store.transition(task, "adjust_time", spent_seconds=1800)
    assert edited["status"] == "paused" and edited["actual_seconds"] == edited["manual_seconds"] == 1800
    assert not store.periods(task)
    store.transition(task, "resume")
    now[0] += 300
    other = make_task(store, "Check wheel bearings")
    store.transition(other, "start")
    assert store.get_task(task)["actual_seconds"] == 2100
    store.transition(task, "resume")
    assert store.get_task(other)["status"] == "paused"
    now[0] += 15000
    store.transition(task, "stop")
    saved = store.transition(task, "finish", spent_seconds=2400, queue_report=True, completion_note="Forgot to stop")
    assert saved["actual_seconds"] == 2400 and saved["manual_seconds"] == 1800
    assert store.periods(task) == [(3600, 3900), (3900, 4200)]
    assert run_one_report(store, dev=False)
    report = store.get_task(task)["report"]
    assert report["elapsed_seconds"] == sum(report["totals"].values()) == 2400
    assert sum(r["seconds"] for r in report["intervals"]) == 600
    assert "30 minutes without a timer" in report["summary"]
    assert report["totals"]["uncertain"] == 2400
    assert store.transition(task, "finish", spent_seconds=1200)["actual_seconds"] == 2400
    rows, _ = analytics_rows(store.completed_tasks(), [])
    assert rows[0]["actual_minutes"] == 40


def test_correct_timer_shortens_capture_replaces_inflight_report_and_updates_future_forecast(tmp_path):
    now = [1000.]
    store = PlannerStore(tmp_path, clock=lambda: now[0])
    task = make_task(store)
    store.transition(task, "start")
    now[0] += 3600
    store.transition(task, "stop")
    store.transition(task, "finish", queue_report=True)
    old = store.report_job(task)
    store.transition(task, "adjust_time", spent_seconds=900)
    assert store.report_job(task)["generation"] != old["generation"]
    assert not store.save_report(task, {"intervals": []}, generation=old["generation"])
    assert store.periods(task) == store.capture_periods(task) == [(1000,1900)]
    assert not store.get_task(task)["report"]
    again = PlannerStore(tmp_path, clock=lambda: now[0])
    assert again.get_task(task)["actual_seconds"] == 900
    target = again.create_task(title="Next handout", description="Print a handout", category="Work",
                               day="2026-11-14", estimate_minutes=15)
    lease = again.claim_prediction()
    assert lease["task"]["id"] == target["id"]
    assert lease["history"][0]["actual_seconds"] == 900


def test_complete_unstarted_with_unknown_time_then_fill_it_in(tmp_path):
    store = PlannerStore(tmp_path)
    task = make_task(store)
    store.transition(task, "finish", completion_note="Already done", queue_report=True)
    assert not store.periods(task)
    assert run_one_report(store, dev=False)
    completed = store.get_task(task)
    assert completed["completion_note"] == "Already done" and completed["report"]["time_unknown"]
    assert not accuracy(analytics_rows([completed], [])[0], "user_minutes")
    store.transition(task, "adjust_time", spent_seconds=1800)
    assert run_one_report(store, dev=False)
    completed = store.get_task(task)
    assert completed["report"]["elapsed_seconds"] == completed["actual_seconds"] == 1800
    assert completed["report"]["intervals"] == []
    assert completed["report"]["totals"]["uncertain"] == 1800
    assert accuracy(analytics_rows([completed], [])[0], "user_minutes")[0]["Tasks"] == 1


def test_manual_pause_continue_and_direct_completion_while_another_timer_runs(tmp_path):
    store = PlannerStore(tmp_path)
    task, other = make_task(store), make_task(store, "Proofread second handout")
    store.transition(other, "start")
    store.transition(task, "adjust_time", spent_seconds=600)
    store.transition(task, "stop")
    paused = store.transition(task, "continue_later", spent_seconds=300)
    assert paused["status"] == "paused" and paused["actual_seconds"] == 300
    store.transition(task, "finish", spent_seconds=1200, completion_note="Finished away from laptop")
    assert store.running_task()["id"] == other
    assert store.get_task(task)["manual_seconds"] == 1200
    with pytest.raises(ValueError):
        store.transition(other, "adjust_time", spent_seconds=42)
    with pytest.raises(ValueError):
        store.transition(task, "adjust_time", spent_seconds=float("nan"))
    assert store.get_task(task)["actual_seconds"] == 1200


def test_running_clock_offers_stop_at_twice_users_original_budget(tmp_path):
    now = [1000]
    store = PlannerStore(tmp_path, clock=lambda: now[0])
    task = make_task(store, minutes=15)
    store.transition(task, "start")
    def render(at):
        code = f"from planner_store import PlannerStore\nfrom app import running_clock\nrunning_clock(PlannerStore({str(tmp_path)!r}, clock=lambda:{at}), {task!r})"
        ui = AppTest.from_string(code).run()
        assert not ui.exception
        return ui
    assert not render(2799).warning
    overdue = render(2800)
    assert "twice" in overdue.warning[0].value
    assert overdue.button[0].label == "Stop and review time"


def test_task_card_can_save_offline_work_then_finish_with_a_note(tmp_path):
    store = PlannerStore(tmp_path)
    task_id = make_task(store)
    # The UI reads the real SQLite workspace on each Streamlit rerun.
    code = f"""
from app import render_task
from planner_store import PlannerStore
store = PlannerStore({str(tmp_path)!r})
render_task(store, store.data_dir/'vectors', store.get_task({task_id!r}), dev=True)
"""
    ui = AppTest.from_string(code).run()
    ui.button(key=f"edit-time-{task_id}").click().run()
    assert not ui.exception
    ui.number_input[0].set_value(30)
    next(b for b in ui.button if b.label == "Save time").click().run()
    assert store.get_task(task_id)["actual_seconds"] == 1800
    ui.button(key=f"finish-{task_id}").click().run()
    ui.number_input[0].set_value(35)
    ui.text_area(key=f"finish-note-{task_id}").set_value("Finished the last page on paper.")
    next(b for b in ui.button if b.label == "Yes, finish task").click().run()
    assert not ui.exception
    saved = store.get_task(task_id)
    assert saved["status"] == "completed" and saved["actual_seconds"] == 2100
    assert saved["completion_note"] == "Finished the last page on paper."


def test_first_known_time_creates_missing_relation_and_only_refreshes_on_match(tmp_path):
    store = PlannerStore(tmp_path)
    old = make_task(store)
    store.transition(old, "finish")
    current = make_task(store, "Print another handout")
    assert store.claim_relation() is None
    store.transition(old, "adjust_time", spent_seconds=720)
    pair = store.claim_relation()
    assert set(pair["pair"]) == {old, current}
    assert store.claim_prediction() is None
    assert store.resolve_relation(pair["pair"], pair["token"], True)
    lease = store.claim_prediction()
    assert lease["task"]["id"] == current and lease["history"][0]["actual_seconds"] == 720


def test_finishing_workspace_check_preserves_inflight_edit_time_click(tmp_path):
    store = PlannerStore(tmp_path)
    task_id = make_task(store)
    code = f"""
from unittest.mock import patch
import streamlit as st
from app import render_task
from planner_store import PlannerStore

def availability(store, task):
    count = st.session_state.get("availability_reads", 0) + 1
    st.session_state["availability_reads"] = count
    # On the edit click the decision arrives between the card's check and its poll.
    return None if count == 2 else {{"eligible": False}}

store = PlannerStore({str(tmp_path)!r})
with patch("workspace_ui.desk.availability", availability):
    render_task(store, store.data_dir/'vectors', store.get_task({task_id!r}), dev=True)
"""
    ui = AppTest.from_string(code).run()
    ui.button(key=f"edit-time-{task_id}").click().run()
    assert not ui.exception
    assert [field.label for field in ui.number_input] == ["Total time spent (minutes)"]
