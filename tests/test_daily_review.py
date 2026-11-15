from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta
import hashlib
import os
import sys

import pytest
from streamlit.testing.v1 import AppTest

import daily_review
from planner_store import PlannerStore


def add(store, day):
    return store.create_task(day=day, title="Sort the lab prints", description="Print and collate the plots.",
                             category="Work", estimate_minutes=15, fixed_duration=True)["id"]


def test_daily_review_at_local_time_once_across_tabs_catch_up_and_acknowledge(tmp_path):
    clock = [datetime(2026, 10, 21, 21, 29, 59)]
    def now():
        return clock[0].timestamp()
    store = PlannerStore(tmp_path / "live", clock=now)
    task = add(store, "2026-10-21")
    assert daily_review.preferences(store)["time"] == "21:30"
    assert daily_review.due(store) is None
    clock[0] += timedelta(seconds=1)
    assert daily_review.due(store) == {"day": "2026-10-21", "total": 1, "done": 0}
    messages = []
    tabs = [PlannerStore(store.data_dir, clock=now) for _ in range(4)]
    with ThreadPoolExecutor(4) as threads:
        results = list(threads.map(lambda tab: daily_review.deliver(tab, messages.append, desktop=False), tabs))
    assert results.count(True) == len(messages) == 1
    store.transition(task, "finish", spent_seconds=900)
    clock[0] = datetime(2026, 10, 22, 8, 30)
    assert daily_review.due(store)["done"] == 1
    assert not daily_review.deliver(store, messages.append, desktop=False)
    daily_review.acknowledge(store, "2026-10-21")
    assert daily_review.due(store) is None
    add(store, "2026-10-22")
    daily_review.configure(store, "08:45", True, False)
    clock[0] = datetime(2026, 10, 22, 9)
    assert daily_review.deliver(store, messages.append, desktop=False)
    assert [message["day"] for message in messages] == ["2026-10-21", "2026-10-22"]
    daily_review.configure(store, "08:45", False, False)
    assert daily_review.due(store) is None


def test_windows_registration_failure_and_missed_or_failed_delivery_can_recover(tmp_path, monkeypatch):
    clock = datetime(2026, 10, 22, 7).timestamp()
    store = PlannerStore(tmp_path / "live", clock=lambda: clock)
    add(store, "2026-10-21")
    def unavailable(*args):
        raise RuntimeError("Registration denied.")
    monkeypatch.setattr(daily_review, "_schedule_windows", unavailable)
    assert daily_review.configure(store, "20:00", True, True) == "Registration denied."
    assert daily_review.preferences(store) == {"time": "20:00", "enabled": True, "desktop": False}
    with pytest.raises(RuntimeError):
        daily_review.deliver(store, unavailable, desktop=False)
    assert daily_review.deliver(store, lambda plan: None, desktop=False)
    assert not daily_review.deliver(store, lambda plan: None, desktop=True)


def test_failed_desktop_delivery_falls_back_to_in_app_once(tmp_path, monkeypatch):
    now = datetime(2026, 10, 21, 22).timestamp()
    store = PlannerStore(tmp_path, clock=lambda: now)
    add(store, "2026-10-21")
    monkeypatch.setattr(daily_review, "_schedule_windows", lambda *args: None)
    assert daily_review.configure(store, "21:30", True, True) is None
    assert not daily_review.deliver(store, lambda plan: None, desktop=False)
    def denied(plan):
        raise RuntimeError("Desktop notifications blocked")
    with pytest.raises(RuntimeError, match="blocked"):
        daily_review.deliver(store, denied, desktop=True)
    assert store.get_setting("daily_review_desktop_error")["day"] == "2026-10-21"
    assert daily_review.deliver(store, lambda plan: None, desktop=False)
    assert not daily_review.deliver(store, lambda plan: None, desktop=True)


@pytest.mark.parametrize("state", ["ready", "running", "paused", "awaiting_finish"])
def test_postponing_preserves_task_evidence_and_stops_running_clock(tmp_path, state):
    now = [datetime(2026, 11, 14, 12).timestamp()]
    store = PlannerStore(tmp_path, clock=lambda: now[0])
    task = add(store, "2026-11-14")
    original = store.get_task(task)["creation_prediction"]
    if state != "ready":
        store.transition(task, "start")
        now[0] += 1800
        if state in ("paused", "awaiting_finish"):
            store.transition(task, "pause" if state == "paused" else "stop")
    target = store.postpone(task, "2026-11-15", previous_day="2026-11-14")
    now[0] += 3600
    assert target["status"] == ("ready" if state == "ready" else "paused")
    assert store.running_task() is None
    assert store.get_task(task)["actual_seconds"] == (0 if state == "ready" else 1800)
    assert store.list_tasks("2026-11-15")[0]["id"] == task
    assert not store.list_tasks("2026-11-14")
    assert bool(store.worked_on(date(2026, 11, 14))) == (state != "ready")
    assert target["creation_prediction"] == original
    with pytest.raises(ValueError, match="already moved"):
        store.postpone(task, "2026-11-18", previous_day="2026-11-14")
    with pytest.raises(ValueError, match="later"):
        store.postpone(task, "2026-11-15", previous_day="2026-11-15")
    if state != "ready":
        assert store.transition(task, "resume")["status"] == "running"
        with store._db() as db:
            assert db.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 1


def test_postpone_during_the_first_estimate_keeps_the_claim_and_recorded_period(tmp_path):
    now = [datetime(2026, 11, 14, 16).timestamp()]
    store = PlannerStore(tmp_path, clock=lambda: now[0])
    task = store.create_task(day="2026-11-14", title="Read the proof", description="Annotate the induction step.",
                             category="Reading", estimate_minutes=25)
    job = store.claim_prediction(lease_seconds=1800)
    assert job["task"]["id"] == task["id"]
    store.transition(task["id"], "start")
    now[0] += 600
    moved = store.postpone(task["id"], "2026-11-16", previous_day=task["day"])
    assert moved["estimate_updating"] and moved["status"] == "paused"
    prediction = {"point_min": 28, "explanation": "Estimated while the card was moving."}
    assert store.resolve_prediction(task["id"], job["token"], prediction)
    result = store.list_tasks("2026-11-16")[0]
    assert result["creation_prediction"] == result["prediction"] == prediction
    assert result["actual_seconds"] == 600 and not store.has_pending_predictions()


def test_postpone_and_daily_review_controls(tmp_path):
    clock = datetime.combine(date.today(), datetime.min.time()).replace(hour=22).timestamp()
    store = PlannerStore(tmp_path, clock=lambda: clock)
    key = add(store, date.today())
    script = f"""
from planner_store import PlannerStore
from review_ui import postpone_action, review_notice, review_settings
store = PlannerStore({str(tmp_path)!r}, clock=lambda: {clock})
review_settings(store)
review_notice(store)
postpone_action(store, store.get_task({key!r}))
"""
    ui = AppTest.from_string(script).run()
    assert not ui.exception
    assert ui.time_input[0].value.isoformat() == "21:30:00"
    ui.button(key="review-open").click().run()
    assert not ui.exception and ui.session_state["review_target_day"] == date.today().isoformat()
    ui.button(key=f"postpone-next-{key}").click().run()
    assert not ui.exception
    assert store.get_task(key)["day"] == (date.today() + timedelta(days=1)).isoformat()


def test_review_button_opens_the_due_plan_from_analytics(tmp_path, monkeypatch):
    import app

    store = PlannerStore(tmp_path / "live")
    add(store, date.today())
    daily_review.configure(store, "00:00", True, False)
    monkeypatch.setattr(app, "DATA_ROOT", tmp_path)
    ui = AppTest.from_string("from app import main\nmain()")
    ui.session_state["plan_day"] = date.today() - timedelta(days=3)
    ui.session_state["planner_view"] = "Analytics & logs"
    ui.run()
    assert not ui.exception
    ui.button(key="review-open").click().run()
    assert not ui.exception
    assert ui.session_state["planner_view"] == "Daily plan"
    assert ui.date_input(key="plan_day").value == date.today()
    assert daily_review.due(store) is None


@pytest.mark.skipif(sys.platform != "win32" or os.getenv("PLANNER_TEST_WINDOWS_SCHEDULER") != "1",
                    reason="Opt in on a Windows machine with a per-user Task Scheduler.")
def test_windows_register_change_and_remove_daily_trigger(tmp_path):
    store = PlannerStore(tmp_path / "reminder test folder")
    name = "Plan Smarter daily review " + hashlib.sha256(
        os.path.normcase(str(store.data_dir.resolve())).encode()).hexdigest()[:16]
    verify = r"""
$task = Get-ScheduledTask -TaskName $env:PLAN_REVIEW_NAME -ErrorAction Stop
$trigger = $task.Triggers | Where-Object { $_.CimClass.CimClassName -eq 'MSFT_TaskDailyTrigger' }
if (-not $trigger -or ([datetime]$trigger.StartBoundary).ToString('HH:mm') -ne $env:EXPECTED_TIME) { throw 'Wrong daily trigger' }
if (-not $task.Settings.StartWhenAvailable) { throw 'Missed runs cannot catch up' }
if (-not $task.Actions.Arguments.Contains('--notify')) { throw 'Incorrect action' }
if (-not (Test-Path -LiteralPath $task.Actions.Execute)) { throw 'Missing Python' }
if ($task.Principal.RunLevel -ne 'Limited') { throw 'Incorrect run level' }
"""
    try:
        for clock_time in ("21:30", "12:42"):
            daily_review._schedule_windows(store, clock_time, True)
            daily_review._powershell(verify, PLAN_REVIEW_NAME=name, EXPECTED_TIME=clock_time)
    finally:
        daily_review._schedule_windows(store, "12:42", False)
    daily_review._powershell(
        "if (Get-ScheduledTask -TaskName $env:PLAN_REVIEW_NAME -ErrorAction SilentlyContinue) { throw 'Task still installed' }",
        PLAN_REVIEW_NAME=name)
