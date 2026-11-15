"""Predeclared Ithaca clock scenarios. SQLite stores UTC instants; the person sees local time."""

from datetime import date, datetime
import os
import time
from zoneinfo import ZoneInfo

import pytest

import daily_review
from planner_analytics import analytics_rows, run_query
from planner_chat import plan_window, planning_context
from planner_store import PlannerStore


ITHACA = ZoneInfo("America/New_York")


def instant(value):
    return datetime.fromisoformat(value).timestamp()


@pytest.fixture
def ithaca():
    if not hasattr(time, "tzset"):
        pytest.skip("Changing the host's local zone in one process requires tzset")
    prior = os.environ.get("TZ")
    os.environ["TZ"] = "America/New_York"
    time.tzset()
    try:
        yield
    finally:
        if prior is None:
            del os.environ["TZ"]
        else:
            os.environ["TZ"] = prior
        time.tzset()


def task(store, day):
    return store.create_task(title="Check the board", description="Check the action list", day=day,
                             estimate_minutes=30, prediction={"point_min": 30})["id"]


def snapshot(stamp, *, day=None, aware=True):
    now = datetime.fromtimestamp(instant(stamp), ITHACA if aware else None)
    return planning_context([], day or now.date(), now)


@pytest.mark.parametrize(("stamp", "day", "minutes"), [
    ("2026-03-07T18:00:00-05:00", date(2026, 3, 8), 23 * 60),
    ("2026-10-31T18:00:00-04:00", date(2026, 11, 1), 25 * 60),
    ("2026-03-08T01:30:00-05:00", date(2026, 3, 8), 21 * 60 + 30),
])
def test_ithaca_calendar_measures_elapsed_minutes(stamp, day, minutes):
    assert plan_window("What fits?", snapshot(stamp, day=day))["available_minutes"] == minutes


@pytest.mark.parametrize("aware", [True, False], ids=["iana-zone", "dashboard-local"])
def test_next_occurrence_of_ambiguous_cutoff(request, aware):
    if not aware:
        request.getfixturevalue("ithaca")
    before = snapshot("2026-11-01T01:40:00-04:00", aware=aware)
    second = snapshot("2026-11-01T01:20:00-05:00", aware=aware)
    assert plan_window("What fits by 1:30 am?", before)["available_minutes"] == 50
    assert plan_window("What fits by 1:30 am?", second)["available_minutes"] == 10


@pytest.mark.parametrize("aware", [True, False], ids=["iana-zone", "dashboard-local"])
def test_ask_for_another_cutoff_during_spring_gap(request, aware):
    if not aware:
        request.getfixturevalue("ithaca")
    spring = snapshot("2026-03-08T01:40:00-05:00", aware=aware)
    assert "error" in plan_window("What fits by 2:30 am?", spring)


def test_followup_retains_cutoff_when_only_budget_changes():
    t = snapshot("2026-11-01T17:30:00-05:00")
    first = plan_window("What fits by 6 pm?", t)
    revised = plan_window("I have 90 minutes", t, first)
    assert revised["budget_minutes"] == 90 and revised["available_minutes"] == 30


def test_followup_retains_elapsed_budget_when_only_cutoff_changes():
    first = plan_window("I have 30 minutes", snapshot("2026-11-01T17:30:00-05:00"))
    t = snapshot("2026-11-01T17:40:00-05:00")
    revised = plan_window("What fits by 8 pm?", t, first)
    assert revised["available_minutes"] == 20


def test_repeated_subminute_followups_never_reset_budget():
    # At 17:32:32, 91 seconds of the original 600 are used, leaving 8 whole minutes.
    result = plan_window("I have 10 minutes", snapshot("2026-11-01T17:31:01-05:00"))
    for stamp in ["2026-11-01T17:31:32-05:00", "2026-11-01T17:32:01-05:00", "2026-11-01T17:32:32-05:00"]:
        result = plan_window("Which task first?", snapshot(stamp), result)
    assert result["available_minutes"] == 8


def test_completed_midnight_session_and_two_0130_instants_survive_restart(ithaca, tmp_path):
    now = [instant("2026-10-31T23:55:00-04:00")]
    store = PlannerStore(tmp_path, clock=lambda: now[0])
    tid = task(store, "2026-10-31")
    store.transition(tid, "start")
    seconds = 15 * 60
    intervals = [{"start": now[0], "seconds": seconds, "label": "on_task"}]
    for offset in ("-04:00", "-05:00"):
        intervals.append({"start": instant("2026-11-01T01:30:00" + offset), "seconds": 60, "label": "on_task"})
    now[0] = instant("2026-11-01T01:35:00-05:00")
    store.transition(tid, "stop")
    store.transition(tid, "finish", report={"intervals": intervals})
    reopened = PlannerStore(tmp_path, clock=lambda: now[0])
    assert [t["id"] for t in reopened.worked_on("2026-11-01")] == [tid]
    tasks, logs = analytics_rows(reopened.completed_tasks(), [])
    assert tasks[0]["completed_day"] == "2026-11-01" and tasks[0]["actual_minutes"] == 160
    by_day, _ = run_query("SELECT substr(started,1,10) AS day, sum(seconds) AS seconds FROM logs GROUP BY day ORDER BY day",
                          tasks, logs)
    assert by_day == [{"day": "2026-10-31", "seconds": 300}, {"day": "2026-11-01", "seconds": 720}]
    stamps = [datetime.fromisoformat(row["started"]).timestamp() for row in logs[-2:]]
    assert stamps == [event["start"] for event in intervals[-2:]]


def test_review_stays_due_during_repeated_hour_and_delivers_once(ithaca, tmp_path):
    now = [instant("2026-11-01T01:35:00-04:00")]
    store = PlannerStore(tmp_path, clock=lambda: now[0])
    task(store, "2026-10-31")
    task(store, "2026-11-01")
    daily_review.configure(store, "01:30", True, False)
    assert daily_review.due(store)["day"] == "2026-11-01"
    # The server slept through the first occurrence. The new clock reads an earlier time.
    now[0] = instant("2026-11-01T01:05:00-05:00")
    reopened = PlannerStore(tmp_path, clock=lambda: now[0])
    sent = []
    assert daily_review.deliver(reopened, sent.append, desktop=False)
    assert sent[0]["day"] == "2026-11-01"
    now[0] = instant("2026-11-01T01:35:00-05:00")
    assert not daily_review.deliver(store, sent.append, desktop=False)
    daily_review.acknowledge(store, "2026-11-01")
    assert daily_review.due(reopened) is None


def test_spring_gap_review_delivers_at_clock_jump_and_only_once(ithaca, tmp_path):
    now = [instant("2026-03-08T01:59:59-05:00")]
    store = PlannerStore(tmp_path, clock=lambda: now[0])
    task(store, "2026-03-08")
    daily_review.configure(store, "02:30", True, False)
    sent = []
    assert not daily_review.deliver(store, sent.append, desktop=False)
    now[0] += 1
    assert daily_review.deliver(store, sent.append, desktop=False)
    now[0] = instant("2026-03-08T03:30:00-04:00")
    assert not daily_review.deliver(store, sent.append, desktop=False)
    assert len(sent) == 1


def test_budget_resets_on_next_calendar_day():
    first = plan_window("I have 15 minutes", snapshot("2026-10-31T23:55:00-04:00"))
    assert first["available_minutes"] == 5
    after_midnight = snapshot("2026-11-01T00:02:00-04:00")
    revised = plan_window("What fits today?", after_midnight, first)
    assert revised["day"] == "2026-11-01"
    assert revised["budget_minutes"] is None
    assert revised["available_minutes"] == 25 * 60 - 2


def test_first_fold_notification_is_not_repeated_after_restart(ithaca, tmp_path):
    now = [instant("2026-11-01T01:30:00-04:00")]
    store = PlannerStore(tmp_path, clock=lambda: now[0])
    task(store, "2026-11-01")
    daily_review.configure(store, "01:30", True, False)
    sent = []
    assert daily_review.deliver(store, sent.append, desktop=False)
    now[0] = instant("2026-11-01T01:30:00-05:00")
    reopened = PlannerStore(tmp_path, clock=lambda: now[0])
    assert not daily_review.deliver(reopened, sent.append, desktop=False)
    assert [notification["day"] for notification in sent] == ["2026-11-01"]
