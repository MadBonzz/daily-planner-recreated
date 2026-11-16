import json
from pathlib import Path

import pytest

from examples.estimation.activity_fixture import task_layout
from planner_store import PlannerStore


FIXTURES = Path(__file__).resolve().parents[1] / "data" / "tasks"
HISTORIES = sorted(path.stem for path in (FIXTURES / "histories").glob("*.json"))


def test_every_bundled_history_has_one_schedule():
    assert sorted(path.stem for path in (FIXTURES / "schedules").glob("*.json")) == HISTORIES


@pytest.mark.parametrize("history_id", HISTORIES)
def test_events_preserve_timer_and_activity_windows_and_do_not_leak_holdouts(tmp_path, history_id):
    history = json.loads((FIXTURES / "histories" / f"{history_id}.json").read_text(encoding="utf-8"))
    schedule = json.loads((FIXTURES / "schedules" / f"{history_id}.json").read_text(encoding="utf-8"))
    rows = {row["id"]: row for row in history["tasks"] + history["open_tasks"]}
    layout = task_layout(history)
    slots = [(event["day_offset"], event["minute"]) for event in schedule]
    assert slots == sorted(slots)
    assert max(d for d, m in slots) == 0 and len({d for d, m in slots}) >= 4
    assert {event["task"] for event in schedule} == set(rows)
    assert all(set(e) == {"day_offset", "minute", "action", "task"} for e in schedule)
    assert all(type(d) is type(m) is int and d <= 0 and 0 <= m < 1440 for d, m in slots)

    store = PlannerStore(tmp_path, clock=lambda: clock)
    ids = {}
    active = None
    open_tasks = set()
    overlaps_completion = False
    for event in schedule:
        key, action = event["task"], event["action"]
        row = rows[key]
        clock = (event["day_offset"] * 1440 + event["minute"]) * 60
        if action == "create":
            assert key not in ids
            task = store.create_task(title=row["title"], description=row["description"], day=row["day"],
                                     estimate_minutes=row["estimate_minutes"], category=row["category"], prediction={})
            ids[key] = task["id"]
            open_tasks.add(key)
            continue
        if action in ("start", "resume"):
            assert active is None, f"{key} overlaps {active}"
            active = key
        elif action in ("pause", "stop"):
            assert active == key
            active = None
        elif action == "finish":
            open_tasks.remove(key)
            overlaps_completion |= bool(open_tasks)
        store.transition(ids[key], action, completion_note=row.get("completion_note", "") if action == "finish" else "")
    assert overlaps_completion
    if active:
        # The live sample ends after the final open slice of the running task.
        start, end = layout[active]["sessions"][-1][-1]
        clock += end - start
    for key, row in rows.items():
        task = store.get_task(ids[key])
        assert task["status"] == row.get("status", "completed")
        assert task["actual_seconds"] == sum(row["sessions_minutes"]) * 60
        if task["status"] == "completed":
            assert task["completion_note"] == row.get("completion_note", "")
        if key in layout:
            # The same constant applied to every boundary preserves watcher alignment,
            # the distance between sessions, and any pauses inside a session.
            original = [p for group in layout[key]["sessions"] for p in group]
            observed = [(start, clock if stop is None else stop) for start, stop in store.periods(ids[key])]
            assert len(original) == len(observed)
            delta = observed[0][0] - original[0][0]
            assert observed == [(a+delta, b+delta) for a, b in original]
