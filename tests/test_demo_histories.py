"""Exercise the complete picker-to-dashboard path for the five interactive calendars."""

from datetime import datetime, timedelta
import json
from types import SimpleNamespace

import pytest

import activity
import dev_workspace
from examples.estimation.run import DEMO_DATASET, histories
from planner_analytics import analytics_rows


def test_picker_keeps_all_ten_benchmark_histories_out_of_the_five_demo_options():
    published = {h["id"] for h in histories(None)}
    visible = set(dev_workspace.history_choices())
    demo_files = {p.stem for p in (DEMO_DATASET / "histories").glob("*.json")}
    assert len(published) == 10 and len(visible) == 5
    assert visible == demo_files and published.isdisjoint(visible)


@pytest.mark.parametrize("profile", sorted(dev_workspace.history_choices()))
def test_historical_notes_timers_and_both_devices_survive_loading_and_adding(tmp_path, monkeypatch, profile):
    anchor = datetime(2026, 11, 15, 20)
    monkeypatch.setattr(dev_workspace, "time", SimpleNamespace(time=anchor.timestamp))
    monkeypatch.setattr(activity, "_get", lambda *_a, **_k: pytest.fail("Demo read a personal ActivityWatch server"))
    history = json.loads((DEMO_DATASET / "histories" / (profile + ".json")).read_text())
    events = json.loads((DEMO_DATASET / "schedules" / (profile + ".json")).read_text())
    created = {row["task"]: row["day_offset"] for row in events if row["action"] == "create"}
    store = dev_workspace.open_dev_workspace(tmp_path, profile)
    store.clock = anchor.timestamp
    completed = store.completed_tasks()
    tasks, logs = analytics_rows(completed, [])
    assert len(tasks) == len(history["tasks"])
    assert {log["classification"] for log in logs} >= {"on_task", "off_task", "uncertain"}
    assert any("Phone" in log["device"] or "phone" in log["device"] for log in logs)
    assert any("Laptop" in log["device"] for log in logs)
    for row in history["tasks"]:
        saved = store.get_task(store.get_setting("dev_task_" + row["id"]))
        expected_day = (anchor.date() + timedelta(days=created[row["id"]])).isoformat()
        assert saved["id"] in {t["id"] for t in store.list_tasks(expected_day)}
        assert (saved["description"], saved["completion_note"]) == (row["description"], row.get("completion_note", ""))
        assert saved["actual_seconds"] == pytest.approx(row["actual_minutes"] * 60)
        assert bool(saved["attachments"]) == bool(row.get("pdf"))
        assert sum(e["seconds"] for e in saved["report"]["intervals"]) == pytest.approx(saved["actual_seconds"])
    for row in history["open_tasks"]:
        saved = store.get_task(store.get_setting("dev_task_" + row["id"]))
        assert (saved["status"], saved["day"]) == (row["status"], anchor.date().isoformat())
    offered = dev_workspace.sample_task_choices(profile)
    assert set(offered) == {t["id"] for t in history["holdouts"]}
    assert all(store.get_setting("dev_task_" + key) is None for key in offered)
    for key, row in offered.items():
        task = dev_workspace.add_sample_task(store, profile, key, anchor.date().isoformat())
        assert task["title"] == row["title"] and task["status"] == "ready"
        assert task["actual_seconds"] == 0 and not task["completion_note"] and not task["report"]
        assert bool(task["attachments"]) == bool(row.get("pdf"))
    assert {task["id"] for task in completed} == {task["id"] for task in store.completed_tasks()}
