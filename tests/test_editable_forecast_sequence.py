"""Open-card priority, late matrix decisions and reopening a live history."""
import sys
from datetime import date

from streamlit.testing.v1 import AppTest

import app
import dev_workspace
from planner_store import PlannerStore


def forecast(minutes):
    return {"point_min": minutes, "low_min": minutes - 2, "high_min": minutes + 2,
            "confidence": "medium", "provenance": {"decision": "direct", "minutes_by": "model"}}


def test_nora_open_forecasts_and_completion_take_priority_over_historical_jobs(tmp_path):
    store = dev_workspace.open_dev_workspace(tmp_path, "d1-systems-student")
    day = store.get_setting("dev_calendar", {})["day"]
    visible = {task["id"]: task for task in store.list_tasks(day)}
    completed = {task["id"] for task in store.completed_tasks()}
    assert len(visible) == 6
    with store._db() as db:
        assert db.execute("SELECT count(*) FROM prediction_jobs").fetchone()[0] > len(visible)
    for index in range(len(visible)):
        job = store.claim_prediction()
        task = job["task"]
        assert task["id"] in visible
        assert set(row["id"] for row in job["history"]) <= completed
        assert store.get_task(task["id"])["estimate_in_progress"]
        assert store.resolve_prediction(task["id"], job["token"], forecast(task["estimate_minutes"]))
        assert store.daily_forecast_totals(day)["pending"] == len(visible) - index - 1
    by_name = {row["title"]: row["id"] for row in visible.values()}
    target = by_name["Attach the minimized trace to the lab issue"]
    for label, actual, related in [("Pick up dish soap", 20, False), ("Minimize the marker-word failure", 110, True)]:
        source = by_name[label]
        if store.get_task(source)["status"] == "ready":
            store.transition(source, "start")
            store.transition(source, "pause")
        store.transition(source, "finish", spent_seconds=actual * 60)
        store = dev_workspace.open_dev_workspace(tmp_path, "d1-systems-student")
        previous = store.get_task(target)["prediction"]
        while store.get_task(target)["awaiting_related_update"] and not store.get_task(target)["estimate_updating"]:
            assert store.claim_prediction() is None  # Runnable related decisions precede historical backfill.
            pair = store.claim_relation()
            assert pair
            store.resolve_relation(pair["pair"], pair["token"], related and target in pair["pair"] and source in pair["pair"])
        assert store.get_task(target)["estimate_updating"] is related
        if related:
            refresh = store.claim_prediction()
            assert refresh["task"]["id"] == target
            assert any(r["id"] == source and r["actual_seconds"] == actual * 60 for r in refresh["history"])
            assert store.resolve_prediction(target, refresh["token"], forecast(82))
            assert store.get_task(target)["prediction"]["point_min"] == 82
        else:
            assert store.get_task(target)["prediction"] == previous


def test_idle_dashboard_observes_new_forecast_without_reloading(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "__main__", sys.modules["__main__"])
    store = PlannerStore(tmp_path / "live")
    row = store.create_task(day=date.today(), title="Attach the trace", description="Include the reproduction.",
                            estimate_minutes=55, prediction=forecast(55))
    original = app.st.fragment
    callbacks = []
    def record(*args, **kwargs):
        decorate = original(*args, **kwargs)
        def register(func):
            if func.__qualname__.startswith("live_estimate_section."):
                callbacks.append(func)
            return decorate(func)
        return register
    monkeypatch.setattr(app.st, "fragment", record)
    monkeypatch.setattr(app, "DATA_ROOT", tmp_path)
    monkeypatch.setattr("workspace_ui.offer_workspace", lambda *_: None)
    page = AppTest.from_string("from app import main\nmain()").run()
    assert not page.exception and any("Planner 55 min" in e.value for e in page.caption)
    assert len(callbacks) == 1
    store.retry_estimate(row["id"])
    job = store.claim_prediction()
    assert store.resolve_prediction(row["id"], job["token"], forecast(82))
    tick = AppTest.from_string("import streamlit as st\nfrom app import live_daily_totals\n"
                               "from datetime import date\nst.session_state.tick()\n"
                               "live_daily_totals(st.session_state.store, date.today())")
    tick.session_state.tick = callbacks[0]
    tick.session_state.store = store
    tick.run()
    assert not tick.exception
    assert any("Planner 82 min" in e.value for e in tick.caption)
    assert any("<small>Planner budget</small><strong>82 min</strong>" in e.proto.body for e in tick.get("html"))
