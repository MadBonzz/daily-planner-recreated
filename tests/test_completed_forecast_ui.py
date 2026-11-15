import sys
import time

import pytest
from streamlit.testing.v1 import AppTest

from planner_store import PlannerStore


MODEL = "ternary-bonsai-2-4b-mlx"
ERROR = f"The model could not supply a valid forecast. {MODEL} returned an incomplete or invalid JSON object."


def forecast(minutes_by):
    direct = minutes_by == "model"
    prediction = {
        "point_min": 18 if direct else 15, "low_min": 14 if direct else 7,
        "high_min": 23, "confidence": "low",
        "provenance": {"provider": "Local endpoint", "model": MODEL,
                       "decision": "direct" if direct else "fallback", "minutes_by": minutes_by},
        "evidence": [], "cautions": [], "related_tasks": [], "used_task_ids": [],
    }
    if direct:
        prediction["model_rationale"] = "Allow time to find the unscented refill."
    else:
        prediction.update(retryable=True, model_error=ERROR, retrieval_error="The earlier history lookup failed.")
    return prediction


@pytest.mark.parametrize("minutes_by", ["user", "history", "model"])
def test_finishing_dish_soap_shows_recorded_time_and_the_last_precompletion_estimate(tmp_path, monkeypatch, minutes_by):
    # AppTest installs the simulated app as __main__. Restore the test runner for spawned processes.
    monkeypatch.setitem(sys.modules, "__main__", sys.modules["__main__"])
    store = PlannerStore(tmp_path)
    initial = forecast("user")
    task = store.create_task(title="Pick up dish soap", description="Buy the unscented refill on the way home.",
                             category="Home", day="2026-10-25", estimate_minutes=15, prediction=initial)
    task_id = task["id"]
    current = forecast(minutes_by)
    if current != initial:
        store.retry_estimate(task_id)
        lease = store.claim_prediction()
        assert store.resolve_prediction(task_id, lease["token"], current)

    monkeypatch.setattr("app.DATA_ROOT", tmp_path)
    monkeypatch.setattr("app.schedule_reports", lambda *args: None)
    monkeypatch.setattr("workspace_ui.offer_workspace", lambda *args: None)
    code = f"""
import streamlit as st
from app import render_task
from planner_store import PlannerStore
store = PlannerStore({str(tmp_path)!r}, clock=lambda: st.session_state["test_clock"])
render_task(store, store.data_dir / "vectors", store.get_task({task_id!r}), dev=True)
"""
    ui = AppTest.from_string(code)
    started_at = time.time()
    ui.session_state["test_clock"] = started_at
    ui.run()
    ui.button(key=f"start-{task_id}").click().run()
    assert not ui.exception and store.get_task(task_id)["status"] == "running"
    ui.session_state["test_clock"] = started_at + 5
    ui.button(key=f"pause-{task_id}").click().run()
    assert not ui.exception
    paused = store.get_task(task_id)
    assert paused["status"] == "paused" and paused["actual_seconds"] == 5
    assert paused["prediction"] == current
    assert any("Recorded 5s" in caption.value for caption in ui.caption)

    if minutes_by != "model":
        assert any(ERROR in item.value for item in ui.warning)
    ui.button(key=f"finish-{task_id}").click().run()
    ui.number_input[0].set_value(20)
    next(button for button in ui.button if button.label == "Yes, finish task").click().run()
    assert not ui.exception
    completed = store.get_task(task_id)
    assert completed["status"] == "completed" and completed["actual_seconds"] == 1200
    assert completed["manual_seconds"] == 1195
    assert store.capture_periods(task_id) == [(started_at, started_at + 5)]
    assert completed["prediction"] == current and completed["creation_prediction"] == initial

    captions = [item.value for item in ui.caption]
    assert any("Time spent 20m 00s (user-adjusted)" in caption for caption in captions)
    assert not any(ERROR in item.value or "history lookup failed" in item.value for item in ui.warning)
    explanation = next(panel for panel in ui.expander if panel.label == "Why this estimate?")
    detail = "\n".join(item.value for item in explanation.caption)
    if minutes_by == "model":
        assert ERROR not in detail and any(f"Model forecast · Local endpoint · {MODEL}" in caption for caption in captions)
        assert any("18 min" in caption and "Time spent 20m" in caption for caption in captions)
    else:
        assert any("Planning estimate 15 min" in caption and "Time spent 20m" in caption for caption in captions)
        source = "your own budget" if minutes_by == "user" else "recorded durations from related tasks"
        assert any(f"Planning estimate used {source}." in caption for caption in captions)
        assert ERROR in detail and "The earlier history lookup failed." in detail
