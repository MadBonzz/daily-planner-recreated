"""End-to-end Dev replay with real timers, SQLite watcher reads and deterministic model answers."""

from datetime import date, datetime, timedelta
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import activity
from activity_db import SQLiteActivitySource
import dev_schedule as schedule
import dev_workspace as dev
import estimator
import retrieval


def load(directory, kind, profile):
    return json.loads((dev.sample_root(profile, directory) / kind / (profile + ".json")).read_text(encoding="utf-8"))


def assert_timers(store, history):
    actual = {t["title"]: t for t in store.list_tasks()}
    assert set(actual) == {row["title"] for row in history["tasks"] + history["open_tasks"]}
    for row in history["tasks"] + history["open_tasks"]:
        saved = actual[row["title"]]
        status = row.get("status", "completed")
        assert saved["status"] == status
        assert saved["actual_seconds"] == pytest.approx(sum(row["sessions_minutes"]) * 60, abs=0.01)
        assert saved["completion_note"] == row.get("completion_note", "")
        if status == "completed":
            assert sum(saved["report"]["totals"].values()) == pytest.approx(saved["actual_seconds"], abs=0.01)


def open_snapshot_at(directory, profile, opened_at, monkeypatch):
    # Keep sample initialization and reads of its ongoing timers on the same local clock.
    clock = opened_at.timestamp
    monkeypatch.setattr(dev, "time", SimpleNamespace(time=clock))
    store = dev.open_dev_workspace(directory, profile)
    store.clock = clock
    return store


@pytest.mark.parametrize("profile", [p.stem for p in sorted((dev.DATASET / "histories").glob("*.json"))])
def test_snapshots_anchor_packaged_schedules_and_preserve_all_timers(tmp_path, profile, monkeypatch):
    monkeypatch.setattr(activity, "_get", lambda *_a, **_k: pytest.fail("Dev contacted a personal watcher"))
    history = load(dev.DATASET, "histories", profile)
    store = open_snapshot_at(tmp_path, profile, datetime(2026, 11, 15, 20), monkeypatch)
    state = schedule.progress(store, profile)
    events = state["events"]
    assert len(events) == state["total"] == state["done"] and all(e["applied"] for e in events)
    source = load(dev.DATASET, "schedules", profile)
    anchor = date.fromisoformat(store.get_setting("dev_calendar")["day"])
    for saved, step in zip(events, source, strict=True):
        assert saved["day"] == (anchor + timedelta(days=step["day_offset"])).isoformat()
    assert len({e["day"] for e in events}) >= 3
    assert all(store.get_task(store.get_setting(f'dev_task_{r["id"]}'))["day"] == anchor.isoformat()
               for r in history["open_tasks"])
    assert_timers(store, history)
    assert any(row.get("app") for task in store.completed_tasks() for row in task["report"]["intervals"])


def test_editable_snapshot_opened_after_midnight_shows_both_activity_dates(tmp_path, monkeypatch):
    profile = "06-neighborhood-clinic"
    opened_at = datetime(2026, 11, 15, 0, 5)
    store = open_snapshot_at(tmp_path, profile, opened_at, monkeypatch)
    flyer = store.get_task(store.get_setting("dev_task_paused-flyer"))

    # The task is on today's plan, but nine minutes of work straddle midnight.
    assert flyer["day"] == "2026-11-15" and flyer["status"] == "paused"
    assert store.periods(flyer["id"]) == [(datetime(2026, 11, 14, 23, 56).timestamp(), opened_at.timestamp())]
    work = [(event["action"], event["day"], event["time"]) for event in schedule.progress(store, profile)["events"]
            if event["task"] == flyer["title"] and event["action"] in {"start", "pause"}]
    assert work == [("start", "2026-11-14", "23:56"), ("pause", "2026-11-15", "00:05")]


@pytest.fixture
def model_stubs(monkeypatch):
    calls = {"forecasts": [], "reports": [], "observations": set()}

    def forecast(task, history, index_dir):
        selected = [t for t in history if "Mara" in t["title"] and "Mara" in task["title"]]
        calls["forecasts"].append((dict(task), list(history)))
        point = task["estimate_minutes"] + len(selected)
        return {"point_min": point, "low_min": point - 2, "high_min": point + 3, "confidence": "medium",
                "explanation": "Deterministic test forecast.", "basis": "Model forecast", "evidence": [],
                "provenance": {"minutes_by": "model", "decision": "direct"}, "seen_titles": [t["title"] for t in history]}

    def classify(messages, schema, **kwargs):
        payload = json.loads(messages[-1]["content"])
        calls["reports"].append(payload)
        return {str(row["id"]): {"label": "on_task", "reason": "The observed app supports the sample task."}
                for row in payload["intervals"]}

    get = SQLiteActivitySource.get

    def logged(self, path, **kwargs):
        result = get(self, path, **kwargs)
        if path.endswith("/events"):
            calls["observations"].update(r["data"]["title"] for r in result if (r["data"] or {}).get("title"))
        return result

    monkeypatch.setattr(estimator, "predict", forecast)
    monkeypatch.setattr(retrieval, "decide_related", lambda a, b, **_kw: "Mara" in a["title"] and "Mara" in b["title"])
    monkeypatch.setattr(estimator, "index_current_task", lambda *_args: None)
    monkeypatch.setattr(activity, "chat_json", classify)
    monkeypatch.setattr(activity, "_get", lambda *_a, **_k: pytest.fail("Dev contacted a personal watcher"))
    monkeypatch.setattr(SQLiteActivitySource, "get", logged)
    return calls


def test_replay_uses_only_completed_history_and_resumes_at_the_next_action(tmp_path, model_stubs):
    profile = "06-neighborhood-clinic"
    history = load(dev.DATASET, "histories", profile)
    store = dev.open_dev_workspace(tmp_path, profile, replay=True)
    assert store.list_tasks() == [] and schedule.progress(store, profile)["done"] == 0
    schedule.advance(store, profile)
    first = store.list_tasks()[0]
    assert not store.completed_tasks() and first["actual_seconds"] == 0 and not first["completion_note"]
    assert len(model_stubs["forecasts"]) == 1 and model_stubs["forecasts"][0][1] == []
    initial_cursor = schedule.progress(store, profile)["done"]
    reopened = dev.open_dev_workspace(tmp_path, profile, replay=True)
    assert reopened.path == store.path and schedule.progress(reopened, profile)["done"] == initial_cursor == 1
    schedule.advance(reopened, profile, through_end=True)
    state = schedule.progress(reopened, profile)
    assert state["done"] == state["total"] and all(e["applied"] for e in state["events"])
    assert_timers(reopened, history)

    # Validate every saved initial forecast and refresh against the exact schedule prefix.
    # The zero-duration 'finish' and following 'create' may share a timestamp.
    done, initial = set(), {}
    for event in state["events"]:
        if event["action"] == "create":
            initial[event["task"]] = done.copy()
        if event["action"] == "finish":
            done.add(event["task"])
    refreshed = []
    for task in reopened.list_tasks():
        timeline = reopened.forecast_timeline(task["id"])
        assert [e["at"] for e in timeline] == sorted(e["at"] for e in timeline)
        assert len([e for e in timeline if e["initial"]]) == 1
        first_entry = timeline[0]
        assert first_entry["initial"] and first_entry["prediction"] == task["creation_prediction"]
        if not task["fixed_duration"]:
            assert set(first_entry["prediction"]["seen_titles"]) == initial[task["title"]]
        for entry in timeline[1:]:
            assert not entry["initial"]
            assert task["title"] not in entry["prediction"]["seen_titles"]
            refreshed.append((task["title"], first_entry["prediction"]["point_min"], entry["prediction"]["point_min"]))
    assert any("Mara" in title and new > old for title, old, new in refreshed)
    assert all(not ({"report", "actual_seconds", "actual_minutes", "completion_note", "sessions_minutes"} & task.keys())
               for task, _ in model_stubs["forecasts"])
    assert model_stubs["reports"] and model_stubs["observations"]
    for payload in model_stubs["reports"]:
        assert not ({"actual_minutes", "actual_seconds", "completion_note", "sessions_minutes"} & payload["task"].keys())
        assert all("label" not in interval for interval in payload["intervals"])
    sent_titles = {row["title"] for payload in model_stubs["reports"] for row in payload["intervals"]}
    assert model_stubs["observations"] & sent_titles
    assert any(e["label_source"] == "model" for t in reopened.completed_tasks() for e in t["report"]["intervals"])
    counts = len(model_stubs["forecasts"]), len(model_stubs["reports"])
    schedule.advance(dev.open_dev_workspace(tmp_path, profile, replay=True), profile, through_end=True)
    assert counts == (len(model_stubs["forecasts"]), len(model_stubs["reports"]))
    assert dev.open_dev_workspace(tmp_path, profile).path != reopened.path


def test_dashboard_opens_replay_steps_and_shows_history_logs(tmp_path, monkeypatch, model_stubs):
    from streamlit.testing.v1 import AppTest
    import estimate_worker
    import report_worker

    # Exercise the original clinic timeline through the dashboard with its archived catalog.
    monkeypatch.setattr(dev, "DEMO_DATASET", dev.DATASET)
    monkeypatch.setenv("PLANNER_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(estimate_worker, "schedule_pending", lambda *_: None)
    monkeypatch.setattr(report_worker, "schedule_reports", lambda *_: None)
    ui = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py", default_timeout=20).run()
    next(w for w in ui.get("segmented_control") if w.key == "planner_mode").set_value("Dev").run()
    assert not ui.exception
    ui.selectbox(key="dev_history").set_value("06-neighborhood-clinic").run()
    ui.radio(key="dev_history_view").set_value("Chronological replay").run()
    assert not ui.exception
    assert ui.button(key="replay-step")
    ui.button(key="replay-step").click().run()
    assert not ui.exception
    assert any(e.label == "Forecast history" for e in ui.expander)
    for _ in range(5):
        ui.button(key="replay-step").click().run()
        assert not ui.exception
    # Mara's first meeting was scheduled a day before the reported work.
    assert any(e.value == "Also worked on this day" for e in ui.subheader)
    note = load(dev.DATASET, "histories", "06-neighborhood-clinic")["tasks"][0]["completion_note"]
    assert any(t.value == note for t in ui.text)
    next(t for t in ui.toggle if t.key.startswith("sample-log-")).set_value(True).run()
    assert not ui.exception
    assert any("Window or tab" in t.value.columns and "Assessment" not in t.value.columns for t in ui.dataframe)
    ui.button(key="replay-run").click().run()
    assert not ui.exception
    assert not any(b.key == "replay-step" for b in ui.button)
    next(w for w in ui.get("segmented_control") if w.key == "planner_view").set_value("Analytics & logs").run()
    assert not ui.exception
    selectors = {s.label: s for s in ui.selectbox}
    assert len(selectors["Activity day"].options) > 1 and len(selectors["Task"].options) > 1
    assert any("Window" in table.value.columns and "Label source" in table.value.columns
               for table in ui.dataframe)


def test_replay_reclaims_abandoned_forecast_at_original_event_time(tmp_path, model_stubs, monkeypatch):
    import estimate_worker

    profile = "06-neighborhood-clinic"
    store = dev.open_dev_workspace(tmp_path, profile, replay=True)

    def interrupted(*args):
        raise InterruptedError("Stopped after obtaining a forecast lease.")

    with monkeypatch.context() as patch:
        patch.setattr(estimate_worker, "_process", interrupted)
        with pytest.raises(InterruptedError):
            schedule.advance(store, profile)
    task = store.list_tasks()[0]
    created = store.get_setting("dev_calendar")["clock"]
    assert task["estimate_updating"] and store.claim_prediction() is None
    resumed = dev.open_dev_workspace(tmp_path, profile, replay=True)
    schedule.advance(resumed, profile)
    entry, = resumed.forecast_timeline(task["id"])
    assert entry["initial"] and entry["at"] == created


@pytest.mark.parametrize("profile", [p.stem for p in sorted((dev.DATASET / "histories").glob("*.json"))])
def test_replay_judges_all_overlaps_and_preserves_initial_history(tmp_path, profile, model_stubs, monkeypatch):
    monkeypatch.setattr(retrieval, "decide_related", lambda a,b,**kw: a["category"] == b["category"])
    store = dev.open_dev_workspace(tmp_path, profile, replay=True)
    schedule.advance(store, profile, through_end=True)
    tasks = {t["title"]: t for t in store.list_tasks()}
    open_ids, finished_titles, expected_pairs = set(), set(), set()
    for event in schedule.progress(store, profile)["events"]:
        task = tasks[event["task"]]
        if event["action"] == "create":
            expected_pairs.update(tuple(sorted((task["id"], other))) for other in open_ids)
            open_ids.add(task["id"])
            timeline = store.forecast_timeline(task["id"])
            assert sum(v["initial"] for v in timeline) == 1 and timeline[0]["initial"]
            if not task["fixed_duration"]:
                assert set(timeline[0]["prediction"]["seen_titles"]) == finished_titles
        elif event["action"] == "finish":
            open_ids.remove(task["id"])
            finished_titles.add(task["title"])
    categories = {t["id"]: t["category"] for t in tasks.values()}
    with store._db() as db:
        matrix = {(a,b): value for a,b,value in db.execute("SELECT left_id,right_id,related FROM task_relations")}
    assert set(matrix) == expected_pairs
    for (a,b), value in matrix.items():
        assert value == (categories[a] == categories[b])
