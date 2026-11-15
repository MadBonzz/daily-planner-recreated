import json
import threading

import activity
from planner_store import PlannerStore
import report_worker


def task(store, title="Review points from the lecture"):
    return store.create_task(title=title, description="Watch the lecture", estimate_minutes=5,
                             category="Study", prediction={}, day="2026-01-14")


def finish(store, task_id):
    for action in ("start", "stop"):
        store.transition(task_id, action)
    return store.transition(task_id, "finish", queue_report=True)


def row(label="off_task"):
    return {"start": 0, "end": 50, "period": 0, "seconds": 50, "app": "browser",
            "domain": "lectures.example", "title": "Fourier series", "phone_app": "",
            "idle": False, "excluded": False, "label": label, "reason": "Lecture",
            "label_source": "model"}


def test_worker_never_blocks_finish_and_replaced_job_cannot_publish(tmp_path, monkeypatch):
    store = PlannerStore(tmp_path)
    source = task(store)
    started = threading.Event()
    release = threading.Event()

    def build(*args, **kwargs):
        started.set()
        assert release.wait(5)
        return {"summary": "obsolete"}

    monkeypatch.setattr(activity, "build_report", build)
    saved = finish(store, source["id"])
    assert saved["status"] == "completed" and saved["report"] is None
    worker = threading.Thread(target=report_worker.run_one_report, args=(store, False))
    worker.start()
    try:
        assert started.wait(5)
        reopened = PlannerStore(tmp_path)
        reopened.request_report(source["id"])
        assert reopened.get_task(source["id"])["status"] == "completed"
    finally:
        release.set()
        worker.join(5)
    assert reopened.get_task(source["id"])["report"] is None
    monkeypatch.setattr(activity, "build_report", lambda *args, **kwargs: {"summary": "current"})
    assert report_worker.run_one_report(reopened, False)
    assert reopened.get_task(source["id"])["report"] == {"summary": "current"}
    assert reopened.next_report_job() is None


def test_user_reviews_revise_totals_and_teach_only_observed_activity(tmp_path, monkeypatch):
    store = PlannerStore(tmp_path)
    source = task(store)
    finish(store, source["id"])
    event = row()
    hidden = {**event, "app": "", "title": "", "domain": "", "excluded": True, "label": "uncertain"}
    initial = activity._summarize({"intervals": [event, hidden], "warnings": []})
    store.save_report(source["id"], initial, generation=store.next_report_job()["generation"])
    assert len(activity.review_groups(initial)) == 1
    store.request_report(source["id"], {activity.activity_key(event): "on_task"})
    assert report_worker.run_one_report(store, False)
    report = store.get_task(source["id"])["report"]
    assert report["totals"] == {"on_task": 50.0, "off_task": 0, "uncertain": 50.0}
    assert activity.reviewed_examples(report)[0]["label"] == "on_task"

    future = task(store, "Review next Fourier lecture")
    overrides, examples = store.report_feedback(future)
    assert len(examples) == 1 and examples[0]["source_id"] == source["id"]
    sent = []

    def model(messages, schema, **kwargs):
        sent.extend(json.loads(messages[1]["content"])["personal_examples"])
        return {"0": {"label": "on_task", "reason": "Lecture matches assignment."}}

    monkeypatch.setattr(activity, "chat_json", model)
    monkeypatch.setattr(activity, "_collect", lambda *args, **kwargs: {"intervals": [row()], "warnings": []})
    rebuilt = activity.build_report(future, [], examples=examples, overrides=overrides)
    assert rebuilt["feedback_source_ids"] == [source["id"]] and sent == examples
    assert rebuilt["intervals"][0]["label_source"] == "model"
    # Explicit corrections also survive a fresh rebuild from workstation events.
    overrides, _ = store.report_feedback(source)
    assert activity.build_report(source, [], overrides=overrides)["intervals"][0]["label_source"] == "user"
    store.set_setting("exclusions", ["Fourier series"])
    assert store.report_feedback(future)[1] == []


def test_continue_later_keeps_one_paused_session_and_unknown_time_out(tmp_path):
    now = [1000]
    store = PlannerStore(tmp_path, clock=lambda: now[0])
    task_id = task(store)["id"]
    store.transition(task_id, "start")
    now[0] = 1010
    store.transition(task_id, "stop")
    now[0] = 2010
    assert store.transition(task_id, "continue_later")["status"] == "paused"
    assert store.transition(task_id, "continue_later")["actual_seconds"] == 10
    store.transition(task_id, "resume")
    now[0] = 2015
    store.transition(task_id, "stop")
    assert store.transition(task_id, "finish")["actual_seconds"] == 15
    with store._db() as db:
        assert db.execute("SELECT COUNT(*) FROM sessions WHERE task_id=?", (task_id,)).fetchone()[0] == 1


def test_completed_card_shows_progress_then_editable_classifications(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest

    store = PlannerStore(tmp_path)
    task_id = task(store)["id"]
    finish(store, task_id)
    script = f"""
from app import activity_section
from planner_store import PlannerStore
store = PlannerStore({str(tmp_path)!r})
activity_section(store, store.get_task({task_id!r}), True)
"""
    monkeypatch.setattr(report_worker, "schedule_reports", lambda *args: None)
    import app
    monkeypatch.setattr(app, "schedule_reports", lambda *args: None)
    card = AppTest.from_string(script).run()
    assert not card.exception
    assert card.info[0].value == "Generating activity report…"
    event = row()
    report = activity._summarize({"intervals": [event], "warnings": []})
    store.save_report(task_id, report, generation=store.next_report_job()["generation"])
    card.run()
    assert not card.exception
    labels = [button.label for button in card.button]
    save = labels.index("Save classifications and regenerate report")
    assert "Rebuild report" in labels
    card.session_state[f"labels-{task_id}"] = {
        "edited_rows": {0: {"Classification": "Work"}}, "added_rows": [], "deleted_rows": []}
    card.button[save].click().run()
    assert not card.exception and store.report_job(task_id)["action"] == "review"
    assert report_worker.run_one_report(store, True)
    assert store.get_task(task_id)["report"]["totals"]["on_task"] == 50


def test_replay_controls_and_daily_plan_do_not_start_live_jobs(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest
    import app

    def unexpected(*args):
        raise AssertionError("Live worker scheduled from timeline replay")

    monkeypatch.setattr(app, "DATA_ROOT", tmp_path)
    monkeypatch.setattr(app, "schedule_reports", unexpected)
    monkeypatch.setattr(app, "schedule_pending", unexpected)
    monkeypatch.setattr(app.estimator, "predict", lambda *args, **kwargs: {
        "point_min": 15, "low_min": 12, "high_min": 20, "confidence": "low",
        "basis": "Test estimate", "explanation": "Estimated from scope."})
    ui = AppTest.from_string("from app import main\nmain()", default_timeout=20)
    ui.session_state["planner_mode"] = "Dev"
    ui.session_state["dev_history_view"] = "Chronological replay"
    ui.run()
    assert not ui.exception
    assert ui.button(key="replay-step") and ui.button(key="replay-run")
    ui.button(key="replay-step").click().run()
    assert not ui.exception
    assert ui.get("progress")[0].proto.text.startswith("1 of ")
    labels = {button.label for button in ui.button}
    assert not labels.intersection({"Start", "Resume", "Stop", "Re-estimate", "Edit relevant tasks",
                                    "＋ Add task", ":material/delete:"})
    assert ui.get("dataframe")
