"""Deterministic user scenarios with real SQLite transactions, workers, timers and reports."""

from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from itertools import combinations
import multiprocessing
from threading import Event
from unittest.mock import patch

import estimate_worker as worker
import report_worker
from planner_store import PlannerStore


def add(store, title, *, prediction=None):
    return store.create_task(day="2026-11-15", title=title, description=f"Finish {title}.",
                             category="Work", estimate_minutes=30, prediction=prediction)["id"]


def test_rapid_submissions_timer_switch_and_completion_while_first_forecast_is_in_flight(tmp_path, monkeypatch):
    now = [1000]
    store = PlannerStore(tmp_path, clock=lambda: now[0])
    second_tab = PlannerStore(tmp_path, clock=store.clock)
    index = store.data_dir / "vectors"
    entered, release, reporting, release_report = (Event() for _ in range(4))
    forecasts, judged, events = [], [], []
    source_note = "Timestamp alignment took two iterations."
    report = {"summary": "Inspected the alignment trace.", "intervals": []}

    def predict(task, history, index_dir):
        forecasts.append((task["id"], [(r["id"], r["actual_seconds"], r["completion_note"], r["report"])
                                       for r in history]))
        events.append("forecast")
        if len(forecasts) == 1:
            entered.set()
            if not release.wait(10):
                raise TimeoutError("Scenario did not release initial forecast.")
        return {"point_min": 30 + len(forecasts), "explanation": "Controlled scenario forecast."}

    def decide(left, right, examples):
        pair = frozenset((left["id"], right["id"]))
        judged.append(pair)
        events.append("decision")
        return left["title"].startswith("GPU") and right["title"].startswith("GPU")

    def build_report(*args, **kwargs):
        reporting.set()
        if not release_report.wait(10):
            raise TimeoutError("Scenario did not release activity report.")
        return report

    monkeypatch.setattr(worker.estimator, "predict", predict)
    monkeypatch.setattr(worker.retrieval, "decide_related", decide)
    monkeypatch.setattr(worker.estimator, "index_current_task", lambda *args: None)
    monkeypatch.setattr(report_worker.activity, "build_report", build_report)
    # Simulate several minutes of user time without expiring a real worker's heartbeat.
    monkeypatch.setattr(worker, "_LEASE_SECONDS", 1800)

    with ThreadPoolExecutor(max_workers=1) as pool, ThreadPoolExecutor(max_workers=1) as reports:
        monkeypatch.setattr(worker, "_POOL", pool)
        first = add(store, "GPU profile analysis")
        worker.schedule_pending(store, index)
        try:
            assert entered.wait(5)
            batch = [add(second_tab, name) for name in ("GPU trace alignment", "Laundry", "GPU trace review")]
            source, laundry, review = batch
            for _ in batch:
                worker.schedule_pending(second_tab, index)
            assert [t for t, history in forecasts] == [first]
            assert all(second_tab.get_task(t)["estimate_updating"] for t in [first, *batch])

            second_tab.transition(laundry, "start")
            now[0] += 90
            second_tab.transition(source, "start")  # Automatically pauses laundry.
            now[0] += 120
            second_tab.transition(source, "pause")
            now[0] += 60
            second_tab.transition(source, "resume")
            now[0] += 240
            second_tab.transition(source, "stop")
            second_tab.transition(source, "finish", completion_note=source_note, queue_report=True)
            second_tab.transition(review, "start")
            report_job = reports.submit(report_worker.run_one_report, second_tab, False)
            assert reporting.wait(5)
            assert second_tab.get_task(source)["status"] == "completed"
            assert second_tab.get_task(source)["report"] is None

            # This submission can see the completed duration but its report is still generating.
            fifth = add(second_tab, "Book a dentist appointment")
            worker.schedule_pending(second_tab, index)
            release_report.set()
            assert report_job.result(timeout=5)
            assert second_tab.get_task(source)["report"] == report
        finally:
            release_report.set()
            release.set()
        pool.submit(lambda: None).result(timeout=10)  # All submissions share the same drain.

    assert events[:5] == ["forecast"] * 5
    assert [t for t, _ in forecasts[:5]] == [first, source, laundry, review, fifth]
    assert [history for _, history in forecasts[:5]] == [[], [], [], [], [(source, 360, source_note, None)]]
    assert Counter(t for t, _ in forecasts[5:]) == Counter([first, review])
    assert all(history == [(source, 360, source_note, report)] for _, history in forecasts[5:])
    for i, task_id in enumerate([first, source, laundry, review, fifth], 1):
        assert store.get_task(task_id)["creation_prediction"]["point_min"] == 30 + i

    # Four simultaneous tasks yield six pairs. The fifth joins after one is complete.
    expected = {frozenset(p) for p in combinations([first, source, laundry, review], 2)}
    expected.update(frozenset((fifth, t)) for t in [first, laundry, review])
    assert Counter(judged) == Counter({pair: 1 for pair in expected})
    with store._db() as db:
        matrix = {frozenset((a, b)): related for a, b, related in
                  db.execute("SELECT left_id,right_id,related FROM task_relations")}
    assert matrix == {pair: int(pair <= {first, source, review}) for pair in expected}
    assert not store.has_estimation_work()
    assert store.next_report_job() is None
    assert store.get_task(laundry)["status"] == "paused"
    assert store.running_task()["id"] == review
    assert store.periods(source) == [(1090, 1210), (1270, 1510)]
    assert store.get_task(laundry)["actual_seconds"] == 90


def _estimate_in_second_process(data_dir, competing, calls):
    """Use the normal worker entry point in a spawn-safe, independent Python interpreter."""
    store = PlannerStore(data_dir)
    real_lock = worker.portalocker.lock

    def observed_lock(handle, flags):
        competing.set()
        return real_lock(handle, flags)

    def predict(task, history, index_dir):
        calls.put(task["title"])
        return {"point_min": 30, "explanation": "Peer forecast."}

    with patch.object(worker.portalocker, "lock", observed_lock), \
         patch.object(worker.estimator, "predict", predict), \
         patch.object(worker.estimator, "index_current_task"):
        assert worker.run_one_job(store, data_dir / "vectors")


def test_independent_worker_waits_for_an_inflight_matrix_decision_before_starting_new_forecast(tmp_path, monkeypatch):
    store = PlannerStore(tmp_path)
    initial = {"point_min": 20, "explanation": "Earlier estimate."}
    a, b = [add(store, name, prediction=initial) for name in ["Draft meeting notes", "Revise meeting notes"]]
    entered, release = Event(), Event()
    context = multiprocessing.get_context("spawn")
    competing = context.Event()
    calls = context.SimpleQueue()

    def decide(left, right, examples):
        calls.put("begin pair")
        entered.set()
        if not release.wait(30):
            raise TimeoutError("Scenario did not release matrix call.")
        calls.put("end pair")
        return True

    monkeypatch.setattr(worker.retrieval, "decide_related", decide)
    peer = context.Process(target=_estimate_in_second_process, args=(tmp_path, competing, calls))
    with ThreadPoolExecutor(1) as primary:
        pair = primary.submit(worker.run_one_job, store, tmp_path / "vectors")
        assert entered.wait(5)
        pending = add(store, "Solve the next exercise")
        peer.start()
        try:
            assert competing.wait(20)
            store.transition(pending, "start")
            assert store.get_task(pending)["creation_prediction"] is None
            # Forget a task after the judge saw it. Its late answer must be discarded.
            store.delete_task(a)
            assert store.running_task()["id"] == pending
        finally:
            release.set()
            peer.join(10)
            if peer.is_alive():
                peer.terminate()
                peer.join(5)
        assert peer.exitcode == 0
        assert pair.result(timeout=5)

    order = []
    while not calls.empty():
        order.append(calls.get())
    assert order == ["begin pair", "end pair", "Solve the next exercise"]
    assert store.get_task(pending)["creation_prediction"]["explanation"] == "Peer forecast."
    assert worker.run_one_job(store, tmp_path / "vectors")
    assert not worker.run_one_job(store, tmp_path / "vectors")
    with store._db() as db:
        rows = db.execute("SELECT left_id,right_id,related FROM task_relations").fetchall()
    assert [(frozenset((left, right)), related) for left, right, related in rows] == [(frozenset((b, pending)), 1)]


def test_corrected_duration_and_report_while_a_legacy_forecast_waits_for_its_turn(tmp_path):
    now = [10_000]
    store = PlannerStore(tmp_path, clock=lambda: now[0])
    key = add(store, "Review the reference timings", prediction={"point_min": 30})
    store.transition(key, "finish", spent_seconds=1800, completion_note="Included an idle break.")
    target = add(store, "Review the next GPU timings")
    unrelated = add(store, "Pick up groceries")

    # This simulates pending forecasts saved before the editable-timer release.
    with store._db(write=True) as db:
        db.execute("UPDATE prediction_history SET snapshot=json_remove(snapshot,'$.actual_seconds','$.completion_note')")
        db.execute("DELETE FROM settings WHERE key='forecast_timer_snapshots'")
    reopened = PlannerStore(tmp_path, clock=store.clock)
    now[0] += 60
    reopened.transition(key, "adjust_time", spent_seconds=600)
    corrected_report = {"summary": "Ten minutes of GPU work."}
    report_job = reopened.next_report_job()
    assert reopened.save_report(key, corrected_report, generation=report_job["generation"])

    related = {"point_min": 30, "related_tasks": [{"task_id": key}]}
    # The judge saw an earlier feedback example about this source but rejected it.
    independent = {"point_min": 12, "related_tasks": [], "feedback_source_ids": [key]}
    # Both original submissions predate the correction, even across an app restart.
    for task_id, prediction in [(target, related), (unrelated, independent)]:
        job = reopened.claim_prediction()
        assert job["task"]["id"] == task_id
        source, = job["history"]
        assert (source["actual_seconds"], source["report"]) == (1800, None)
        assert reopened.resolve_prediction(task_id, job["token"], prediction)
    # Now that the forecast identifies its comparisons, only its affected task needs a refresh.
    refresh = reopened.claim_prediction()
    assert refresh["task"]["id"] == target
    source, = refresh["history"]
    assert (source["actual_seconds"], source["report"]) == (600, corrected_report)
    assert reopened.resolve_prediction(target, refresh["token"], dict(related, point_min=10))
    assert reopened.claim_prediction() is None
    assert reopened.get_task(target)["creation_prediction"]["point_min"] == 30

    # The source had already finished before this task was created. Later report
    # edits still propagate using the saved historical comparison.
    reopened.save_report(key, {"summary": "Reviewed the corrected GPU classification."})
    updated = reopened.claim_prediction()
    assert updated["task"]["id"] == target
    assert reopened.resolve_prediction(target, updated["token"], related)
    assert reopened.claim_prediction() is None
