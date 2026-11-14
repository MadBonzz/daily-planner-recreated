from contextlib import closing
import threading
import time

import pytest
from qdrant_client import QdrantClient

from estimate_worker import schedule_pending
import estimate_worker as worker
from planner_store import PlannerStore


def create(store, **overrides):
    return store.create_task(**{
        "day": "2026-11-18", "title": "Read a proof", "description": "Derive the bound.",
        "estimate_minutes": 25, "category": "Reading", **overrides,
    })


def drain(store):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        with worker._LOCK:
            if store.path.resolve() not in worker._ACTIVE:
                break
        time.sleep(0.01)
    else:
        raise AssertionError("Background estimate did not finish.")
    assert not store.has_pending_predictions()


@pytest.mark.parametrize("finish_before_prediction", [False, True])
def test_pdf_worker_runs_without_blocking_timer_or_duplicating_calls(tmp_path, monkeypatch, finish_before_prediction):
    store = PlannerStore(tmp_path / "live")
    task_id = create(store, attachments=[("proof.pdf", b"%PDF-test")])["id"]
    parsing = threading.Event()
    release = threading.Event()
    calls = []

    def parse(content):
        assert content == b"%PDF-test"
        parsing.set()
        assert release.wait(5)
        return "Page 2: Derive the bound."

    def predict(task, history, index_dir):
        calls.append((task["id"], task["pdf_text"], history))
        return dict(point_min=35, low_min=20, high_min=50, explanation="Derive two lemmas.")

    monkeypatch.setattr(worker, "extract_pdf_text", parse)
    monkeypatch.setattr(worker.estimator, "predict", predict)
    index = store.data_dir / "vectors"
    schedule_pending(store, index)
    try:
        assert parsing.wait(2)
        assert store.get_task(task_id)["prediction"]["status"] == "pending"
        assert store.transition(task_id, "start")["status"] == "running"
        if finish_before_prediction:
            store.transition(task_id, "stop")
            store.transition(task_id, "finish")
        for _ in range(5):
            schedule_pending(store, index)
    finally:
        release.set()
    drain(store)
    assert calls == [(task_id, "Page 2: Derive the bound.", [])]
    result = store.get_task(task_id)
    assert result["pdf_text"] == "Page 2: Derive the bound."
    status = "completed" if finish_before_prediction else "running"
    assert (result["status"], result["prediction"]["point_min"]) == (status, 35)
    with closing(QdrantClient(path=str(index))) as client:
        rows, _ = client.scroll(worker.estimator.COLLECTION, with_vectors=True)
        assert len(rows) == 1
        payload = rows[0].payload
        assert payload["status"] == status and payload["planner_minutes"] == 35
        assert payload["attachments"][0]["name"] == "proof.pdf"
        assert rows[0].vector == pytest.approx(worker.estimator._vector(result), abs=1e-6)
        if finish_before_prediction:
            assert payload["actual_seconds"] == result["actual_seconds"]


def test_expired_lease_resumes_and_model_failure_finishes(tmp_path, monkeypatch):
    store = PlannerStore(tmp_path / "live")
    task_id = create(store)["id"]
    abandoned = store.claim_prediction(lease_seconds=0.1)
    assert abandoned["task"]["id"] == task_id
    monkeypatch.setattr(worker, "_RECHECK_SECONDS", 0.03)

    def fail(*_):
        raise RuntimeError("simulated background failure")

    monkeypatch.setattr(worker.estimator, "predict", fail)
    schedule_pending(store, store.data_dir / "vectors")
    drain(store)
    result = store.get_task(task_id)["prediction"]
    assert result["point_min"] == 25 and result["basis"] == "User estimate"
    assert "could not be completed" in result["explanation"]


def test_deleted_task_cannot_be_restored_by_late_result(tmp_path, monkeypatch):
    store = PlannerStore(tmp_path / "live")
    task_id = create(store)["id"]
    started = threading.Event()
    release = threading.Event()

    def predict(*_):
        started.set()
        assert release.wait(5)
        return {"point_min": 35, "explanation": "Example"}

    monkeypatch.setattr(worker.estimator, "predict", predict)
    schedule_pending(store, store.data_dir / "vectors")
    try:
        assert started.wait(2)
        store.delete_task(task_id)
    finally:
        release.set()
    drain(store)
    assert store.list_tasks() == []


def test_refresh_saves_grounded_caution_and_preserves_original_forecast(tmp_path, monkeypatch):
    stamp = 1000.0
    store = PlannerStore(tmp_path / "live", clock=lambda: stamp)
    initial = {"point_min": 20, "low_min": 10, "high_min": 30, "evidence": [], "cautions": []}
    current = create(store, prediction=initial)
    earlier = create(store, prediction=initial)
    store.transition(earlier["id"], "start")
    stamp += 30 * 60
    store.transition(earlier["id"], "stop")
    note = "I needed another pass to derive the concentration inequality."
    store.transition(earlier["id"], "finish", completion_note=note)

    monkeypatch.setattr(worker.estimator, "chat_json", lambda *a, **kw: {
        "point_min": 33, "low_min": 25, "high_min": 43, "confidence": "medium",
        "rationale": "The earlier proof took another pass.", "confidence_reason": "One past proof is available.",
        "comparisons": [{"slot": 1, "reason": "Both tasks derive the bound."}],
        "warnings": [{"observation_id": "S1.1", "advice": "Reserve time for another pass over the inequality."}],
    })
    schedule_pending(store, store.data_dir / "vectors")
    drain(store)
    refreshed = store.get_task(current["id"])
    assert refreshed["creation_prediction"] == initial
    assert refreshed["prediction"]["cautions"][0]["sources"] == [
        {"task_id": earlier["id"], "title": earlier["title"], "origin": "Completion note", "observation": note}
    ]
