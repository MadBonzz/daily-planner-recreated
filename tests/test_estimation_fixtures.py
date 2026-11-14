from copy import deepcopy
from datetime import date

import pytest

from examples.estimation import run


def fixture_history():
    task = {
        "id": "edit", "day": "2026-03-01", "title": "Review a draft",
        "description": "Read and comment on the new draft.", "category": "Editing",
        "details": "", "estimate_minutes": 20, "actual_minutes": 30, "sessions_minutes": [10, 20],
    }
    tasks = [{**task, "id": f"past-{i}", "day": f"2026-03-0{i}"} for i in range(1, 6)]
    holdouts = [{**task, "id": f"future-{i}", "day": f"2026-03-1{i}",
                 "check": "Ground truth is not available to the model."} for i in (1, 2)]
    return {"id": "mock", "name": "Test editor", "scenario": "Synthetic writing tasks.",
            "tasks": tasks, "holdouts": holdouts}


def test_holdouts_use_only_pretask_inputs_and_fixed_history(tmp_path, monkeypatch):
    history = fixture_history()
    run.validate(history, tmp_path)
    received = []
    prediction = run.estimator.predict

    def inspect(fields, completed, index_dir):
        received.append((set(fields), [row["id"] for row in completed]))
        return prediction(fields, completed, index_dir)

    def no_remote_calls(*args, **kwargs):
        pytest.fail("Offline scoring unexpectedly sent a model request")

    monkeypatch.setattr(run.estimator, "predict", inspect)
    monkeypatch.setattr(run.estimator, "chat_json", no_remote_calls)
    cases = run.evaluate(history, tmp_path, tmp_path / "index", offline=True)
    expected_ids = [t["id"] for t in history["tasks"]]
    assert received == [(set(run.INPUT_KEYS) | {"pdf_text"}, expected_ids)] * 2
    assert all(c["category"] == 30 and c["model_status"] == "offline" for c in cases)
    assert run.metrics(cases)["mae_minutes"]["category"] == 0


def test_isolated_profile_has_sessions_without_holdout_outcomes(tmp_path, monkeypatch):
    history = fixture_history()
    monkeypatch.setattr(run.estimator, "chat_json",
                        lambda *args, **kwargs: {"matches": []})
    profile = tmp_path / "separate-profile"
    run.populate_profile(history, tmp_path, profile)
    store = run.PlannerStore(profile / "live")
    completed = store.completed_tasks()
    assert len(completed) == 5 and completed[0]["prediction"] == {}
    assert completed[0]["actual_seconds"] == 1800
    assert [end - start for start, end in store.periods(completed[0]["id"])] == [600, 1200]
    assert store.capture_periods(completed[0]["id"]) == []
    forecasts = [row for row in store.list_tasks(date.today()) if row["status"] == "ready"]
    assert len(forecasts) == 2 and all(t["actual_seconds"] == 0 for t in forecasts)
    with pytest.raises(ValueError, match="empty"):
        run.populate_profile(history, tmp_path, profile)


@pytest.mark.parametrize("field,bad", [("sessions_minutes", [4, 5]), ("day", "2026-02-01"),
                                       ("id", "../unsafe")])
def test_validation_rejects_bad_cases(tmp_path, field, bad):
    history = deepcopy(fixture_history())
    history["holdouts"][0][field] = bad
    with pytest.raises(ValueError):
        run.validate(history, tmp_path)


@pytest.mark.parametrize("available,expected", [(True, 0), (False, 1)])
def test_strict_evaluation_saves_results_even_on_failure(tmp_path, monkeypatch, available, expected):
    import json

    history = fixture_history()
    monkeypatch.setattr(run, "histories", lambda selector: [history])
    monkeypatch.setattr(run, "model_settings", lambda: ("https://api.cohere.com/v1/chat?trial=private", "test-model"))

    def select(*args, **kwargs):
        if not available:
            raise run.ModelUnavailable("Model unavailable")
        # A functioning model may legitimately find no close match.
        return {"matches": []}

    monkeypatch.setattr(run.estimator, "chat_json", select)
    out = tmp_path / "scores.json"
    result = run.main(["evaluate", "--strict-model", "--json", str(out)])
    assert result == expected
    data = json.loads(out.read_text())
    assert len(data["cases"]) == 2
    assert data["cases"][0]["model_status"] == ("ok" if available else "unavailable")
    assert data["configuration"]["endpoint"] == "https://api.cohere.com/v1/chat"
    assert data["configuration"]["model"] == "test-model"
    assert data["configuration"]["elapsed_seconds"] >= data["cases"][0]["model_seconds"]
