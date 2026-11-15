from copy import deepcopy
from datetime import date

import pytest

from examples.estimation import run


@pytest.fixture(autouse=True)
def isolated_local_models(monkeypatch):
    monkeypatch.setattr("retrieval.embed_task", lambda *args: [1.0, 0.0])
    monkeypatch.setattr("retrieval.decide_related", lambda *args: True)


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


def estimate_reply():
    return {"point_min": 23, "low_min": 16, "high_min": 34, "confidence": "medium",
            "rationale": "One new draft needs review.", "confidence_reason": "Its length may vary.",
            "comparisons": [], "warnings": []}


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
                        lambda *args, **kwargs: estimate_reply())
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
        return estimate_reply()

    monkeypatch.setattr(run.estimator, "chat_json", select)
    out = tmp_path / "scores.json"
    result = run.main(["evaluate", "--strict-model", "--json", str(out)])
    assert result == expected
    data = json.loads(out.read_text())
    assert len(data["cases"]) == 2
    assert data["cases"][0]["model_status"] == ("ok" if available else "unavailable")
    assert data["configuration"]["endpoint"] == "https://api.cohere.com/v1/chat"
    assert data["configuration"]["model"] == "test-model"
    if available:
        assert all(case["provenance"]["decision"] == "direct" and case["planner"] == 23
                   and case["confidence_reason"] for case in data["cases"])
    assert data["configuration"]["elapsed_seconds"] >= data["cases"][0]["model_seconds"]


def test_grading_uses_only_prospective_fields_and_scores_citations(tmp_path, monkeypatch):
    from examples.estimation.scoring import validate_rubric

    history = fixture_history()
    row = history["holdouts"][0]
    rubric = {
        "kind": "patterned", "expected_ratio_range": [1.4, 1.5],
        "evidence_ids": ["past-1"], "rationale": "Use historical editing overruns.",
        "expected_caution": {"evidence_ids": ["past-1"], "rationale": "Budget time for a second pass."},
    }
    row.update(fixed_duration=False, evaluation=rubric, completion_note="HELD_OUT_NOTE")
    history["tasks"][0]["completion_note"] = "HISTORICAL_NOTE"
    history["open_tasks"] = [{"title": "OPEN_TASK"}]
    validate_rubric(rubric, {"past-1"})

    def predict(fields, past, _):
        assert set(fields) == set(run.INPUT_KEYS) | {"fixed_duration", "pdf_text"}
        assert fields["fixed_duration"] is False
        assert not ({"actual_minutes", "evaluation", "completion_note", "report", "sessions_minutes", "check"} & fields.keys())
        assert "HISTORICAL_NOTE" in str(past)
        assert all(s not in str((fields, past)) for s in ("HELD_OUT_NOTE", "OPEN_TASK", "Budget time for a second pass."))
        return {"point_min": 28, "low_min": 20, "high_min": 30, "confidence": "low", "basis": "Model-selected tasks",
                "evidence": [{"task_id": "past-1"}],
                "cautions": [{"advice": "Allow a second pass.", "sources": [{"task_id": "past-1"}]}]}

    monkeypatch.setattr(run.estimator, "predict", predict)
    cases = run.evaluate(history, tmp_path, tmp_path / "index", case=row["id"])
    scored = cases[0]["evaluation"]
    assert scored["predicted_ratio"] == 1.4 and scored["prediction_in_band"]
    assert scored["realized_ratio"] == 1.5 and scored["realization_in_band"]
    assert scored["selected_expected_ids"] == ["past-1"] and scored["caution_cites_expected_source"]
    assert scored["unknown_caution_source_ids"] == []
    assert run.metrics(cases)["patterned_cases_in_declared_band"] == {"original": 0, "category": 1, "planner": 1, "actual": 1}
    with pytest.raises(ValueError, match="earlier tasks"):
        validate_rubric({**rubric, "evidence_ids": [row["id"]]}, {"past-1"})


def test_unpredictable_case_is_graded_only_on_realized_outcome():
    from examples.estimation.scoring import grade, rubric_summary
    rubric = {"kind": "control", "expected_ratio_range": None, "evidence_ids": [], "rationale": "Unforeseeable outage."}
    prediction = {"point_min": 30, "evidence": [], "cautions": []}
    scored = grade(rubric, prediction, 30, 51, set())
    summary = rubric_summary([{"evaluation": scored, "original": 30, "category": 30, "planner": 30, "actual": 51}])
    assert scored["prediction_in_band"] is None and scored["caution_cites_expected_source"] is None
    assert summary["patterned_cases"] == 0 and summary["control_cases"] == 1
    assert not run.model_succeeded({"model_status": "ok", "retrieval_error": "Embedding service unavailable"})
    assert run.model_succeeded({"basis": "Scheduled duration", "fixed_duration": True, "model_status": "not_called"})
    assert not run.model_succeeded({"basis": "User estimate", "fixed_duration": False, "model_status": "not_called"})


def test_shared_pdf_stem_is_resolved_locally(tmp_path):
    folder = tmp_path / "pdfs" / "history"
    folder.mkdir(parents=True)
    (folder / "handout.pdf").write_bytes(b"test file")
    assert run.documents({"id": "part-b", "pdf": "handout"}, "history", tmp_path) == [("handout.pdf", b"test file")]
    with pytest.raises(ValueError, match="Invalid fixture PDF"):
        run.documents({"id": "part-b", "pdf": "../handout"}, "history", tmp_path)


def test_recovered_result_preserves_each_real_response(tmp_path, monkeypatch):
    history = fixture_history()
    target = history["holdouts"][0]
    target["evaluation"] = {"kind": "control", "expected_ratio_range": None, "evidence_ids": [], "rationale": "No band."}
    answers = iter([run.ModelUnavailable("Output length exceeded"), {"1": 0}])
    def response(*args, **kwargs):
        value = next(answers)
        if isinstance(value, Exception):
            raise value
        return value
    def exercise(fields, rows, index):
        args = ([{"role": "user", "content": "Synthetic prompt"}], {})
        with pytest.raises(run.ModelUnavailable):
            run.estimator.chat_json(*args)
        run.estimator.chat_json(*args)
        return {"point_min": 20, "low_min": 10, "high_min": 30,
                "confidence": "low", "basis": "User estimate", "evidence": []}
    monkeypatch.setattr(run.estimator, "chat_json", response)
    monkeypatch.setattr(run.estimator, "predict", exercise)
    row = run.evaluate(history, tmp_path, tmp_path / "index", case=target["id"])[0]
    assert row["model_status"] == "ok_after_retry" and row["model_error"] is None and run.model_succeeded(row)
    assert row["model_calls"][0]["error"] == "Output length exceeded"
    assert row["model_calls"][1]["response"] == {"1": 0}


def test_strict_rejects_semantically_invalid_model_forecasts(tmp_path, monkeypatch):
    import json

    history = fixture_history()
    monkeypatch.setattr(run, "histories", lambda selector: [history])
    # A schema-valid response can still place the point above its upper bound.
    monkeypatch.setattr(run.estimator, "chat_json", lambda *args, **kwargs: {**estimate_reply(), "point_min": 400})
    path = tmp_path / "invalid.json"
    assert run.main(["evaluate", "--strict-model", "--json", str(path)]) == 1
    rows = json.loads(path.read_text())["cases"]
    assert len(rows) == 2
    assert all(row["model_status"] == "unavailable" and row["model_error"]
               and row["provenance"]["decision"] == "fallback" for row in rows)
