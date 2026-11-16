import json
from io import BytesIO
from unittest.mock import patch

import pytest
from pypdf import PdfWriter
from qdrant_client import QdrantClient

import estimator
import retrieval


@pytest.fixture(autouse=True)
def local_models(monkeypatch):
    # The forecast tests exercise the real index with deterministic model transports.
    monkeypatch.setattr(retrieval, "embed_task", lambda task, config=None, query=False:
                        [0.6, 0.8] if task.get("pdf_text") else [0.8, 0.6])
    monkeypatch.setattr(retrieval, "decide_related", lambda *args: True)


def paper(task_id, ratio=1, category="Reading", details="Work through the derivations"):
    return {"id": task_id, "title": "Read the attention paper", "description": "Reproduce the proof in section 2",
            "estimate_minutes": 60, "actual_seconds": ratio * 3600, "category": category, "details": details,
            "pdf_text": "Page 2: concentration bound and proof"}


def forecast(**overrides):
    return {"point_min": 87, "low_min": 76, "high_min": 111, "confidence": "medium",
            "rationale": "Working through the proof includes checking two boundary cases.",
            "confidence_reason": "The proof's length is known, but the derivation may require another pass.",
            "comparisons": [], "warnings": [], **overrides}


def test_first_estimate_uses_only_task_scope_and_keeps_model_decisions(tmp_path):
    task = {**paper("new"), "completion_note": "TARGET OUTCOME", "report": {"summary": "TARGET OUTCOME"}}
    output = forecast(confidence="low")
    with patch("estimator.chat_json", return_value=output) as model:
        prediction = estimator.predict(task, [dict(task, id="untimed", actual_seconds=0)], tmp_path)
    model.assert_called_once()
    messages, schema = model.call_args.args
    payload = json.loads(messages[-1]["content"])
    record = payload["new_task"]
    assert payload["history"] == [] and record["user_estimate_minutes"] == 60
    assert payload["timing_context"] == prediction["timing_context"]
    assert "no earlier completed tasks" in prediction["timing_context"].lower()
    assert record["description"] == task["description"]
    assert record["category"] == "Reading" and record["details"] == task["details"]
    assert "concentration bound" in record["document_excerpt"]
    assert "TARGET OUTCOME" not in str(payload) and "actual_seconds" not in record
    assert set(schema["required"]) == set(output) - {"warnings", "comparisons"}
    assert all(prediction[key] == output[key] for key in ("low_min", "point_min", "high_min", "confidence", "confidence_reason"))
    assert prediction["model_rationale"] == output["rationale"]
    assert prediction["evidence"] == [] and prediction["cautions"] == []
    assert prediction["provenance"]["minutes_by"] == "model"
    assert prediction["provenance"]["decision"] == "direct"


def test_model_controls_forecast_and_cites_timed_history(tmp_path):
    # Three comparable 1.5x outcomes produce 90 via the legacy median rule;
    # this response deliberately differs in its point, bounds and confidence.
    past = [paper(str(i), 1.5) for i in range(3)]
    task = paper("new")
    reply = forecast(confidence="high", comparisons=[{
        "slot": 1, "reason": "Both tasks reproduce the proof in section 2.",
    }])
    with patch("estimator.chat_json", return_value=reply) as model:
        result = estimator.predict(task, [*past, paper("new", 99), paper("untimed", 0),
                                          dict(paper("class", 22), fixed_duration=True)], tmp_path)
    model.assert_called_once()
    payload = json.loads(model.call_args.args[0][-1]["content"])
    assert len(payload["history"]) == 3
    assert all((row["user_estimate_minutes"], row["actual_minutes"]) == (60, 90) for row in payload["history"])
    assert all(result[k] == reply[k] for k in ("point_min", "low_min", "high_min", "confidence", "confidence_reason"))
    assert result["basis"] == "Model forecast" and result["model_rationale"] == reply["rationale"]
    cited, = result["evidence"]
    assert cited["task_id"] in {row["id"] for row in past}
    assert cited["actual_min"] == 90 and cited["estimate_min"] == 60
    assert cited["reason"] == reply["comparisons"][0]["reason"]
    assert cited["source_quote"] == past[0]["description"]


def test_first_estimate_model_unavailable(tmp_path):
    with patch("estimator.chat_json", side_effect=estimator.ModelUnavailable("HTTP 401")):
        result = estimator.predict(paper("new"), [], tmp_path)
    assert (result["point_min"], result["low_min"], result["high_min"]) == (60, 30, 90)
    assert "HTTP 401" in result["model_error"] and result["provenance"]["decision"] == "fallback"
    assert result["confidence"] == "low" and result["evidence"] == []


@pytest.mark.parametrize("invalid", [
    {"point_min": 120}, {"low_min": 0}, {"point_min": 87.5}, {"point_min": True},
    {"comparisons": [{"slot": 3, "reason": "Similar proof."}]},
    {"comparisons": [{"slot": 1, "reason": "These tasks are not related."}]},
    {"warnings": [{"observation_id": "S1.9", "advice": "Allow extra setup."}]},
])
def test_invalid_or_invented_model_outputs_cannot_be_published(tmp_path, invalid):
    reply = forecast(**invalid)
    with patch("estimator.chat_json", return_value=reply) as model:
        result = estimator.predict(paper("now"), [paper("prior", 1.4)], tmp_path)
    assert 1 <= model.call_count <= 2
    assert result["provenance"]["decision"] == "fallback" and result["model_error"]
    assert result["cautions"] == [] and result.get("model_rationale", "") != reply["rationale"]


def test_retry_after_truncation_returns_the_second_valid_model_forecast(tmp_path):
    output = forecast()
    with patch("estimator.chat_json", side_effect=[estimator.ModelResponseTruncated("truncated"), output]) as model:
        result = estimator.predict(paper("now"), [paper("prior")], tmp_path)
    assert model.call_count == 2 and result["provenance"]["decision"] == "direct"
    assert result["point_min"] == output["point_min"]


@pytest.mark.parametrize("valid_citation", [True, False])
def test_bonsai_forecast_recovers_once_with_the_same_grounding_and_validation(tmp_path, monkeypatch, valid_citation):
    monkeypatch.setenv("PLANNER_MODEL", "ternary-bonsai-2-4b-mlx")
    monkeypatch.setenv("PLANNER_MODEL_ENDPOINT", "http://127.0.0.1:1235/v1/chat/completions")
    monkeypatch.setenv("PLANNER_SINGLE_MODEL", "0")
    history = dict(paper("prior", 1.4), completion_note="I needed to check the boundary cases twice.")
    reply = forecast(
        comparisons=[{"slot": 1 if valid_citation else 99, "reason": "Both reproduce the same proof."}],
        warnings=[{"observation_id": "S1.1", "advice": "Set aside time to recheck the boundary cases."}],
    )
    attempts = []

    def complete(request, **kwargs):
        body = json.loads(request.data)
        attempts.append(body)
        # The first model answer starts an object but genuinely fails JSON parsing.
        content = '{"point_min":' if len(attempts) == 1 else json.dumps(reply)
        return BytesIO(json.dumps({"choices": [{"finish_reason": "stop", "message": {"content": content}}]}).encode())

    with patch("model_client.build_opener") as client:
        client.return_value.open.side_effect = complete
        result = estimator.predict(paper("now"), [history], tmp_path)
    first, second = attempts
    assert all("response_format" not in request for request in attempts)
    assert first["messages"][-1] == second["messages"][-1]
    assert "observation_id" in second["messages"][0]["content"] and "properties" in second["messages"][0]["content"]
    assert result["provenance"]["minutes_by"] == ("model" if valid_citation else "history")
    if valid_citation:
        assert result["point_min"] == reply["point_min"] and result["model_rationale"] == reply["rationale"]
        assert result["evidence"][0]["reason"] == reply["comparisons"][0]["reason"]
        assert result["cautions"][0]["sources"][0]["task_id"] == history["id"]
    else:
        assert "model_error" in result and "model_rationale" not in result


def test_disk_index_updates_and_forgets_open_and_completed_tasks(tmp_path):
    row = paper("current")
    originally_open = {**row, "status": "open", "actual_seconds": 0}
    estimator.index_task(tmp_path, originally_open)
    estimator.index_task(tmp_path, {**row, "status": "completed"})
    client = QdrantClient(path=str(tmp_path))
    try:
        records, _ = client.scroll(estimator.COLLECTION)
        assert len(records) == 1
        assert records[0].payload["actual_seconds"] == 3600
    finally:
        client.close()
    estimator.forget_task(tmp_path, row["id"])
    client = QdrantClient(path=str(tmp_path))
    try:
        assert client.count(estimator.COLLECTION).count == 0
    finally:
        client.close()
    estimator.clear_index(tmp_path)
    client = QdrantClient(path=str(tmp_path))
    try:
        assert not client.collection_exists(estimator.COLLECTION)
    finally:
        client.close()


def test_offline_model_still_returns_explained_budget(tmp_path):
    with patch("estimator.chat_json", side_effect=estimator.ModelUnavailable("offline")):
        result = estimator.predict(paper("now"), [paper("then", 1.5)], tmp_path)
    assert result["basis"] == "Selected history calculation"
    assert "unavailable" in result["explanation"]
    assert result["confidence"] == "low"
    assert result["evidence"][0]["task_id"] == "then"


def test_pdf_text_extraction_and_invalid_attachment():
    # Minimal PDF with one text drawing operator and a built-in Type 1 font.
    from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject
    writer = PdfWriter()
    page = writer.add_blank_page(200, 200)
    font = DictionaryObject({NameObject("/Type"): NameObject("/Font"), NameObject("/Subtype"): NameObject("/Type1"),
                             NameObject("/BaseFont"): NameObject("/Helvetica")})
    page[NameObject("/Resources")] = DictionaryObject({NameObject("/Font"): DictionaryObject({NameObject("/F1"): font})})
    contents = DecodedStreamObject()
    contents.set_data(b"BT /F1 12 Tf 10 100 Td (Read and derive theorem two) Tj ET")
    page[NameObject("/Contents")] = writer._add_object(contents)
    output = BytesIO()
    writer.write(output)
    assert "derive theorem two" in estimator.extract_pdf_text(output.getvalue())
    assert estimator.extract_pdf_text(b"invalid") == ""


@pytest.mark.parametrize("estimate", [0, -1, float("nan")])
def test_positive_estimates(estimate, tmp_path):
    with pytest.raises(ValueError):
        estimator.predict({**paper("new"), "estimate_minutes": estimate}, [], tmp_path)


def test_full_pdf_reaches_estimation_prompt_and_late_requirements(tmp_path):
    from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

    writer = PdfWriter()
    for i in range(10):
        page = writer.add_blank_page(400, 400)
        font = DictionaryObject({
            NameObject("/Type"): NameObject("/Font"), NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/Helvetica"),
        })
        page[NameObject("/Resources")] = DictionaryObject({
            NameObject("/Font"): DictionaryObject({NameObject("/F1"): font})
        })
        stream = DecodedStreamObject()
        paragraph = "General background for the project. " * 35
        if i == 8:
            paragraph += "Benchmark the lock-free ring buffer with four producer threads."
        stream.set_data(f"BT /F1 8 Tf 5 280 Td ({paragraph}) Tj ET".encode())
        page[NameObject("/Contents")] = writer._add_object(stream)

    output = BytesIO()
    writer.write(output)
    extracted = estimator.extract_pdf_text(output.getvalue())
    assert len(extracted.splitlines()) == 10
    goal = {**paper("now"), "description": "Benchmark the lock-free ring buffer under contention.",
            "pdf_text": extracted}
    captured = {}

    def inspect(messages, schema, **kwargs):
        captured.update(json.loads(messages[1]["content"]))
        return forecast()

    with patch("estimator.chat_json", side_effect=inspect):
        estimator.predict(goal, [dict(goal, id="prior")], tmp_path)
    for record, cap in [(captured["new_task"], 1800), (captured["history"][0], 900)]:
        assert "Page 9:" in record["document_excerpt"]
        assert "four producer threads" in record["document_excerpt"]
        assert len(record["document_excerpt"]) <= cap


def test_retrieval_repairs_an_old_pre_pdf_vector(tmp_path):
    task = {**paper("past"), "status": "completed", "prediction": {"point_min": 90}}
    estimator.index_task(tmp_path, {**task, "pdf_text": "", "status": "ready", "prediction": {}})
    estimator._search(paper("new"), [task], tmp_path)
    with estimator._index(tmp_path) as client:
        records, _ = client.scroll(estimator.COLLECTION, with_vectors=True)
        assert records[0].payload["planner_minutes"] == 90
        assert records[0].payload["status"] == "completed"
        assert records[0].vector == pytest.approx(retrieval.embed_task(task), abs=1e-6)


def test_fixed_duration_skips_model_and_timed_comparisons(tmp_path):
    with patch("estimator.chat_json") as model, patch("estimator._search") as search:
        task = {**paper("show"), "fixed_duration": True, "estimate_minutes": 75}
        result = estimator.predict(task, [paper("overrun", 3)], tmp_path)
    model.assert_not_called()
    search.assert_not_called()
    assert (result["low_min"], result["point_min"], result["high_min"]) == (75, 75, 75)
    assert result["basis"] == "Scheduled duration" and result["cautions"] == []


def test_model_can_cite_obstacle_without_treating_earlier_duration_as_comparable(tmp_path):
    prior = {**paper("field"), "title": "Run USB bridge field calibration", "category": "Robotics",
             "description": "Connect the USB bridge and calibrate for a full shift.", "actual_seconds": 20000,
             "completion_note": "The USB bridge disconnected during capture."}
    task = {**prior, "id": "now", "title": "Run USB bridge bench calibration", "estimate_minutes": 15,
            "completion_note": "FUTURE OUTCOME", "description": "Connect the USB bridge to check the bench link."}
    advice = "Check the USB bridge connection before starting the bench run."

    def answer(messages, schema, **kwargs):
        payload = json.loads(messages[-1]["content"])
        assert 'FUTURE OUTCOME' not in str(payload)
        item, = payload["history"][0]["observations"]
        assert item["observation"] == prior["completion_note"]
        return forecast(low_min=12, point_min=19, high_min=28,
                        warnings=[{"advice": advice, "observation_id": item["id"]}])

    with patch("estimator.chat_json", side_effect=answer) as model:
        result = estimator.predict(task, [prior], tmp_path)
    model.assert_called_once()
    assert result["point_min"] == 19 and result["evidence"] == []
    caution, = result["cautions"]
    assert caution["advice"] == advice and caution["sources"] == [
        {"task_id": "field", "title": prior["title"], "origin": "Completion note", "observation": prior["completion_note"]}]


def test_only_observed_eligible_activity_is_disclosed_or_cited(tmp_path):
    task = dict(paper("now"), title="Study a lecture", description="Watch the lecture on harmonics.")
    visible = {"seconds": 180, "domain": "youtube.com", "title": "Cup highlights",
               "label": "off_task", "idle": False, "excluded": False}
    intervals = [{**visible, "title": "task lecture", "label": "on_task"},
                 {**visible, "title": "private medical tab", "excluded": True},
                 {**visible, "title": "uncertain tab", "label": "uncertain"},
                 {**visible, "title": "idle page", "idle": True}, visible, {**visible, "seconds": 60}]
    history = [{**task, "id": "past", "report": {"intervals": intervals}}]

    def answer(messages, schema, **kwargs):
        payload = json.loads(messages[-1]["content"])
        for private in ("private medical tab", "uncertain tab", "idle page", "task lecture"):
            assert private not in str(payload)
        item, = payload["history"][0]["observations"]
        assert "4.0 min" in item["observation"] and "Cup highlights" in item["observation"]
        return forecast(warnings=[{"observation_id": item["id"], "advice": "Consider turning off sports recommendations."}])

    with patch("estimator.chat_json", side_effect=answer):
        result = estimator.predict(task, history, tmp_path)
    citation, = result["cautions"][0]["sources"]
    assert citation["origin"] == "ActivityWatch" and citation["task_id"] == "past"
    assert result["cautions"][0]["advice"] == "Consider turning off sports recommendations."


def test_offline_forecast_credits_notes_and_keeps_consistent_ratios(tmp_path):
    history = [{**paper(str(i), ratio), "completion_note": "Reconnected the serial link."}
               for i, ratio in enumerate((1.20, 1.25, 1.20))]
    with patch("estimator.chat_json", side_effect=estimator.ModelUnavailable("offline")):
        result = estimator.predict(paper("now"), history, tmp_path)
    assert result["basis"] == "Selected history calculation" and result["point_min"] == 72
    caution, = result["cautions"]
    assert "Reconnected the serial link." in caution["advice"]
    assert [source["observation"] for source in caution["sources"]] == ["Reconnected the serial link."] * 3
    assert {source["origin"] for source in caution["sources"]} == {"Completion note"}

def test_cautions_quote_new_obstacles_verbatim_and_keep_different_observations_apart():
    issue = "The joystick driver rejected CRLF in the calibration file."
    note = {"task_id": "driver", "title": "Configure joystick", "origin": "Completion note", "observation": issue}
    other = {**note, "task_id": "bench", "observation": "The bench was occupied until noon."}
    repeated = {**note, "task_id": "second-driver"}
    cautions = estimator.cautions_from_sources([note, other, repeated, note])
    assert len(cautions) == 2
    assert issue in cautions[0]["advice"] and "usually" not in cautions[0]["advice"]
    assert cautions[0]["sources"] == [note, repeated]
    assert cautions[1]["sources"] == [other] and issue not in cautions[1]["advice"]


def test_rejected_history_never_reaches_model_or_its_warning_sources(tmp_path, monkeypatch):
    past = [{**paper("approved"), "completion_note": "USB bridge stopped responding."},
            {**paper("rejected"), "completion_note": "UNRELATED_PRIVATE_NOTE"}]
    monkeypatch.setattr(retrieval, "decide_related", lambda left, right, examples: right["id"] == "approved")
    with patch("estimator.chat_json", return_value=forecast(warnings=[
            {"observation_id": "S1.1", "advice": "Allow time for reconnecting the bridge."}])) as model:
        prediction = estimator.predict(paper("new"), past, tmp_path)
    payload = json.loads(model.call_args.args[0][-1]["content"])
    assert len(payload["history"]) == 1 and "UNRELATED_PRIVATE_NOTE" not in str(payload)
    assert prediction["cautions"][0]["sources"][0]["task_id"] == "approved"
    assert [r["task_id"] for r in prediction["related_tasks"]] == ["approved"]


def test_pinned_and_excluded_tasks_and_learned_examples(tmp_path, monkeypatch):
    approved = dict(paper("user-picked"), fixed_duration=True)
    prior = [approved, paper("excluded"), paper("automatic")]
    example = {"target_id": "review-t", "candidate_id": "review-c", "target": prior[0],
               "candidate": prior[2], "related": True}
    task = dict(paper("new"), relevance_overrides={"user-picked": True, "excluded": False},
                relevance_examples=[example])
    with patch.object(retrieval, "decide_related", return_value=True) as decide,             patch("estimator.chat_json", return_value=forecast(comparisons=[
                {"slot": 1, "reason": "The same kind of proof."}])) as model:
        prediction = estimator.predict(task, prior, tmp_path)
    decide.assert_called_once_with(task, prior[2], [example])
    assert {r["task_id"] for r in prediction["related_tasks"]} == {"user-picked", "automatic"}
    assert [r["task_id"] for r in prediction["evidence"]] == ["user-picked"]
    assert set(prediction["feedback_source_ids"]) == {"review-t", "review-c"}
    payload = json.loads(model.call_args.args[0][-1]["content"])
    assert payload["reviewed_examples"][0]["related"] is True
    assert "actual_seconds" not in str(payload["reviewed_examples"])
    assert "review-t" not in str(payload)


def test_unavailable_local_retrieval_preserves_cold_model_forecast(tmp_path, monkeypatch):
    monkeypatch.setattr(retrieval, "embed_task", lambda *args: (_ for _ in ()).throw(
        retrieval.ModelUnavailable("Embedding model unavailable.")))
    with patch("estimator.chat_json", return_value=forecast()) as model:
        result = estimator.predict(paper("new"), [paper("same-title")], tmp_path)
    assert "incomplete" in json.loads(model.call_args.args[0][1]["content"])["timing_context"]
    assert model.called and result["related_tasks"] == [] and result["evidence"] == []
    assert result["point_min"] == 87 and result["provenance"]["decision"] == "direct"
    assert result["retryable"] and "Embedding model" in result["retrieval_error"]


def test_overlong_historical_pdf_preserves_other_usable_comparisons(tmp_path, monkeypatch):
    def embed(task, config, query=False):
        if task["id"] == "long":
            raise retrieval.DocumentTooLong("The complete uploaded text exceeds the embedding context.")
        return [1.0, 0.0]
    monkeypatch.setattr(retrieval, "embed_task", embed)
    with patch("estimator.chat_json", return_value=forecast()):
        result = estimator.predict(paper("new"), [paper("long"), paper("fits")], tmp_path)
    assert [item["task_id"] for item in result["related_tasks"]] == ["fits"]
    assert result["provenance"]["decision"] == "direct"
    assert "historical task was skipped" in result["retrieval_error"]


def test_reference_scales_the_new_budget_and_keeps_provider_point(tmp_path):
    prior = {**paper("past"), "estimate_minutes": 20, "actual_seconds": 30 * 60}
    new = {**paper("now"), "estimate_minutes": 40}
    with patch("estimator.chat_json", return_value=forecast()) as call:
        result = estimator.predict(new, [prior], tmp_path)
    record, = json.loads(call.call_args.args[0][1]["content"])["history"]
    assert record["same_scope_minutes"] == 60
    assert result["point_min"] == 87 and result["provenance"]["minutes_by"] == "model"


def test_schema_error_gets_one_repair_but_connection_error_does_not(tmp_path):
    from model_client import ModelResponseInvalid
    for err, count in [(ModelResponseInvalid("invalid point_min field (type)"), 2),
                       (estimator.ModelUnavailable("Connection refused"), 1)]:
        with patch("estimator.chat_json", side_effect=[err, forecast()]) as model:
            result = estimator.predict(paper("new"), [], tmp_path)
        assert model.call_count == count
        assert (result["provenance"]["decision"] == "direct") == (count == 2)
        if count == 2:
            assert "point_min" in model.call_args.args[0][0]["content"]


def test_recurring_secondary_distraction_is_available_to_forecaster():
    def source(domain, seconds):
        return {"domain": domain, "seconds": seconds, "label": "off_task"}

    history = [
        {"report": {"intervals": [source("sports.example", 400), source("video.example", 120)]}},
        {"report": {"intervals": [source("games.example", 480), source("video.example", 90)]}},
    ]
    for row in history:
        obs = estimator._observations(row)
        assert len(obs) == 2 and "video.example" in obs[1]["observation"]


def test_paired_audit_cases_change_only_historical_timing_and_derived_fields(tmp_path):
    from examples.estimation.forecast_audit import challenge, confidence_cases

    arms = {}
    for target, past, _ in challenge() + confidence_cases():
        if target["id"].startswith(("same-indexing-", "supply-count-")):
            with patch("estimator.chat_json", return_value=forecast()) as model:
                estimator.predict(target, past, tmp_path)
            messages, schema = model.call_args.args
            payload = json.loads(messages[1]["content"])
            for row in payload["history"]:
                for field in ("actual_minutes", "same_scope_minutes", "actual_to_budget_ratio"):
                    del row[field]
            arms[target["id"]] = (payload, messages[0], schema, model.call_args.kwargs)
    assert arms["same-indexing-slower"] == arms["same-indexing-faster"] == arms["same-indexing-variable"]
    assert arms["supply-count-tight"] == arms["supply-count-variable"]
    assert arms["supply-count-tight"] != arms["same-indexing-slower"]


def test_point_only_forecast_keeps_model_minutes_without_another_generation(tmp_path):
    reply = {"point_min": 15, "low_min": 15, "high_min": 15, "confidence": "low",
             "rationale": "Collect the card from the department office.",
             "confidence_reason": "There are no comparable pick-ups."}
    task = {"title": "Pick up the lab access card", "description": "Collect the replacement card.",
            "id": "card", "estimate_minutes": 20}
    with patch("estimator.chat_json", return_value=reply) as generation:
        result = estimator.predict(task, [], tmp_path)
    generation.assert_called_once()
    assert (result["low_min"], result["point_min"], result["high_min"]) == (7, 15, 23)
    assert result["range_source"] == "planner_margin" and result["provenance"]["minutes_by"] == "model"
    assert "model_error" not in result
