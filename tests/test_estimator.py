import json
from io import BytesIO
from unittest.mock import patch

import pytest
from pypdf import PdfWriter
from qdrant_client import QdrantClient

import estimator


def paper(task_id, ratio=1, category="Reading", details="Work through the derivations"):
    return {"id": task_id, "title": "Read the attention paper", "description": "Reproduce the proof in section 2",
            "estimate_minutes": 60, "actual_seconds": ratio * 3600, "category": category, "details": details,
            "pdf_text": "Page 2: concentration bound and proof"}


def test_first_estimate_uses_task_pdf_independently(tmp_path):
    output = {"low_min": 70, "point_min": 90, "high_min": 120, "reason": "Reproducing the proof requires deriving the bound."}
    task = paper("new")
    with patch("estimator.chat_json", return_value=output) as model:
        prediction = estimator.predict(task, [dict(task, id="untimed", actual_seconds=0)], tmp_path)
    messages, _ = model.call_args.args
    record = json.loads(messages[1]["content"])
    assert set(record) == {"title", "description", "category", "details", "document_excerpt"}
    assert record["description"] == task["description"]
    assert record["category"] == "Reading" and record["details"] == task["details"]
    assert "concentration bound" in record["document_excerpt"]
    assert "actual_seconds" not in record
    assert (prediction["point_min"], prediction["low_min"], prediction["high_min"]) == (90, 70, 120)
    assert prediction["confidence"] == "low" and prediction["evidence"] == []
    assert prediction["basis"] == "Model estimate from task scope"
    assert output["reason"] in prediction["explanation"]


def test_first_estimate_preserves_explanation_and_final_assumption(tmp_path):
    reason = (
        "The task calls for reconstructing the central argument, verifying the intermediate lemmas, "
        "and writing out the assumptions for the final bound. Each step takes time because you plan "
        "to reproduce the derivation in a notebook. The range allows for revisiting sections that "
        "use unfamiliar notation, as well as checking that the boundary conditions hold. "
        "This assumes you already know the concentration inequality used in the proof."
    )
    reply = {"low_min": 60.0, "point_min": 90.0, "high_min": 120.0, "reason": reason}
    with patch("estimator.chat_json", return_value=reply):
        result = estimator.predict(paper("new"), [], tmp_path)
    assert result["explanation"].endswith(reason)
    assert result["point_min"] == 90 and type(result["point_min"]) is int


def test_first_estimate_model_unavailable(tmp_path):
    with patch("estimator.chat_json", side_effect=estimator.ModelUnavailable("HTTP 401")):
        result = estimator.predict(paper("new"), [], tmp_path)
    assert (result["point_min"], result["low_min"], result["high_min"]) == (60, 30, 90)
    assert "HTTP 401" in result["explanation"]
    assert result["confidence"] == "low" and result["evidence"] == []


@pytest.mark.parametrize("low,point,high", [(80, 60, 100), (40, 110, 90), (0, 60, 100),
                                            (60, 60, 60), (False, 10, 25), (20, 35.5, 60)])
def test_first_estimate_rejects_invalid_range(low, point, high, tmp_path):
    with patch("estimator.chat_json", return_value=dict(low_min=low, point_min=point, high_min=high, reason="Scope")):
        result = estimator.predict(paper("new"), [], tmp_path)
    assert result["basis"] == "User estimate" and result["point_min"] == 60
    assert "invalid estimate" in result["explanation"]


def test_uses_only_selected_outcomes_once_and_snapshots_math(tmp_path):
    history = [paper(str(i), r) for i, r in enumerate((1.25, 1.5, 2))]
    request = {}

    def select(messages, schema, **kwargs):
        request.update(json.loads(messages[1]["content"]))
        return {"matches": [{"slot": i, "reason": "Same deliverable and depth of reading."}
                            for i in [1, 2, 2, 3]]}

    with patch("estimator.chat_json", side_effect=select):
        result = estimator.predict(paper("new"), history + [paper("new", 99), paper("empty", 0)], tmp_path)
    assert set(row["task_id"] for row in result["evidence"]) == {"0", "1", "2"}
    assert result["point_min"] == 90
    assert (result["low_min"], result["high_min"]) == (75, 120)
    assert len(request["candidates"]) == 3
    assert all(set(record).isdisjoint({"actual_seconds", "estimate_minutes", "prediction"}) for record in request["candidates"])
    assert request["new_task"]["details"] and request["new_task"]["document_excerpt"]


def test_model_declines_then_category_fallback_and_distinct_cold_start(tmp_path):
    with patch("estimator.chat_json", return_value={"matches": []}):
        result = estimator.predict(paper("new"), [paper("past", 2)], tmp_path)
        missing = estimator.predict(paper("other", category="Coding"), [paper("past")], tmp_path)
    assert result["confidence"] == "low"
    assert result["basis"] == "Category history"
    assert 60 < result["point_min"] < 120
    assert missing["basis"] == "User estimate"


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
    assert result["basis"] == "Text similarity"
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
        return {"matches": []}

    with patch("estimator.chat_json", side_effect=inspect):
        estimator.predict(goal, [dict(goal, id="prior")], tmp_path)
    for record, cap in [(captured["new_task"], 1800), (captured["candidates"][0], 900)]:
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
        assert records[0].vector == pytest.approx(estimator._vector(task), abs=1e-6)


def test_fixed_duration_skips_model_and_timed_comparisons(tmp_path):
    with patch("estimator.chat_json") as model, patch("estimator._search") as search:
        task = {**paper("show"), "fixed_duration": True, "estimate_minutes": 75}
        result = estimator.predict(task, [paper("overrun", 3)], tmp_path)
    model.assert_not_called()
    search.assert_not_called()
    assert (result["low_min"], result["point_min"], result["high_min"]) == (75, 75, 75)
    assert result["basis"] == "Scheduled duration" and result["cautions"] == []


def test_meeting_uses_observed_ratios_with_no_surcharge_for_lateness(tmp_path):
    task = dict(paper("target"), title="Schedule rota with Sam", category="Meetings",
                description="Review all of this month's handovers with Sam.", estimate_minutes=30)
    note = "Sam joined six minutes after the start. We still covered the whole rota."
    history = [{**task, "id": str(i), "actual_seconds": 30 * 60 * ratio,
                "completion_note": note if not i else ""}
               for i, ratio in enumerate((1.20, 1.25, 1.20))]
    fixed = {**task, "id": "training", "fixed_duration": True, "actual_seconds": 9000}
    target = {**task, "completion_note": "TARGET OUTCOME MUST STAY PRIVATE"}

    def choose(messages, schema, **kwargs):
        record = json.loads(messages[-1]["content"])
        assert "TARGET OUTCOME" not in messages[-1]["content"]
        assert len(record["candidates"]) == 3
        return {"matches": [{"slot": row["slot"], "reason": "Same person and review.", "flag": 1}
                            for row in record["candidates"]]}

    with patch("estimator.chat_json", side_effect=choose) as model:
        result = estimator.predict(target, [*history, fixed], tmp_path)
    model.assert_called_once()
    assert result["point_min"] == 36 and result["low_min"] < 36 < result["high_min"]
    assert result["confidence"] == "medium"
    caution, = result["cautions"]
    assert note in caution["advice"]
    assert caution["sources"] == [{"task_id": "0", "title": task["title"],
                                  "origin": "Completion note", "observation": note}]


def test_relevant_aw_flags_exclude_private_idle_and_unclassified_intervals(tmp_path):
    task = dict(paper("now"), title="Study a lecture", description="Watch the lecture on harmonics.")
    visible = {"seconds": 180, "domain": "youtube.com", "title": "Cup highlights",
               "label": "off_task", "idle": False, "excluded": False}
    intervals = [
        {**visible, "title": "task lecture", "label": "on_task"},
        {**visible, "title": "private medical tab", "excluded": True},
        {**visible, "title": "attendee late", "label": "uncertain"},
        {**visible, "title": "idle page", "idle": True},
        visible, {**visible, "seconds": 60},
    ]
    history = [{**task, "id": key, "report": {"intervals": intervals}} for key in ("a", "b", "unrelated")]

    def choose(messages, schema, **kwargs):
        records = json.loads(messages[-1]["content"])["candidates"]
        assert all(r["observations"] == ["ActivityWatch: 4.0 min on youtube.com labeled off-task (Cup highlights)."]
                   for r in records)
        return {"matches": [{"slot": 1, "reason": "Same lecture format.", "flag": 1},
                            {"slot": 1, "reason": "Duplicate.", "flag": 1},
                            {"slot": 2, "reason": "Same lecture format.", "flag": 1}]}

    with patch("estimator.chat_json", side_effect=choose):
        result = estimator.predict(task, history, tmp_path)
    assert len(result["cautions"]) == 1
    advice = result["cautions"][0]
    assert "youtube.com" in advice["advice"]
    assert {source["task_id"] for source in advice["sources"]} == {e["task_id"] for e in result["evidence"]}
    assert len(advice["sources"]) == 2
    assert {source["origin"] for source in advice["sources"]} == {"ActivityWatch"}


def test_invalid_and_category_only_caution_references_are_ignored(tmp_path):
    prior = {**paper("prior"), "completion_note": "The serial link disconnected."}
    for matches in ([], [{"slot": 1, "reason": "Similar scope.", "flag": 2}]):
        with patch("estimator.chat_json", return_value={"matches": matches}):
            result = estimator.predict(paper("now"), [prior], tmp_path)
        assert result["cautions"] == []


def test_offline_forecast_credits_notes_and_keeps_consistent_ratios(tmp_path):
    history = [{**paper(str(i), ratio), "completion_note": "Reconnected the serial link."}
               for i, ratio in enumerate((1.20, 1.25, 1.20))]
    with patch("estimator.chat_json", side_effect=estimator.ModelUnavailable("offline")):
        result = estimator.predict(paper("now"), history, tmp_path)
    assert result["basis"] == "Text similarity" and result["point_min"] == 72
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
