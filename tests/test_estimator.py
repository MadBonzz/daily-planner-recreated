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


def test_cold_start_without_model(tmp_path):
    task = paper("new")
    with patch("estimator.chat_json") as model:
        prediction = estimator.predict(task, [], tmp_path)
    model.assert_not_called()
    assert (prediction["point_min"], prediction["low_min"], prediction["high_min"]) == (60, 30, 90)
    assert prediction["evidence"] == []


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
