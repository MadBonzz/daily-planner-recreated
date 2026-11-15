"""Exercise the distributed retrieval artifacts with the local model endpoints offline."""

import json

import pytest

import retrieval
import sample_retrieval as cache
from examples.estimation.package_retrieval import inputs, needed_pairs
from examples.estimation.run import histories


@pytest.fixture(autouse=True)
def configured_sample_models(monkeypatch):
    monkeypatch.setattr(retrieval, "planner_settings", lambda: {})


def test_only_query_embeddings_receive_the_instruction(monkeypatch):
    sent = []

    def record(url, payload, config):
        sent.append(payload["input"])
        return {"data": [{"embedding": [1.0]}]}

    monkeypatch.setattr(retrieval, "_local_post", record)
    task = {"title": "New scoped job", "description": "Compare two drill heads", "pdf_text": "Sheet 4."}
    retrieval.embed_task(task)
    retrieval.embed_query(task)
    plain = "Title: New scoped job\nDescription: Compare two drill heads\nDocuments: Sheet 4."
    assert sent == [plain, f"Instruct: {retrieval.QUERY_INSTRUCTION}\nQuery: {plain}"]


def test_all_sample_replays_work_from_packaged_model_inputs(monkeypatch):
    def disallow_network(*_):
        pytest.fail("A shipped sample vector or pair judgment required a local model.")

    monkeypatch.setattr(retrieval, "_local_post", disallow_network)
    for history in histories(None):
        tasks = inputs(history)
        vectors = {key: {"document": retrieval.embed_task(task), "query": retrieval.embed_query(task)}
                   for key, task in tasks.items()}
        assert all(len(roles["document"]) == len(roles["query"]) == 1024 and roles["document"] != roles["query"]
                   for roles in vectors.values())
        expected = needed_pairs(history, vectors)
        bundle = json.loads((cache.DIRECTORY / (history["id"] + ".json")).read_text(encoding="utf-8"))
        recorded = {(row["target"], row["candidate"]): row["related"] for row in bundle["decisions"].values()}
        assert expected <= recorded.keys()
        for first, second in expected:
            assert retrieval.decide_related(tasks[first], tasks[second]) is recorded[first, second]


def test_content_changes_and_reviewed_examples_require_fresh_local_inference(monkeypatch):
    tasks = inputs(histories(None)[0])
    a, b = tasks["bridge-proof"], tasks["barrier-proof"]

    def offline(*_):
        raise retrieval.ModelUnavailable("local models are offline")

    monkeypatch.setattr(retrieval, "_local_post", offline)
    assert retrieval.decide_related(a, b)
    assert retrieval.embed_task({**a, "completion_note": "The outcome does not change task scope."})
    assert retrieval.embed_query(a)
    with pytest.raises(retrieval.ModelUnavailable):
        retrieval.embed_query({**a, "pdf_text": "Entirely different handout."})
    with pytest.raises(retrieval.ModelUnavailable):
        retrieval.embed_query(a, {"PLANNER_EMBEDDING_MODEL": "different-model"})
    with pytest.raises(retrieval.ModelUnavailable):
        retrieval.decide_related(a, b, [{"target": a, "candidate": b, "related": False}])


def test_unavailable_custom_history_does_not_hide_packaged_matches(tmp_path, monkeypatch):
    tasks = inputs(histories(None)[0])
    target = tasks["bridge-proof"]
    historical = {**tasks["barrier-proof"], "id": "fresh-random-workspace-id"}
    custom = {**historical, "id": "another-random-id", "title": "Different work from a real user"}

    def offline(*_):
        raise retrieval.ModelUnavailable("unavailable custom embedding")

    monkeypatch.setattr(retrieval, "_local_post", offline)
    errors = set()
    found = retrieval.search(target, [custom, historical], tmp_path, errors=errors)
    assert [r["id"] for r, _ in found] == [historical["id"]]
    assert any("unavailable custom embedding" in error for error in errors)


def test_readable_float32_vectors_reject_invalid_input_and_malformed_files(tmp_path):
    import math
    import struct

    path = cache.csv_path(tmp_path / "sample.json", "task-one")
    raw = [1.0 / math.sqrt(cache.DIMENSIONS)] * cache.DIMENSIONS
    raw[0] += 1e-8
    cache.write_vectors(path, {"document": raw, "query": raw})
    lines = path.read_text(encoding="ascii").splitlines()
    assert lines[0] == "document,query" and len(lines) == 1025
    restored = cache.read_vectors(tmp_path / "sample.json", "task-one")
    assert struct.pack("<1024f", *raw) == struct.pack("<1024f", *restored["document"])
    for first, rest in [("wrong,query", lines[1:]), (lines[0], lines[2:]),
                        (lines[0], ["nan,0.1", *lines[2:]]), (lines[0], ["0.0,0.0"] * 1024),
                        (lines[0], ["=1+1,0.1", *lines[2:]])]:
        path.write_text("\n".join([first, *rest]) + "\n", encoding="ascii")
        cache._columns.cache_clear()
        with pytest.raises(ValueError):
            cache.read_vectors(tmp_path / "sample.json", "task-one")
    with pytest.raises(ValueError):
        cache.write_vectors(path, {"document": raw[:-1], "query": raw})


def test_exported_metadata_matches_public_fixture_files_and_inputs():
    import hashlib
    from examples.estimation.run import DATASET

    for history in histories(None):
        index = cache.DIRECTORY / (history["id"] + ".json")
        bundle = json.loads(index.read_text(encoding="utf-8"))
        assert bundle["format"] == cache.FORMAT
        assert bundle["public_sources"]["repository"] == "MadBonzz/daily-planner-recreated"
        for relative, sha in bundle["public_sources"]["files_sha256"].items():
            assert hashlib.sha256((DATASET / relative).read_bytes()).hexdigest() == sha
        tasks = inputs(history)
        assert set(tasks) == set(bundle["vectors"])
        for task_id, task in tasks.items():
            for role in cache.ROLES:
                text = retrieval.embedding_input(task, role == "query")
                assert bundle["vectors"][task_id][role] == {"input_sha256": cache.text_hash(text)}
            assert cache.csv_path(index, task_id).is_file()
