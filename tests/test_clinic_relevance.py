"""Relevance is decided from each task pair, including pairs with identical titles."""

import json
from unittest.mock import patch

import pytest

import retrieval


def test_single_joint_embedding_includes_full_late_document_and_omits_outcome_and_category():
    doc = "Page one " * 1800 + "Page 9: nonlinear stability criterion"
    task = dict(title="Proof", description="Stability", details="Two lemmas", pdf_text=doc,
                category="COURSE_CATEGORY", completion_note="PRIVATE_OUTCOME")
    vector = [1.] + [0.] * (retrieval.sample_retrieval.DIMENSIONS - 1)
    with patch.object(retrieval, "_local_post", return_value={"data": [{"embedding": vector}]}) as send:
        assert retrieval.embed_task(task) == vector
    content = send.call_args.args[1]["input"]
    assert content == f"Title: Proof\nDescription: Stability\nDetails: Two lemmas\nDocuments: {doc}"


@pytest.mark.parametrize("probability, expected", [(0.01, False), (0.99, True)])
def test_pairwise_decision_uses_systemone_and_personal_examples(probability, expected):
    task = {"title": "Review", "description": "Reconcile payroll receipts"}
    prior = {"title": "Review", "description": "Check network router settings"}
    examples = [{"target": task, "candidate": prior, "related": False}]
    with patch.object(retrieval, "_local_post", return_value={"answers": {"relevant": {"noul": probability}}}) as send:
        assert retrieval.decide_related(task, prior, examples) is expected
    endpoint, payload, _ = send.call_args.args
    assert endpoint.endswith("/v1/systemone") and "reviewed_examples" in payload["state"]
    assert "Reconcile payroll" in payload["state"] and "network router" in payload["state"]
    assert list(payload["questions"]) == ["relevant"]


@pytest.mark.parametrize("reply", [{}, {"answers": {}}, {"answers": {"relevant": {"noul": None}}},
                                    {"answers": {"relevant": {"noul": float("nan")}}}])
def test_unknown_decisions_stay_unknown(reply):
    with patch.object(retrieval, "_local_post", return_value=reply), pytest.raises(retrieval.ModelUnavailable):
        retrieval.decide_related({"title": "Study"}, {"title": "Study"})


@pytest.mark.parametrize("configured,expected", [(None, 0.50), ("0.45", 0.50), ("0.50", 0.50),
                                                ("0.70", 0.70), ("0.85", 0.85), ("1", 1.0)])
def test_retrieval_floor_including_older_settings(configured, expected):
    assert retrieval.relevance_threshold({"PLANNER_RELEVANCE_THRESHOLD": configured}) == expected


def test_even_an_old_cutoff_cannot_admit_a_candidate_below_the_half_cosine_floor(tmp_path, monkeypatch):
    import math

    target = {"id": "target", "title": "Fill a form", "estimate_minutes": 12}
    rows = [{**target, "id": "below", "cosine": 0.49}, {**target, "id": "above", "cosine": 0.51}]
    def embed(task, config, query=False):
        cosine = task.get("cosine", 1)
        return [cosine, math.sqrt(1 - cosine ** 2)]
    monkeypatch.setattr(retrieval, "embed_task", embed)
    monkeypatch.setenv("PLANNER_RELEVANCE_THRESHOLD", "0.45")
    assert [row["id"] for row, _ in retrieval.search(target, rows, tmp_path)] == ["above"]


def test_threshold_is_applied_before_judging_and_cache_repairs_content(tmp_path, monkeypatch):
    def task(i):
        return {"id": i, "title": "Identical title", "description": i, "estimate_minutes": 10, "category": "Office"}
    target, strong, weak, excluded = map(task, ["new", "cross-category", "weak", "deleted"])
    strong["category"] = "School"
    vecs = {"new": [1., 0.], "cross-category": [1., .1], "weak": [0., 1.], "deleted": [1., 0.]}
    calls = []

    def embed(row, config, query=False):
        calls.append(row["id"])
        return vecs[row["id"]]
    monkeypatch.setattr(retrieval, "embed_task", embed)
    monkeypatch.setenv("PLANNER_RELEVANCE_THRESHOLD", "0.85")
    retrieval.index_task(tmp_path, excluded)
    assert [r["id"] for r, _ in retrieval.search(target, [strong, weak], tmp_path)] == ["cross-category"]
    calls.clear()
    retrieval.search(target, [strong, weak], tmp_path)
    assert calls == ["new", "new"]
    strong["pdf_text"] = "New page"
    retrieval.search(target, [strong, weak], tmp_path)
    assert calls == ["new", "new", "new", "new", "cross-category"]


def test_forgotten_task_cannot_be_reinserted_by_revoked_worker(tmp_path, monkeypatch):
    task = {"id": "forgotten", "title": "Private study", "description": "Review a handout", "estimate_minutes": 10}
    monkeypatch.setattr(retrieval, "embed_task", lambda *args: [1.0, 0.0])
    retrieval.index_task(tmp_path, task)
    retrieval.forget_task(tmp_path, task["id"])
    with retrieval.estimation_lease(lambda: False), pytest.raises(retrieval.ModelUnavailable):
        retrieval.search(dict(task, id="new"), [task], tmp_path)
    with retrieval._index(tmp_path) as client:
        assert client.count(retrieval.COLLECTION).count == 0


def test_pair_upgrades_sampled_documents_from_legacy_or_fixed_tasks(tmp_path, monkeypatch):
    from planner_store import PlannerStore
    from estimate_worker import _process_relation

    store = PlannerStore(tmp_path)
    for name in ("Morning", "Evening"):
        store.create_task(title=name, description="Lab booking", category="Lab", day="2026-01-05",
                          estimate_minutes=15, fixed_duration=True, pdf_text="Page 1: stale preview", attachments=[("scope.pdf", b"%PDF-stub")])
    pair = store.claim_relation()
    monkeypatch.setattr("task_files.extract_pdf_text", lambda data: "Page 1: torque calibration on the test rig")
    with patch.object(retrieval, "decide_related", return_value=True) as judge:
        _process_relation(store, pair)
    assert all("torque calibration" in row["pdf_text"] for row in judge.call_args.args)
    assert all(store.get_task(row["id"])["pdf_text_version"] == 1 for row in judge.call_args.args)


def test_unmanaged_embedding_sends_the_full_input_directly_to_the_endpoint(monkeypatch):
    from io import BytesIO

    text = "underwater " * 5000
    calls = []

    class NativeClient:
        def open(self, request, timeout):
            calls.append((request.full_url, request.get_method(), json.loads(request.data)))
            return BytesIO(b'{"data":[{"embedding":[0.5,0.5]}],"usage":{"prompt_tokens":16}}')

    monkeypatch.setattr(retrieval, "build_opener", lambda *args: NativeClient())
    endpoint = "http://localhost:1235/v1/embeddings"
    payload = {"model": "custom-embedding-model", "input": text}
    assert retrieval._local_post(endpoint, payload, {})["data"][0]["embedding"] == [0.5, 0.5]
    assert calls == [(endpoint, "POST", payload)]


def test_either_qwen_query_can_qualify_a_pair_and_each_candidate_appears_once(tmp_path, monkeypatch):
    task = {"id": "target", "title": "Write the lab brief", "estimate_minutes": 30}
    scores = {"instructed-only": [1, 0, 0], "plain-only": [0, 1, 0],
              "both": [.6, .8, 0], "neither": [0, 0, 1]}
    rows = [{**task, "id": key} for key in scores]

    def embed(row, config, query=False):
        if row["id"] == "target":
            return [1, 0, 0] if query else [0, 1, 0]
        return scores[row["id"]]

    monkeypatch.setattr(retrieval, "embed_task", embed)
    selected = retrieval.search(task, rows, tmp_path)
    assert {row["id"]: score for row, score in selected} == pytest.approx(
        {"instructed-only": 1, "plain-only": 1, "both": .8})
    assert len(selected) == 3


def test_one_unavailable_query_role_preserves_matches_and_discloses_incomplete_search(tmp_path, monkeypatch):
    task = {"id": "new", "title": "Finish the lab", "estimate_minutes": 20}
    prior = {**task, "id": "old"}

    def embed(row, config, query=False):
        if query:
            raise retrieval.DocumentTooLong("The query instruction exceeds context capacity.")
        return [0.6, 0.8]

    monkeypatch.setattr(retrieval, "embed_task", embed)
    errors = set()
    hits = retrieval.search(task, [prior], tmp_path, errors=errors)
    assert [row["id"] for row, _ in hits] == ["old"]
    assert any("instructed task embedding" in message for message in errors)


@pytest.mark.parametrize("threshold,expected", [("0.50", {"on-floor", "identical"}),
                                              ("1.00", {"identical"})])
def test_exact_configured_cosine_floor_is_inclusive(tmp_path, monkeypatch, threshold, expected):
    import math

    target = {"id": "query", "title": "Review", "estimate_minutes": 20}
    vectors = {"query": [1., 0.], "identical": [1., 0.], "on-floor": [.5, math.sqrt(.75)],
               "below": [0., 1.]}
    candidates = [{**target, "id": key} for key in ("identical", "on-floor", "below")]
    monkeypatch.setenv("PLANNER_RELEVANCE_THRESHOLD", threshold)
    monkeypatch.setattr(retrieval, "embed_task", lambda task, *_: vectors[task["id"]])
    assert {row["id"] for row, _ in retrieval.search(target, candidates, tmp_path)} == expected


def test_default_qwen_rejects_wrong_dimensions_and_rebuilds_a_legacy_incompatible_index(tmp_path, monkeypatch):
    target = {"id": "target", "title": "New quantitative task", "estimate_minutes": 20}
    malformed = {**target, "id": "bad", "title": "Malformed output task"}
    valid = {**target, "id": "valid", "title": "Past quantitative task"}
    vector = [1.] + [0.] * (retrieval.sample_retrieval.DIMENSIONS - 1)

    def serve(url, payload, config):
        return {"data": [{"embedding": [1., 0.] if "Malformed" in payload["input"] else vector}]}

    monkeypatch.setattr(retrieval, "_local_post", serve)
    # Older code could create this index from an otherwise well-formed response.
    with retrieval._index(tmp_path) as client:
        client.create_collection(retrieval.COLLECTION,
                                 vectors_config=retrieval.models.VectorParams(size=2, distance=retrieval.models.Distance.COSINE))
    errors = set()
    hits = retrieval.search(target, [malformed, valid], tmp_path, errors=errors)
    assert [row["id"] for row, _ in hits] == ["valid"]
    assert "requires 1024" in " ".join(errors)
    with retrieval._index(tmp_path) as client:
        assert client.count(retrieval.COLLECTION).count == 1
        assert client.get_collection(retrieval.COLLECTION).config.params.vectors.size == 1024
    assert retrieval.embed_task(malformed, {"PLANNER_EMBEDDING_MODEL": "custom"}) == [1., 0.]
