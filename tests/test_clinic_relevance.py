"""Relevance is decided from each task pair, including pairs with identical titles."""

from unittest.mock import patch

import pytest

import retrieval


def test_single_joint_embedding_includes_full_late_document_and_omits_outcome_and_category():
    doc = "Page one " * 1800 + "Page 9: nonlinear stability criterion"
    task = dict(title="Proof", description="Stability", details="Two lemmas", pdf_text=doc,
                category="COURSE_CATEGORY", completion_note="PRIVATE_OUTCOME")
    with patch.object(retrieval, "_local_post", return_value={"data": [{"embedding": [1.0, 0.5]}]}) as send:
        assert retrieval.embed_task(task) == [1.0, 0.5]
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
    assert calls == ["new"]
    strong["pdf_text"] = "New page"
    retrieval.search(target, [strong, weak], tmp_path)
    assert calls == ["new", "new", "cross-category"]


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
    monkeypatch.setattr("estimate_worker.extract_pdf_text", lambda data: "Page 1: torque calibration on the test rig")
    with patch.object(retrieval, "decide_related", return_value=True) as judge:
        _process_relation(store, pair)
    assert all("torque calibration" in row["pdf_text"] for row in judge.call_args.args)
    assert all(store.get_task(row["id"])["pdf_text_version"] == 1 for row in judge.call_args.args)


def test_native_lmstudio_prevents_silent_truncation_before_embedding(monkeypatch):
    from io import BytesIO
    from types import SimpleNamespace
    from urllib.parse import urlsplit

    class NativeClient:
        def open(self, endpoint, timeout):
            assert endpoint.endswith("/api/v1/models")
            return BytesIO(b'{"models":[{"loaded_instances":[{"id":"qwen3-embedding-0.6b","config":{"context_length":16}}]}]}')

    captured = {}

    def count(cmd, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace(stdout=b"[20,16]")

    monkeypatch.setattr(retrieval.subprocess, "run", count)
    input_text = "underwater " * 20
    with pytest.raises(retrieval.DocumentTooLong, match="20 text tokens"):
        retrieval._check_embedding_context(urlsplit("http://localhost:1234/v1/embeddings"),
                                           {"model": retrieval.EMBEDDING_MODEL, "input": input_text}, NativeClient(), 5)
    assert captured["input"] == input_text.encode()
    assert not any(k in captured["env"] for k in ("HTTPS_PROXY", "ALL_PROXY", "http_proxy"))
