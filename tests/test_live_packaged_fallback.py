"""Live-first Qwen and Kev can reuse only the exact shipped computation on failure."""

import pytest

import focus_judge
import retrieval
import sample_retrieval
import workspace_eligibility
from examples.estimation.package_retrieval import inputs
from examples.estimation.run import histories
from fallback_provenance import capture_fallbacks
from model_client import ModelUnavailable


LIVE = {"PLANNER_EMBEDDING_MODEL": "qwen3-embedding-0.6b-live", "PLANNER_DECISION_MODEL": "kev-4b-live"}


@pytest.fixture
def proof_tasks():
    return inputs(histories(None)[0])


def test_live_embedding_roles_try_fresh_then_fall_back_only_to_the_matching_input(monkeypatch, proof_tasks):
    task = proof_tasks["bridge-proof"]
    calls = []
    unavailable = ModelUnavailable("Qwen unavailable")
    def embed(_, request, config):
        calls.append(request.copy())
        if len(calls) == 1:
            return {"data": [{"embedding": [1.] + [0.] * 1023}]}
        raise unavailable
    monkeypatch.setattr(retrieval, "_local_post", embed)
    with capture_fallbacks() as used:
        assert retrieval.embed_task(task, LIVE)[0] == 1
        assert used == set()
        for query in (False, True):
            wanted = retrieval.embedding_input(task, query)
            assert retrieval.embed_task(task, LIVE, query) == sample_retrieval.vector_for(retrieval.EMBEDDING_MODEL, wanted)
        assert used == {"embedding"}
    with capture_fallbacks() as misses:
        for config in (LIVE, {**LIVE, "PLANNER_EMBEDDING_MODEL_KEY": "unrelated/embedding-model"},
                       {"PLANNER_EMBEDDING_MODEL": "other"}):
            data = {**task, "pdf_text": "Different assignment"} if config == LIVE else task
            with pytest.raises(ModelUnavailable) as error:
                retrieval.embed_task(data, config)
            assert error.value is unavailable
        assert not misses
    assert len(calls) == 6 and calls[0]["model"] == LIVE["PLANNER_EMBEDDING_MODEL"]


@pytest.mark.parametrize("outcome", [True, False])
def test_live_kev_pair_keeps_hash_direction_and_negative_verdict(monkeypatch, proof_tasks, outcome):
    left, right = proof_tasks["bridge-proof"], proof_tasks["barrier-proof"]
    calls = []
    unavailable = ModelUnavailable("Kev unavailable")
    def fail(request, config):
        calls.append(request.copy())
        raise unavailable
    monkeypatch.setattr(retrieval, "planner_settings", lambda: LIVE)
    monkeypatch.setattr(retrieval, "_live_decision", fail)
    # A legacy public sample may contain only one direction of a symmetric pair.
    digest = sample_retrieval.decision_hash(retrieval.decision_payload(right, left, [], retrieval.DECISION_MODEL))
    monkeypatch.setattr(sample_retrieval, "decision_for",
                        lambda request: {digest: outcome}.get(sample_retrieval.decision_hash(request)))
    first, second = retrieval.active_decision_order(left, right)
    assert first is right and second is left
    with capture_fallbacks() as used:
        assert retrieval.decide_related(first, second) is outcome
        assert used == {"decision"}
    with pytest.raises(ModelUnavailable) as error:
        retrieval.decide_related(first, second, [{"target": first, "candidate": second, "related": not outcome}])
    assert error.value is unavailable
    assert len(calls) == 2 and all(request["model"] == LIVE["PLANNER_DECISION_MODEL"] for request in calls)


def test_nonmatching_keys_and_managed_explicit_key_do_not_confuse_models():
    assert sample_retrieval.packaged_model("decision", "kev-4b-live", {"PLANNER_DECISION_MODEL_KEY": "custom/kev-4b"}) is None
    assert sample_retrieval.packaged_model("embedding", "other-embedding", {}) is None
    settings = {"PLANNER_SINGLE_MODEL": "1", "PLANNER_EMBEDDING_MODEL_KEY": "Qwen/Qwen3-Embedding-0.6B-GGUF"}
    assert sample_retrieval.packaged_model("embedding", "my-local-model", settings) == "qwen3-embedding-0.6b"
    assert not sample_retrieval.prefer_packaged("embedding", "qwen3-embedding-0.6b", settings)


@pytest.mark.parametrize("role,model,saved", [("embedding", sample_retrieval.QWEN, [0.5]),
                                             ("decision", sample_retrieval.KEV, False)])
def test_managed_precomputed_mode_contacts_model_only_on_a_miss(role, model, saved):
    config = {"PLANNER_SINGLE_MODEL": "1", "PLANNER_PREFER_PRECOMPUTED": "1"}
    def unexpected():
        pytest.fail("A packaged result should avoid a live request")
    assert sample_retrieval.resolve(role, model, config, lambda _: saved, unexpected)[0] == saved
    assert sample_retrieval.resolve(role, model, config, lambda _: None, lambda: "live") == ("live", model)


def test_site_and_workspace_use_exact_fallback_before_another_forecast(monkeypatch):
    request = {"model": "kev-4b-live", "state": "task scope", "questions": {"relevant": {"type": "noul"}}}
    archived = {**request, "model": "kev-4b"}
    digest = sample_retrieval.decision_hash(archived)
    calls = []
    def fail(payload, config):
        calls.append(payload)
        raise ModelUnavailable("Kev offline")
    monkeypatch.setattr(focus_judge, "live_decision", fail)
    monkeypatch.setattr(focus_judge, "packaged", lambda payload: False if payload == archived else None)
    monkeypatch.setattr(workspace_eligibility, "live", fail)
    monkeypatch.setattr(workspace_eligibility, "packaged", lambda key: True if key == digest else None)
    monkeypatch.setattr(workspace_eligibility, "chat_json", lambda *_: {"eligible": False})
    with capture_fallbacks() as used:
        assert focus_judge.decide(request, LIVE) is False
        assert workspace_eligibility.decide(request, LIVE) == (True, "Packaged kev-4b decision (local model unavailable)")
        assert used == {"decision"}
    modified = {**request, "state": "different file, website or reviewed example"}
    with capture_fallbacks() as misses:
        with pytest.raises(ModelUnavailable, match="Kev offline"):
            focus_judge.decide(modified, LIVE)
        assert workspace_eligibility.decide(modified, LIVE)[0] is False
        assert not misses
    assert calls == [request, request, modified, modified]
    assert request["model"] == "kev-4b-live"


def test_live_bare_embedding_reused_only_for_identical_task_and_configuration_within_job(monkeypatch):
    calls = []
    def infer(content, model, config):
        calls.append((content, model))
        return [float(len(calls))]
    monkeypatch.setattr(retrieval, "_live_vector", infer)
    config = {"PLANNER_EMBEDDING_MODEL": "test-encoder"}
    task = {"id": "target", "title": "A proof"}
    with retrieval.reuse_task_embedding("target"):
        first = retrieval.embed_task(task, config)
        assert retrieval.embed_task(task, config) == first
        assert retrieval.embed_query(task, config) != first
        assert retrieval.embed_task({**task, "title": "A new proof"}, config) != first
        assert retrieval.embed_task(task, {**config, "PLANNER_EMBEDDING_MODEL": "another-encoder"}) != first
    assert retrieval.embed_task(task, config) != first
    assert len(calls) == 5
