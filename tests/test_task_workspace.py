"""Workspace applicability, task isolation, same timer transitions, and verifiable PDF answers."""

from pathlib import Path

import pytest
from model_client import ModelUnavailable
from planner_store import PlannerStore
from pdf_reader import extract_pdf_text
import task_workspace as desk
import workspace_eligibility as gate


def save_task(store, title, *, files=()):
    return store.create_task(title=title, description="Review the supplied materials.", day="2026-11-14",
                             estimate_minutes=30, fixed_duration=True, attachments=files)


def approve(store, task, monkeypatch):
    key = desk.decision_hash(gate.payload(task, desk.planner_settings()))
    monkeypatch.setattr(gate, "package", lambda: {key: True})
    return desk.availability(store, task)["token"]


def test_new_packaged_workspace_verdict_recovers_previous_outage(tmp_path, monkeypatch):
    store = PlannerStore(tmp_path)
    task = save_task(store, "Read the new sample")
    key = desk.decision_hash(gate.payload(task, desk.planner_settings()))
    desk._record(store, task["id"], key, None, "Local model was unavailable")
    monkeypatch.setattr(gate, "packaged", lambda fingerprint: True if fingerprint == key else None)
    row = desk.availability(store, task)
    assert row["eligible"] and row["source"].startswith("Packaged ")


def test_workspace_access_and_start_share_timer_but_page_visits_never_resume(tmp_path, monkeypatch):
    now = [1000.]
    store = PlannerStore(tmp_path / "a", clock=lambda: now[0])
    other_store = PlannerStore(tmp_path / "b")
    primary = save_task(store, "Review supplied paper")
    secondary = save_task(store, "Edit rebuttal")
    token = approve(store, primary, monkeypatch)
    store.transition(secondary["id"], "start")
    assert desk.open_workspace(store, token)[0]["status"] == "ready"
    assert desk.open_workspace(other_store, token) is None
    now[0] += 120
    desk.start_workspace(store, token)
    desk.start_workspace(store, token)
    assert store.running_task()["id"] == primary["id"]
    assert store.get_task(secondary["id"])["actual_seconds"] == 120
    now[0] += 60
    store.transition(primary["id"], "pause")
    desk.save_draft(store, token, "Draft from this desk", previous="")
    with pytest.raises(ValueError):
        desk.save_draft(store, token, "Stale second browser tab", previous="")
    assert desk.open_workspace(store, token)[0]["status"] == "paused"
    assert not store.running_task()
    assert store.periods(primary["id"]) == [(1120,1180)]
    assert desk.open_workspace(store, token)[1] == "Draft from this desk"
    store.delete_task(primary["id"])
    assert desk.open_workspace(store, token) is None and desk.open_workspace(store, secondary["id"]) is None
    with pytest.raises(ValueError):
        desk.save_draft(store, token, "cannot write after deletion", previous="Draft from this desk")


def test_real_pdf_text_and_validated_citations(monkeypatch):
    document = next((Path(__file__).parents[1]/"data/tasks/pdfs").rglob("*.pdf"))
    text = extract_pdf_text(document.read_bytes())
    passages = desk.passages(text, "What does this assignment require?")
    assert passages
    page, passage = passages[0]
    calls = []
    def reply(messages, schema, **kw):
        calls.append(messages[1]["content"])
        return {"supported": True, "answer": "The assignment asks for a written response.",
                "sources": [{"page": page, "quote": passage[:45]}]}
    monkeypatch.setattr(desk, "chat_json", reply)
    answer = desk.answer({"title": "Study", "completion_note": "private-other-record"}, "What does it require?",
                         document=document.name, text=text, draft="private-different-draft")
    assert answer["sources"][0]["page"] == page and "private" not in calls[0]


def test_retrieves_far_page_and_refuses_invented_citation(monkeypatch):
    text = "\n".join(f"Page {i}: " + ("Neutral copy about the lecture. " * 55) +
                     (" Exponential kick-out bound." if i == 94 else "") for i in range(1, 100))
    cited = desk.passages(text, "Where is the exponential kick-out bound on page 94?")
    assert any(p == 94 and "kick-out" in body for p, body in cited)
    assert sum(len(body) for _, body in cited) <= 15000
    calls = []
    def hallucination(messages, schema, **kwargs):
        calls.append(1)
        return {"supported": True, "answer": "Invented", "sources": [{"page": 999, "quote":"no source"}]}
    monkeypatch.setattr(desk, "chat_json", hallucination)
    with pytest.raises(ModelUnavailable, match="citation"):
        desk.answer({}, "Where is the exponential kick-out bound?", document="Paper", text=text)
    assert len(calls) == 2
    with pytest.raises(ModelUnavailable, match="extractable"):
        desk.answer({}, "What did the scanned page say?", document="Scan", text="")


def test_eligibility_fallback_and_paired_kev_decision(monkeypatch):
    task = {"title":"Jog in the park", "description":"Run 3 km on the trail", "attachments":[]}
    request = gate.payload(task, {})
    monkeypatch.setattr(gate, "package", lambda: {})
    monkeypatch.setattr(gate, "_local_post", lambda *a: {"answers":{"relevant":{"noul":0.9},"external":{"noul":1},"needs_document":{"noul":0}}})
    assert gate.decide(request, {}) == (False, request["model"])
    def offline(*a):
        raise ModelUnavailable("Local Kev is offline")
    monkeypatch.setattr(gate, "_local_post", offline)
    monkeypatch.setenv("PLANNER_MODEL", "ternary-bonsai-4b")
    def fallback(messages, schema, **options):
        assert "essential material is elsewhere" in messages[0]["content"]
        return {"eligible": False}
    monkeypatch.setattr(gate, "chat_json", fallback)
    allowed, source = gate.decide(request, {})
    assert allowed is False and source == "ternary-bonsai-4b"


def test_every_actionable_sample_has_current_workspace_decision_without_model_calls(monkeypatch):
    from examples.estimation.run import histories
    seen = set()
    labels = {}
    def no_network(*args):
        pytest.fail("Packaged sample workspace unexpectedly required a model")
    monkeypatch.setattr(gate, "_local_post", no_network)
    for history in histories(None):
        for row in [*history["open_tasks"], *history["holdouts"]]:
            request = gate.payload({**row, "attachments": bool(row.get("pdf"))}, {})
            seen.add(desk.decision_hash(request))
            labels[row["id"]] = gate.decide(request, {})[0]
    assert seen == set(gate.package())
    assert labels["bridge-proof"] and labels["patch-note"]
    assert not any(labels[k] for k in ("bake-dishes", "duvet", "paused-blue-load", "async-flicker", "held-seed"))


@pytest.mark.parametrize("bad", ["{", "[]", '{"decisions": null}', '{"decisions": {"hash": "false"}}'])
def test_invalid_workspace_cache_cannot_grant_access(tmp_path, monkeypatch, bad):
    cache = tmp_path / "desk.json"
    cache.write_text(bad)
    monkeypatch.setattr(gate, "PACK", cache)
    gate.package.cache_clear()
    try:
        assert gate.package() == {}
    finally:
        gate.package.cache_clear()


def test_workspace_gate_requires_the_actual_source_even_if_a_reader_would_be_suitable(monkeypatch):
    request = gate.payload({"title": "Read the trials paper", "description": "Read the article saved on my desktop.",
                            "attachments": []}, {})
    def decision(*args):
        return {"answers": {"relevant": {"noul": .91}, "external": {"noul": .18},
                            "needs_document": {"noul": .89}}}
    monkeypatch.setattr(gate, "_local_post", decision)
    assert gate.live(request, {}) is False


def test_interrupted_workspace_packaging_keeps_prior_package_and_resumes(tmp_path, monkeypatch):
    import json
    from examples.estimation import package_workspaces as exporter

    released = tmp_path / "published.json"
    checkpoint = tmp_path / "progress.json"
    previous = '{"model": "kev-4b", "decisions": {"old": true}}'
    released.write_text(previous)
    rows = [{"id": k, "title": k} for k in ("Read the next chapter", "Go to the store")]
    monkeypatch.setattr(exporter, "PACK", released)
    monkeypatch.setattr(exporter, "histories", lambda _: [{"id": "sample", "open_tasks": rows, "holdouts": []}])
    calls = []
    def judge(request, config):
        calls.append(json.loads(request["state"])["task"]["title"])
        if len(calls) == 2:
            raise ModelUnavailable("Interrupted")
        return len(calls) == 1
    monkeypatch.setattr(exporter, "live", judge)
    with pytest.raises(ModelUnavailable, match="Interrupted"):
        exporter.build({}, checkpoint=checkpoint)
    assert released.read_text() == previous and checkpoint.exists()
    exporter.build({}, checkpoint=checkpoint)
    exporter.build({}, check=True)
    assert calls == [rows[0]["title"], rows[1]["title"], rows[1]["title"]]
    bundle = json.loads(released.read_text())["decisions"]
    assert len(bundle) == 2 and "old" not in bundle and sorted(bundle.values()) == [False, True]
