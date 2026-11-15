import json

import pytest

import focus_judge
from model_client import ModelUnavailable


def request(**options):
    target = {"title": "Analyze latency", "description": "Compare cold cache misses across traces."}
    return focus_judge.payload(options.get("target", target), "chess.com",
                               {"title": "Tune the cache", "id": "private-task-id",
                                "report": {"intervals": ["unrelated-visit-data"]}},
                               options.get("titles", ["Blitz arena", "Daily chess"]),
                               options.get("config", {}))


def test_stable_request_uses_scope_browser_evidence_and_model_without_opaque_task_data():
    first = request()
    assert first == request(titles=[" Daily  chess ", "Blitz arena", "Blitz arena"])
    state = json.loads(first["state"])
    assert state["observed_off_task_browser_visit"] == {"label": "off_task", "domain": "chess.com",
                                   "tab_titles": ["Blitz arena", "Daily chess"]}
    assert "private-task-id" not in first["state"] and "unrelated-visit-data" not in first["state"]
    assert list(first["questions"]) == ["specific", "needed"]
    for different in (request(target={"title": "Study competitive chess"}), request(titles=["Tournament relay"]),
                      request(config={"PLANNER_DECISION_MODEL": "other-model"})):
        assert focus_judge.key(different) != focus_judge.key(first)


@pytest.mark.parametrize("verdict", [True, False])
def test_exact_public_hit_never_contacts_network(tmp_path, monkeypatch, verdict):
    req = request()
    path = tmp_path / "focus-decisions.json"
    path.write_text(json.dumps({"format": 1, "decisions": {focus_judge.key(req): verdict}}))
    monkeypatch.setattr(focus_judge, "CACHE_PATH", path)
    monkeypatch.setattr(focus_judge.retrieval, "_local_post", lambda *a: pytest.fail("Network on a public hit"))
    assert focus_judge.packaged(req) is verdict and focus_judge.decide(req, {}) is verdict
    assert focus_judge.packaged(request(titles=["Different evidence"])) is None


def test_missing_public_decision_uses_only_the_local_decision_route(tmp_path, monkeypatch):
    monkeypatch.setattr(focus_judge, "CACHE_PATH", tmp_path / "absent.json")
    calls = []
    def judge(endpoint, data, config):
        calls.append((endpoint, data))
        return {"answers": {"specific": {"noul": 0.85}, "needed": {"noul": 0.90}}}
    monkeypatch.setattr(focus_judge.retrieval, "_local_post", judge)
    req = request()
    assert focus_judge.packaged(req) is None and calls == []
    assert focus_judge.decide(req, {"PLANNER_DECISION_ENDPOINT": "http://localhost:7890/v1/systemone"}) is False
    assert calls == [("http://localhost:7890/v1/systemone", req)]


@pytest.mark.parametrize("raw", [{}, {"answers": {"specific": {"noul": 2}, "needed": {"noul": 0.2}}},
                                  {"answers": {"specific": {"noul": True}, "needed": {"noul": 0.2}}},
                                  {"answers": {"specific": {"noul": 0.5}, "needed": {"noul": float("nan")}}}])
def test_invalid_local_decision_is_an_unavailable_model(monkeypatch, raw):
    monkeypatch.setattr(focus_judge.retrieval, "_local_post", lambda *a: raw)
    with pytest.raises(ModelUnavailable, match="invalid website blocking decision"):
        focus_judge.live_decision(request(), {})


@pytest.mark.parametrize(("specific", "needed", "block"), [(0.90, 0.10, True), (0.90, 0.80, False),
                                                         (0.20, 0.10, False)])
def test_only_identified_activity_on_an_unneeded_site_is_selected(monkeypatch, specific, needed, block):
    reply = {"answers": {"specific": {"noul": specific}, "needed": {"noul": needed}}}
    monkeypatch.setattr(focus_judge.retrieval, "_local_post", lambda *a: reply)
    assert focus_judge.live_decision(request(), {}) is block


def test_attached_resource_reference_is_forwarded_without_unrelated_pdf_text():
    common = {"title": "Summarize the next chapter"}
    unrelated = request(target={**common, "pdf_text": "Fill the property table and record each failure."})
    assert unrelated == request(target=common)
    relevant = request(target={**common, "pdf_text": "Open your saved game on the Chess server and annotate turn five."})
    assert "annotate turn five" in json.loads(relevant["state"])["site_reference_in_attachment"]
    assert focus_judge.key(relevant) != focus_judge.key(unrelated)


@pytest.mark.parametrize("generic,needed,blocks", [(.85,.1,False), (.1,.8,False), (.1,.2,True)])
def test_repeated_visits_still_check_identifiability_and_current_use(monkeypatch, generic, needed, blocks):
    req = focus_judge.payload({"title": "Rebuild the workbench"}, "store.example",
                              [{"title": "Replace a joint"}, {"title": "Sand the top"}],
                              ["Maple rolltop desk auction ending in 10 minutes"], {})
    assert set(req["questions"]) == {"generic", "needed"}
    assert json.loads(req["state"])["related_tasks_with_off_task_visits"] == 2
    reply = {"answers": {"generic": {"noul": generic}, "needed": {"noul": needed}}}
    monkeypatch.setattr(focus_judge.retrieval, "_local_post", lambda *_: reply)
    assert focus_judge.live_decision(req, {}) is blocks
