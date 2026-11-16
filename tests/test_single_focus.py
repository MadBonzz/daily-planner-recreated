"""Exercise a single website observation from a forecast through the running browser policy."""

import json

import pytest

import activity
import estimate_worker
import estimator
import focus_bridge
import focus_domains
import focus_judge
import retrieval
from planner_store import PlannerStore


def create(store, title, **fields):
    return store.create_task(title=title, description="Reproduce a stalled print job, then fix the spooler.",
                             estimate_minutes=30, day="2026-11-14", category="Software", **fields)["id"]


def visit(domain="puzzling.example", title="Daily cryptic crossword", **fields):
    return {"app": "Chrome", "title": title, "domain": domain, "label": "off_task",
            "seconds": 240, "idle": False, "excluded": False, **fields}


def finished(store, events):
    source = create(store, "Repair the print spooler", prediction={"point_min": 35})
    store.transition(source, "finish", spent_seconds=30 * 60, report={"intervals": events})
    return source


@pytest.fixture
def model_calls(monkeypatch):
    """Only model responses are simulated. Retrieval, forecast jobs, reports and focus reads use SQLite/Qdrant."""
    calls = []
    monkeypatch.setattr(retrieval, "embed_task", lambda *_args, **_kwargs: [0.6, 0.8])
    monkeypatch.setattr(retrieval, "decide_related", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(estimator, "_model_forecast", lambda *_args: {
        "point_min": 32, "low_min": 25, "high_min": 40, "evidence": [], "cautions": [],
        "confidence": "medium", "explanation": "Comparable spooler changes took half an hour."})

    def local_judge(_endpoint, payload, _config):
        assert set(payload["questions"]) == {"specific", "needed"}
        calls.append(payload)
        domain = json.loads(payload["state"])["observed_off_task_browser_visit"]["domain"]
        probability = 0.15 if domain == "labwiki.example" else 0.95
        return {"answers": {"specific": {"noul": probability}, "needed": {"noul": 0.05}}}

    monkeypatch.setattr(retrieval, "_local_post", local_judge)
    return calls


def estimate(store):
    assert estimate_worker.run_one_job(store, store.data_dir / "vectors")


def test_single_visit_persists_a_yes_and_a_no_and_bridge_honors_user_edits(tmp_path, model_calls):
    store = PlannerStore(tmp_path / "live")
    finished(store, [visit(), visit("labwiki.example", "Department wiki home")])
    target = create(store, "Diagnose a second print queue")

    # A normal new task has no estimate or block list until the worker commits it.
    assert store.get_task(target)["prediction"]["status"] == "pending"
    assert focus_domains.policy(store, target)["domains"] == []
    estimate(store)
    by_host = {json.loads(req["state"])["observed_off_task_browser_visit"]["domain"]: req for req in model_calls}
    assert set(by_host) == {"puzzling.example", "labwiki.example"} and len(model_calls) == 2
    assert all(focus_judge.packaged(req) is None for req in model_calls)
    prediction = store.get_task(target)["prediction"]
    assert prediction["point_min"] == 32 and prediction["provenance"]["minutes_by"] == "model"
    assert prediction["focus_verdicts"] == {focus_judge.key(by_host["puzzling.example"]): True,
                                             focus_judge.key(by_host["labwiki.example"]): False}

    # Reopen the workspace and read it through the same read-only transaction as the extension.
    reopened = PlannerStore(store.data_dir)
    config = focus_bridge.bind(tmp_path, reopened.path)
    assert focus_bridge.snapshot(tmp_path, config)["domains"] == []
    reopened.transition(target, "start")
    choice = focus_domains.policy(reopened, target)
    assert choice["domains"] == ["puzzling.example"] and not choice["pending"]
    assert focus_bridge.snapshot(tmp_path, config)["domains"] == ["puzzling.example"]
    focus_domains.save_domains(reopened, target, [], choice["domains"])
    dismissed = focus_domains.policy(reopened, target)
    assert dismissed["removed"] == ["puzzling.example"]
    assert focus_bridge.snapshot(tmp_path, config)["domains"] == []
    focus_domains.save_domains(reopened, target, ["puzzling.example"], dismissed["domains"])
    assert focus_bridge.snapshot(tmp_path, config)["domains"] == ["puzzling.example"]
    reopened.transition(target, "pause")
    assert focus_bridge.snapshot(tmp_path, config)["domains"] == []
    assert len(model_calls) == 2


def test_filtered_or_mixed_evidence_never_goes_to_single_site_judge(tmp_path, model_calls):
    store = PlannerStore(tmp_path)
    store.set_setting("exclusions", ["confidential"])
    source = finished(store, [
        visit(), visit("video.example", "Bloopers"), visit("work.video.example", "Recorded seminar", label="on_task"),
        visit("private.example", "Confidential client workspace"), visit("hidden.example", excluded=True),
        visit("simultaneous.example", phone_app="Sports"), visit("idle.example", idle=True),
        visit("lab-manual.example"), visit("allowed.example"), visit("uncertain.example", label="uncertain"),
    ])
    focus_domains.save_domains(store, source, [], ["allowed.example"])
    target = create(store, "Fix a queue under load", details="Follow https://lab-manual.example/queues first.")
    estimate(store)
    assert len(model_calls) == 1
    observed = json.loads(model_calls[0]["state"])["observed_off_task_browser_visit"]
    assert observed["domain"] == "puzzling.example" and observed["tab_titles"] == ["Daily cryptic crossword"]
    choice = focus_domains.policy(store, target)
    assert choice["domains"] == ["puzzling.example"] and not choice["pending"]


@pytest.mark.parametrize("final_change", ["correct report", "forget task"])
def test_old_yes_cannot_follow_a_changed_tab_or_corrected_or_forgotten_source(tmp_path, model_calls, final_change):
    store = PlannerStore(tmp_path)
    source = finished(store, [visit()])
    target = create(store, "Repair another stalled print submission")
    estimate(store)
    assert focus_domains.policy(store, target)["domains"] == ["puzzling.example"]
    revised_visit = visit(title="Tutorial on creating printable crossword layouts")
    store.save_report(source, {"intervals": [revised_visit]})
    stale = focus_domains.policy(store, target)
    assert stale["domains"] == [] and "puzzling.example" in stale["pending"]
    assert len(model_calls) == 1
    estimate(store)
    assert focus_domains.policy(store, target)["domains"] == ["puzzling.example"]
    assert len(model_calls) == 2 and focus_judge.key(model_calls[0]) != focus_judge.key(model_calls[1])
    if final_change == "correct report":
        store.request_report(source, {activity.activity_key(revised_visit): "on_task"})
    else:
        store.archive_task(source)
        assert focus_domains.policy(store, target)["domains"] == ["puzzling.example"]
        store.delete_task(source)
    current = focus_domains.policy(store, target)
    assert current["domains"] == [] and not current["pending"]
    assert len(model_calls) == 2


def test_unavailable_single_site_judge_leaves_forecast_intact_and_can_be_retried(tmp_path, model_calls, monkeypatch):
    store = PlannerStore(tmp_path)
    finished(store, [visit()])
    target = create(store, "Check the queued cartridge jobs")
    with monkeypatch.context() as offline:
        def unavailable(*_):
            raise retrieval.ModelUnavailable("Simulated local decision server downtime.")

        offline.setattr(retrieval, "_local_post", unavailable)
        estimate(store)
    assert store.get_task(target)["prediction"]["point_min"] == 32
    result = focus_domains.policy(store, target)
    assert result["domains"] == [] and "puzzling.example" in result["pending"]
    assert store.retry_estimate(target)
    estimate(store)
    assert focus_domains.policy(store, target)["domains"] == ["puzzling.example"]
    assert len(model_calls) == 1
    # An ordinary estimate refresh can reuse both the site's verdict and its evidence key.
    assert store.retry_estimate(target)
    estimate(store)
    assert focus_domains.policy(store, target)["domains"] == ["puzzling.example"]
    assert len(model_calls) == 1


def test_saved_dev_forecast_picks_up_released_singleton_without_local_models(tmp_path, monkeypatch):
    from examples.estimation.run import DATASET, histories, historical_rows, task_inputs

    # These are the actual accepted comparisons and ActivityWatch reports in the shipped fixture.
    history = histories("03")[0]
    fixture = next(row for row in history["holdouts"] if row["id"] == "appendix-proof")
    saved_run = DATASET.parent / "results/holdouts/2026-11-15-qwen-query-instruction/instructed/appendix-proof.json"
    previous = json.loads(saved_run.read_text(encoding="utf-8"))
    monkeypatch.setenv("PLANNER_DECISION_MODEL", retrieval.DECISION_MODEL)
    monkeypatch.setattr(retrieval, "_local_post", lambda *_: pytest.fail("Saved forecast requested local inference"))
    store = PlannerStore(tmp_path / "live")
    sources = {}
    for row in historical_rows(history, DATASET):
        fields = {k: row[k] for k in ("title", "description", "details", "category", "estimate_minutes", "day", "pdf_text")}
        key = store.create_task(**fields, prediction={})["id"]
        store.transition(key, "finish", spent_seconds=row["actual_seconds"], report=row["report"])
        sources[row["id"]] = key
    related = [sources[comp["task_id"]] for comp in previous["related_tasks"]]
    task, files = task_inputs(fixture, history["id"], DATASET)
    target = store.create_task(**task, attachments=files, prediction={"point_min": previous["planner"],
                               "related_tasks": [{"task_id": key} for key in related]})["id"]
    reopened = PlannerStore(store.data_dir)
    assert "focus_verdicts" not in reopened.get_task(target)["prediction"]
    req = focus_domains.site_requests(reopened, target, related)["nba.com"]
    assert focus_judge.packaged(req) is True
    choice = focus_domains.policy(reopened, target)
    assert "nba.com" in choice["domains"] and not choice["pending"]
    assert "one related task" in choice["suggestions"]["nba.com"]
    config = focus_bridge.bind(tmp_path, reopened.path)
    reopened.transition(target, "start")
    assert focus_bridge.snapshot(tmp_path, config)["domains"] == choice["domains"]


def test_pending_single_site_is_selectable_in_task_editor_without_inference(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest

    store = PlannerStore(tmp_path)
    source = finished(store, [visit()])
    target = create(store, "Fix the overnight print queue", prediction={"related_tasks": [{"task_id": source}]})
    monkeypatch.setattr(retrieval, "_local_post", lambda *_: pytest.fail("Rendering made a local inference request"))
    # The old forecast has no verdict; the task editor must let the user decide directly.
    app = AppTest.from_string(
        "from planner_store import PlannerStore\nfrom focus_ui import site_editor\n"
        f"store = PlannerStore({str(tmp_path)!r})\nsite_editor(store, store.get_task({target!r}))"
    ).run()
    assert not app.exception
    sites = next(field for field in app.multiselect if field.label == "Blocked websites")
    assert sites.value == [] and "puzzling.example" in sites.options
    sites.set_value(["puzzling.example"])
    next(button for button in app.button if button.label == "Save websites").click().run()
    assert not app.exception
    choice = focus_domains.policy(store, target)
    assert choice["added"] == ["puzzling.example"] and not choice["pending"]


def test_timer_started_before_estimate_applies_committed_review_and_keeps_user_edits(tmp_path, model_calls):
    store = PlannerStore(tmp_path / "live")
    finished(store, [visit()])
    target = create(store, "Fix the late print queue")
    config = focus_bridge.bind(tmp_path, store.path)
    store.transition(target, "start")
    assert not focus_bridge.snapshot(tmp_path, config)["domains"]
    focus_domains.save_domains(store, target, ["scores.example"], [])
    assert focus_bridge.snapshot(tmp_path, config)["domains"] == ["scores.example"]
    estimate(store)
    assert focus_bridge.snapshot(tmp_path, config)["domains"] == ["puzzling.example", "scores.example"]
    assert store.get_task(target)["status"] == "running"
    store.transition(target, "pause")
    assert not focus_bridge.snapshot(tmp_path, config)["domains"]
    store.transition(target, "resume")
    assert focus_bridge.snapshot(tmp_path, config)["domains"] == ["puzzling.example", "scores.example"]
    store.archive_task(target)
    assert not focus_bridge.snapshot(tmp_path, config)["domains"]
    assert len(model_calls) == 1

@pytest.mark.parametrize("superseded", [False, True])
def test_forecast_is_visible_during_site_review_and_late_verdict_cannot_overwrite_it(tmp_path, model_calls,
                                                                                      monkeypatch, superseded):
    store = PlannerStore(tmp_path)
    finished(store, [visit()])
    target = create(store, "Diagnose a stuck print queue")
    decide = focus_judge.decide
    observed = []

    def during_review(request, config):
        saved = store.get_task(target)
        observed.append((saved["prediction"].get("point_min"), saved["estimate_updating"]))
        if superseded:
            store.retry_estimate(target)
        return decide(request, config)

    monkeypatch.setattr(focus_judge, "decide", during_review)
    estimate(store)
    assert observed == [(32, False)]
    saved = store.get_task(target)
    assert ("focus_verdicts" in saved["prediction"]) is not superseded
    assert saved["estimate_updating"] is superseded
