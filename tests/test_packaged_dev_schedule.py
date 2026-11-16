"""Run the Dev dashboard paths with the packaged Qwen/Kev data and a simulated Cohere HTTP endpoint."""

from collections import Counter
from io import BytesIO
import json
from types import SimpleNamespace

import pytest

import dev_schedule
import dev_workspace
import estimate_worker
import focus_domains
import model_client
import retrieval
import task_workspace
from relevance_feedback import related_ids


def exercise_packaged_history(tmp_path, monkeypatch, history_id, *, local=False):
    calls = Counter()
    missing = []
    endpoint = ("http://127.0.0.1:1235/v1/chat/completions" if local
                else "https://api.cohere.ai/compatibility/v1/chat/completions")
    model = "ternary-bonsai-2-4b-mlx" if local else "command-r7b-12-2024"
    for key, value in {
        "PLANNER_MODEL_ENDPOINT": endpoint, "PLANNER_MODEL": model,
        "COHERE_API": "test-credential", "PLANNER_EMBEDDING_MODEL": retrieval.EMBEDDING_MODEL,
        "PLANNER_DECISION_MODEL": retrieval.DECISION_MODEL, "PLANNER_RELEVANCE_THRESHOLD": "0.50",
    }.items():
        monkeypatch.setenv(key, value)

    def no_local_model(endpoint, payload, config):
        if "state" in payload:
            state = json.loads(payload["state"])
            if "observed_off_task_browser_visit" in state:
                missing.append(("website", state["task_now"][:100], state["observed_off_task_browser_visit"]["domain"]))
            else:
                missing.append(("timing comparison", state["target"]["title"], state["candidate"]["title"]))
        else:
            missing.append(payload["input"][:90])
        raise retrieval.ModelUnavailable("Packaged task made a local model request.")

    def completion_http(request, timeout):
        assert request.full_url == endpoint
        body = json.loads(request.data)
        assert body["model"] == model
        prompt = json.loads(body["messages"][1]["content"])
        if local:
            assert "response_format" not in body
            fields = json.loads(body["messages"][0]["content"].rsplit("matching this schema:\n", 1)[1])["properties"]
        else:
            fields = body["response_format"]["json_schema"]["schema"]["properties"]
        if "intervals" in prompt:
            calls["activity"] += 1
            reply = {str(r["id"]): {"label": "uncertain", "reason": "Simulated classifier."}
                     for r in prompt["intervals"]}
        else:
            calls["forecast"] += 1
            minutes = int(prompt["new_task"]["user_estimate_minutes"])
            reply = {"point_min": minutes, "low_min": max(1, minutes - 5), "high_min": minutes + 5,
                     "confidence": "low", "rationale": "Simulated forecast.", "confidence_reason": "Test response."}
            if "comparisons" in fields:
                calls["screened forecast"] += 1
                prior = prompt["history"][0]
                reply["comparisons"] = [{"slot": prior["slot"], "reason": f"Both involve work on {prior['title']}."}]
            if "warnings" in fields:
                reply["warnings"] = []
        content = json.dumps(reply)
        if local:
            content = [{"type": "reasoning", "text": "untrusted draft"},
                       {"type": "text", "text": "```json\n" + content + "\n```"}]
        envelope = {"choices": [{"finish_reason": "stop", "message": {"content": content}}]}
        return BytesIO(json.dumps(envelope).encode())

    monkeypatch.setattr(retrieval, "_local_post", no_local_model)
    monkeypatch.setattr(model_client, "build_opener", lambda *_: SimpleNamespace(open=completion_http))

    replay = dev_workspace.open_dev_workspace(tmp_path, history_id, replay=True)
    dev_schedule.advance(replay, history_id, through_end=True)
    progress = dev_schedule.progress(replay, history_id)
    assert progress["done"] == progress["total"]
    calls["calendar events"] = progress["done"]
    assert calls["activity"] and calls["forecast"]

    # Each submission receives a distinct runtime ID, including a second copy of a sample.
    editable = dev_workspace.open_dev_workspace(tmp_path, history_id)
    samples = dev_workspace.sample_task_choices(history_id)
    assert samples
    submissions = [*samples, next(iter(samples))]
    added = [dev_workspace.add_sample_task(editable, history_id, sample, "2026-11-15")["id"]
             for sample in submissions]
    assert len(set(added)) == len(submissions)
    calls["sample submissions including duplicate"] = len(added)
    for task in editable.list_tasks():
        if task["status"] != "completed":
            desk = task_workspace.availability(editable, task)
            assert desk and desk["source"] == "Packaged " + retrieval.DECISION_MODEL
            calls["immediate workspace checks"] += 1
            calls["desk eligible"] += bool(desk["eligible"])
    while estimate_worker.run_one_job(editable, editable.data_dir / "vectors"):
        pass
    for task in editable.list_tasks():
        if task["status"] != "completed":
            policy = focus_domains.policy(editable, task["id"])
            assert not policy["pending"], (task["title"], list(policy["pending"]))
            calls["cards with a suggested website"] += bool(policy["suggestions"])
            calls["suggested website entries"] += len(policy["suggestions"])

    added_tasks = (editable.get_task(task_id) for task_id in added)
    target = next(task for task in added_tasks if not task["fixed_duration"])
    reviewed = [row["id"] for row in editable.comparison_options(target["id"])]
    selected = [row["id"] for row in editable.completed_tasks()
                if any(e.get("label") == "off_task" and e.get("domain")
                       for e in (row.get("report") or {}).get("intervals", []))]
    editable.revise_related_tasks(target["id"], selected, previous_ids=related_ids(target["prediction"]),
                                  reviewed_ids=reviewed)
    while estimate_worker.run_one_job(editable, editable.data_dir / "vectors"):
        pass
    assert set(related_ids(editable.get_task(target["id"])["prediction"])) == set(selected)
    assert not focus_domains.policy(editable, target["id"])["pending"]
    calls["manual comparisons selected"] = len(selected)

    assert not missing, f"{history_id}: {len(missing)} local model requests, first misses: {missing[:5]}"
    calls["local inference requests"] = len(missing)
    assert calls["screened forecast"]
    for store in (replay, editable):
        assert not store.has_estimation_work()
        assert store.next_report_job() is None
        with store._db() as db:
            assert not db.execute("SELECT 1 FROM report_jobs").fetchone()
            cells = [row[0] for row in db.execute("SELECT related FROM task_relations")]
            assert cells and set(cells) <= {0, 1}
            calls["editable matrix cells" if store is editable else "replay matrix cells"] = len(cells)
            if store is editable:
                pair = sorted((added[0], added[-1]))
                duplicate = db.execute("SELECT related FROM task_relations WHERE left_id=? AND right_id=?", pair).fetchone()
                assert duplicate is not None and duplicate[0] in (0, 1)
        with retrieval._index(store.data_dir / "vectors") as client:
            assert client.count(retrieval.COLLECTION).count > 0
        for task in store.list_tasks():
            if task["status"] == "completed":
                assert task["report"] is not None
            if task["fixed_duration"]:
                continue
            forecasts = [task["creation_prediction"], task["prediction"]]
            forecasts.extend(entry["prediction"] for entry in store.forecast_timeline(task["id"]))
            for prediction in forecasts:
                assert prediction["provenance"]["provider"] == ("Local endpoint" if local else "Cohere")
                assert prediction["provenance"]["minutes_by"] == "model"
                assert not prediction.get("retrieval_error")
                assert not prediction.get("model_error")
    return calls


@pytest.mark.parametrize("history_id", sorted(set(dev_workspace.history_choices()).union(
    path.stem for path in (dev_workspace.DATASET / "histories").glob("*.json"))))
def test_packaged_replay_and_editable_samples_need_only_the_forecast_provider(tmp_path, monkeypatch, history_id,
                                                                             record_property):
    for label, value in exercise_packaged_history(tmp_path, monkeypatch, history_id).items():
        record_property(label, value)

def test_local_completion_blocks_across_nora_replay_and_edits(tmp_path, monkeypatch):
    calls = exercise_packaged_history(tmp_path, monkeypatch, "d1-systems-student", local=True)
    assert calls["screened forecast"] > 0 and calls["local inference requests"] == 0
