import json
from unittest.mock import Mock

import pytest

import workspace_forecast as forecast
from model_client import ModelUnavailable


TASK = {"title": "Prove the rate", "description": "Derive and check the bound.", "estimate_minutes": 30}
BASE = {"point_min": 35, "low_min": 30, "high_min": 50, "confidence": "medium",
        "explanation": "An earlier proof needed a separate boundary check.",
        "evidence": [{"title": "Convergence proof", "actual_min": 36}],
        "cautions": [{"advice": "Check the boundary case."}],
        "provenance": {"model": "earlier-model", "decision": "direct", "minutes_by": "model"}}


def snapshot(*messages, new_message_start=0):
    return {"paused_at": 1000, "elapsed_seconds": 325, "new_message_start": new_message_start,
            "messages": [{"role": role, "content": content, "document": "rate.pdf"} for role, content in messages]}


def supported(quote, turn=1):
    return {"assessment": "supported", "evidence": [{"quote": quote, "turn": turn, "kind": "completed"}],
            "remaining_low_min": 5, "remaining_min": 8, "remaining_high_min": 14, "confidence": "medium",
            "confidence_reason": "The boundary case could need a separate argument.",
            "explanation": "The recurrence is solved. Allow eight minutes to check the boundary case."}


def test_user_work_sample_updates_total_preserves_historical_evidence(monkeypatch):
    quote = "r(1) = 0.25\nr(2) = 0.125"
    progress = snapshot(("user", f"My derivation so far:\n{quote}\nDoes the boundary case follow?"),
                        ("assistant", "Check the boundary separately."))
    model = Mock(return_value=supported(quote))
    monkeypatch.setattr(forecast, "chat_json", model)
    monkeypatch.setenv("PLANNER_MODEL", "pause-model")
    monkeypatch.setenv("PLANNER_MODEL_ENDPOINT", "https://api.cohere.ai/compatibility/v1/chat/completions")
    prediction = forecast.revise(TASK, BASE, progress)
    messages, _ = model.call_args.args
    assert [m["role"] for m in messages] == ["system", "user"]
    assert "All JSON fields" in messages[0]["content"]
    payload = json.loads(messages[1]["content"])
    assert payload["messages"][0]["content"] == progress["messages"][0]["content"]
    assert payload["baseline_forecast"]["point_min"] == 35
    assert (prediction["low_min"], prediction["point_min"], prediction["high_min"]) == (11, 14, 20)
    update = prediction["workspace_update"]
    assert update["evidence"] == [{"quote": quote, "document": "rate.pdf", "kind": "completed"}]
    assert (update["status"], update["remaining_min"], update["previous_total_min"]) == ("adjusted", 8, 35)
    assert update["elapsed_minutes"] == 325 / 60
    assert prediction["provenance"] == {"provider": "Cohere", "model": "pause-model", "decision": "direct",
                                        "minutes_by": "model"}
    assert prediction["cautions"] == BASE["cautions"] and prediction["evidence"] == BASE["evidence"]


@pytest.mark.parametrize("quote,turn", [("finish the proof", 1), ("Try solving it recursively.", 2),
                                        ("I proved it.", 1), ("unknown", 4)])
def test_questions_assistant_suggestions_and_invented_quotes_cannot_change_minutes(monkeypatch, quote, turn):
    progress = snapshot(("user", "How do I finish the proof?"), ("assistant", "Try solving it recursively."))
    model = Mock(side_effect=[supported(quote, turn), {"assessment": "insufficient_evidence"}])
    monkeypatch.setattr(forecast, "chat_json", model)
    prediction = forecast.revise(TASK, BASE, progress)
    assert model.call_count == 2
    assert prediction == {**BASE, "workspace_update": prediction["workspace_update"]}
    update = prediction["workspace_update"]
    assert update["status"] == "insufficient_evidence" and update["remaining_min"] is None
    assert not update["evidence"] and update["confidence"] == "low"


def test_prior_citation_cannot_apply_twice_but_can_inform_a_new_scope_baseline(monkeypatch):
    quote = "My recurrence is solved."
    progress = snapshot(("user", quote), ("user", "Can you explain the boundary condition?"), new_message_start=1)
    model = Mock(side_effect=[supported(quote), {"assessment": "insufficient_evidence"}, supported(quote)])
    monkeypatch.setattr(forecast, "chat_json", model)
    prediction = forecast.revise(TASK, BASE, progress)
    assert prediction["workspace_update"]["status"] == "insufficient_evidence"
    payload = json.loads(model.call_args.args[0][1]["content"])
    assert [row["fresh"] for row in payload["messages"]] == [False, True]
    combined_refresh = forecast.revise(TASK, BASE, progress, refreshed_history=True)
    assert combined_refresh["workspace_update"]["status"] == "adjusted"


def test_offline_and_persistently_ungrounded_models_keep_saved_forecast(monkeypatch):
    progress = snapshot(("user", "I checked the base case."))
    monkeypatch.setenv("PLANNER_MODEL_ENDPOINT", "http://127.0.0.1:1234/v1/chat/completions")
    for model in (Mock(side_effect=ModelUnavailable("Local endpoint unavailable.")),
                  Mock(return_value=supported("The full proof is finished."))):
        monkeypatch.setattr(forecast, "chat_json", model)
        prediction = forecast.revise(TASK, BASE, progress)
        update = prediction["workspace_update"]
        assert (prediction["point_min"], prediction["provenance"]) == (BASE["point_min"], BASE["provenance"])
        assert update["status"] == "unavailable" and update["remaining_min"] is None
        assert update["provenance"]["provider"] == "Local endpoint" and prediction["retryable"]
        assert "could not" in update["explanation"] and not update["evidence"]


def test_confirmed_blocker_can_raise_total_and_keep_unresolved_retrieval_warning(monkeypatch):
    quote = "The boundary case fails for zero and I need a different proof."
    reply = supported(quote)
    reply.update(remaining_min=40, remaining_low_min=25, remaining_high_min=65,
                 explanation="The zero case needs a second proof.")
    reply["evidence"][0]["kind"] = "blocker"
    monkeypatch.setattr(forecast, "chat_json", Mock(return_value=reply))
    failed_base = {**BASE, "retryable": True, "model_error": "Earlier request failed."}
    for warning in ({}, {"retrieval_error": "A source was unavailable."}):
        revised = forecast.revise(TASK, {**failed_base, **warning}, snapshot(("user", quote)))
        assert revised["point_min"] == 46 and revised["workspace_update"]["remaining_min"] == 40
        assert "model_error" not in revised and bool(revised.get("retryable")) == bool(warning)
        assert revised.get("retrieval_error") == warning.get("retrieval_error")


def test_insufficient_assessment_retains_why_and_clears_resolved_progress_failure(monkeypatch):
    progress = snapshot(("user", "Can you explain convergence?"))
    baseline = forecast.unavailable(progress, BASE, "The endpoint was offline.")
    monkeypatch.setattr(forecast, "chat_json", Mock(return_value={
        "assessment": "insufficient_evidence", "explanation": "You asked about convergence without sharing a proof.",
        "confidence_reason": "The question gives no indication of which cases you already checked.",
    }))
    retained = forecast.revise(TASK, baseline, progress)
    assert not retained.get("retryable") and retained["point_min"] == BASE["point_min"]
    assert "asked about convergence" in retained["workspace_update"]["explanation"]
    assert "already checked" in retained["workspace_update"]["confidence_reason"]


def test_unrevised_minutes_keep_their_last_verified_evidence_across_pauses(monkeypatch):
    quote = "My recurrence is solved."
    progress = snapshot(("user", quote))
    newer = snapshot(("user", quote), ("user", "Would another approach work?"), new_message_start=1)
    model = Mock(side_effect=[supported(quote), {"assessment": "insufficient_evidence"},
                              ModelUnavailable("Offline"), {"assessment": "insufficient_evidence"}])
    monkeypatch.setattr(forecast, "chat_json", model)
    original = forecast.revise(TASK, BASE, progress)
    insufficient = forecast.revise(TASK, original, newer)
    failed = forecast.revise(TASK, insufficient, newer)
    for prediction in (insufficient, failed):
        update = prediction["workspace_update"]
        assert prediction["point_min"] == original["point_min"] and update["evidence"] == []
        assert update["last_adjusted"] == original["workspace_update"]
        assert update["last_adjusted"]["evidence"][0]["quote"] == quote
        assert "last_adjusted" not in update["last_adjusted"]
    fresh_history = forecast.revise(TASK, BASE, newer, refreshed_history=True)
    assert "last_adjusted" not in fresh_history["workspace_update"]
