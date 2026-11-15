from unittest.mock import Mock

import pytest

from model_client import ModelUnavailable
from planner_chat import _rank_plan


def choice(task_id, reason="Uses the supplied budget."):
    return {"id": task_id, "reason": reason}


def test_valid_sequence_preserves_exact_order_and_explanations_without_second_call():
    decide = Mock(return_value={"order": [choice(2.0, "Fifteen minutes remain."), choice(1, "Needs an hour.")]})
    ranks, note, source = _rank_plan([{"id": 1}, {"id": 2}], {}, decide)
    assert decide.call_count == 1 and not note and source == "model sequence"
    assert ranks == {2: {"rank": 1, "reason": "Fifteen minutes remain."},
                     1: {"rank": 2, "reason": "Needs an hour."}}


@pytest.mark.parametrize("bad_order", [
    [], [choice(2)], [choice(2), choice(2)], [choice(2), choice(1), choice(1)],
    [choice(3), choice(1)], [choice(2), choice(1, "  ")], [choice(True), choice(2)],
])
def test_incomplete_or_ambiguous_sequence_requires_one_keyed_repair(bad_order):
    decide = Mock(side_effect=[{"order": bad_order}, {"priorities": {
        "1": {"rank": 1, "reason": "Can start first."}, "2": {"rank": 1, "reason": "Equally viable."}}}])
    ranks, note, source = _rank_plan([{"id": 1}, {"id": 2}], {}, decide)
    assert decide.call_count == 2 and "same model rank" in note and source == "model coverage repair"
    assert set(ranks) == {1, 2} and ranks[2]["reason"] == "Equally viable."


@pytest.mark.parametrize("responses", [[ModelUnavailable("offline")],
                                       [{"order": []}, ModelUnavailable("offline")]])
def test_unavailable_model_marks_all_priorities_as_local_and_stops(responses):
    decide = Mock(side_effect=responses)
    ranks, note, source = _rank_plan([{"id": 1}], {}, decide)
    assert not ranks and "offline" in note and source == "local plan order"
    assert decide.call_count == len(responses)


@pytest.mark.parametrize("instruction", [{"rank": 0, "reason": "First"}, {"rank": 1, "reason": "  "}])
def test_invalid_required_reason_or_rank_falls_back_once(instruction):
    decide = Mock(side_effect=[{"order": []}, {"priorities": {"1": instruction}}])
    ranks, note, source = _rank_plan([{"id": 1}], {}, decide)
    assert not ranks and "omitted" in note and source == "local plan order"
    assert decide.call_count == 2


def test_malformed_model_sequence_has_one_repair_but_no_network_retry():
    from model_client import ModelResponseInvalid

    decide = Mock(side_effect=[ModelResponseInvalid("invalid order"), {"priorities": {
        "1": {"rank": 1, "reason": "Fits the open window."}}}])
    ranks, note, source = _rank_plan([{"id": 1}], {}, decide)
    assert decide.call_count == 2 and not note and source == "model coverage repair"
    assert ranks[1]["reason"] == "Fits the open window."


@pytest.mark.parametrize("invalidated_at", ["first incomplete", "first complete", "during repair"])
def test_changed_generation_drops_plan_and_never_resends_purged_task(invalidated_at):
    from datetime import datetime
    from unittest.mock import patch

    from planner_analytics import ask_analytics
    from planner_chat import planning_context

    now = datetime(2026, 11, 13, 18)
    plan = planning_context([{"title": "Discarded draft", "id": "remove", "day": now.date().isoformat(),
                              "estimate_minutes": 15, "status": "ready"}], now.date(), now)
    current = True
    def respond(messages, schema, **kwargs):
        nonlocal current
        if "tool" in schema["properties"]:
            return {"tool": "plan_remaining", "sql": "", "scope": "all", "task_ids": []}
        assert current, "A stale planned task was sent again."
        if "order" in schema["properties"]:
            current = invalidated_at == "during repair"
            return {"order": [choice(1)] if invalidated_at == "first complete" else []}
        current = False
        return {"priorities": {"1": {"rank": 1, "reason": "Discarded draft fits."}}}

    with patch("planner_analytics.chat_json", side_effect=respond) as model:
        result = ask_analytics("I have 20 minutes. Which task fits?", [], [], plan=plan,
                               still_current=lambda: current)
    assert model.call_count == (3 if invalidated_at == "during repair" else 2)
    assert "plan_rows" not in result and "Discarded draft" not in str(result)
    assert "records changed" in result["answer"]
