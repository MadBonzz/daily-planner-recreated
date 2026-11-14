"""Task matches must establish relevant work, even when a model selects every slot."""

import json

import pytest

from estimator import grounded_candidates, grounded_comparison
from examples.estimation.run import DATASET


def clinic_rows():
    history = json.loads((DATASET / "histories" / "06-neighborhood-clinic.json").read_text(encoding="utf-8"))
    return {row["id"]: row for row in history["open_tasks"]}, history["tasks"]


def test_shared_location_and_category_are_insufficient_for_an_estimate():
    targets, completed = clinic_rows()
    ranked = [(row, 1.0) for row in completed]  # Even a high vector score cannot supply the evidence.
    assert [row["id"] for row, _ in grounded_candidates(targets["paused-flyer"], ranked)] == ["campus-posters"]
    assert grounded_candidates(targets["ready-door"], ranked) == []


def test_writing_and_copyediting_remain_eligible_across_categories_and_paraphrases():
    task = {"title": "Copyedit the application packet", "description": "Correct grammar and formatting.",
            "category": "Study"}
    prior = {"title": "Proofread a packet", "description": "Fix formatting and mark typos.",
             "category": "Career"}
    assert grounded_candidates(task, [(prior, .12)]) == [(prior, .12)]
    rationale = "Both tasks require editing a packet and correcting formatting."
    evidence = grounded_comparison(task, prior, rationale)
    assert evidence["model_rationale"] == rationale and evidence["reason_source"] == "model"


@pytest.mark.parametrize("source_id,reason", [
    ("mara-evening", "Both involved confirming changes with the lead."),
    ("bus-pass", "Both tasks involved printing an updated door flyer."),
    ("campus-posters", "The flyer and signs are not comparable."),
    ("campus-posters", "Collecting and analyzing sensor data."),
    ("campus-posters", "The category is the same."),
])
def test_model_cannot_use_unrelated_work_or_a_generic_explanation(source_id, reason):
    targets, completed = clinic_rows()
    source = next(row for row in completed if row["id"] == source_id)
    assert grounded_comparison(targets["paused-flyer"], source, reason) is None


@pytest.mark.parametrize("reason", [
    "Both require changing a posted sign and printing a replacement for the entrance.",
    "Replacing signs",
])
def test_resolves_the_original_description_for_a_paraphrased_model_explanation(reason):
    targets, completed = clinic_rows()
    prior = next(row for row in completed if row["id"] == "campus-posters")
    evidence = grounded_comparison(targets["paused-flyer"], prior, reason)
    assert evidence == {"reason": reason, "reason_source": "model", "model_rationale": reason,
                        "source_quote": prior["description"]}


def test_short_live_rover_comparison_accepts_a_paraphrase_and_uses_saved_description():
    h = json.loads((DATASET / "histories" / "09-rover-night.json").read_text(encoding="utf-8"))
    target = next(row for row in h["holdouts"] if row["id"] == "bump-crossing")
    prior = next(row for row in h["tasks"] if row["id"] == "hall-encoder")
    reason = "Collecting and analyzing data from a rover"
    evidence = grounded_comparison(target, prior, reason)
    assert evidence["model_rationale"] == reason and evidence["source_quote"] == prior["description"]
