from planner_store import PlannerStore
from estimator import _observations
from activity import reviewed_examples


def test_exclusions_changed_after_completion_apply_to_queued_history_and_reviews(tmp_path):
    store = PlannerStore(tmp_path)
    original = store.create_task(day="2026-11-10", title="Calibrate A", description="Calibrate sensor",
                                 estimate_minutes=20, prediction={"point_min": 20})
    old = {"start": 200, "seconds": 20, "app": "browser", "phone_app": "Private Chat",
           "domain": "sports.test", "title": "Fixture", "label": "off_task", "label_source": "user"}
    keep = {**old, "start": 220, "phone_app": "", "domain": "scores.test"}
    store.transition(original["id"], "finish", spent_seconds=1500, report={"intervals": [old, keep]})
    new = store.create_task(day="2026-11-11", title="Calibrate B", description="Calibrate sensor", estimate_minutes=20)
    first = store.claim_prediction()
    assert first["history"][0]["report"]["intervals"] == [old, keep]
    store.set_setting("exclusions", ["pRiVaTe chAt"])
    assert not store.renew_prediction(new["id"], first["token"])
    second = store.claim_prediction()
    prior, = second["history"]
    assert prior["report"]["intervals"] == [keep]
    observations = str(_observations(prior)) + str(reviewed_examples(prior["report"]))
    assert "scores.test" in observations and "sports.test" not in observations
    assert store.get_task(original["id"])["report"]["intervals"] == [old, keep]
