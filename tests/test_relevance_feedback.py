import pytest
from streamlit.testing.v1 import AppTest

from planner_store import PlannerStore


BUDGET = {"point_min": 20, "low_min": 15, "high_min": 25, "confidence": "medium",
          "basis": "User estimate", "related_tasks": [], "evidence": []}


def add(store, title, *, prediction=BUDGET):
    return store.create_task(day="2026-10-28", title=title, description=f"Prepare {title}.",
                             category="Work", estimate_minutes=20, prediction=prediction)


@pytest.fixture
def comparisons(tmp_path):
    now = 1000
    store = PlannerStore(tmp_path, clock=lambda: now)
    sources = []
    for name in ("Shared screen setup", "Team minutes", "Laundry"):
        item = add(store, name)
        store.transition(item["id"], "start")
        now += 60
        store.transition(item["id"], "stop")
        store.transition(item["id"], "finish")
        sources.append(item)
    budget = dict(BUDGET, related_tasks=[{"task_id": sources[1]["id"], "title": sources[1]["title"]}])
    target = add(store, "Check screen connectivity", prediction=budget)
    return store, target, [source["id"] for source in sources]


def test_revised_pool_survives_restart_and_teaches_both_directions(comparisons):
    store, task, pool = comparisons
    keep, drop, unrelated = pool
    pair = sorted((task["id"], drop))
    with store._db(write=True) as db:
        db.execute("INSERT INTO task_relations(left_id,right_id,related) VALUES(?,?,1)", pair)
        db.execute("INSERT INTO task_relation_sources(left_id,right_id,source_id) VALUES(?,?,?)", (*pair, unrelated))
    store.revise_related_tasks(task["id"], [keep], previous_ids=[drop], reviewed_ids=pool)
    store = PlannerStore(store.data_dir)
    changed = store.claim_prediction()
    assert changed["task"]["relevance_overrides"] == {keep: True, drop: False, unrelated: False}
    assert changed["task"]["relevance_examples"] == []
    with store._db() as db:
        assert sorted(row[0] for row in db.execute("SELECT related FROM task_relations")) == [0, 1]
        assert not db.execute("SELECT 1 FROM task_relation_sources").fetchone()
    store.resolve_prediction(task["id"], changed["token"], BUDGET)
    second = add(store, "Set up a video call", prediction=None)
    future = store.claim_prediction()
    assert future["task"]["id"] == second["id"]
    examples = future["task"]["relevance_examples"]
    assert {(row["candidate_id"], row["related"]) for row in examples} == {(keep, True), (drop, False)}
    assert all(row["target_id"] == task["id"] and row["target"]["title"] == task["title"] for row in examples)
    store.delete_task(keep)
    assert not store.resolve_prediction(second["id"], future["token"], BUDGET)
    remaining = store.claim_prediction()["task"]["relevance_examples"]
    assert [(row["candidate_id"], row["related"]) for row in remaining] == [(drop, False)]


def test_revisions_are_atomic_and_allow_explicit_empty_selection(comparisons):
    store, task, pool = comparisons
    for selected, old, seen in (([task["id"]], pool[1:2], pool), ([], [], pool), ([], pool[1:2], pool[:2])):
        with pytest.raises(ValueError):
            store.revise_related_tasks(task["id"], selected, previous_ids=old, reviewed_ids=seen)
        assert not store.has_pending_predictions()
    store.archive_task(pool[0])
    assert {row["id"] for row in store.comparison_options(task["id"])} == set(pool)
    store.revise_related_tasks(task["id"], [], previous_ids=pool[1:2], reviewed_ids=pool)
    assert store.claim_prediction()["task"]["relevance_overrides"] == dict.fromkeys(pool, False)


def test_editor_opens_with_suggested_sources_and_can_save_an_empty_selection(comparisons):
    store, task, pool = comparisons
    app = AppTest.from_string(f"""
from app import edit_comparisons
from planner_store import PlannerStore
store = PlannerStore({str(store.data_dir)!r})
task = store.get_task({task["id"]!r})
if not task["estimate_updating"]:
    edit_comparisons(store, task, store.comparison_options(task["id"]))
""").run()
    assert not app.exception
    assert app.multiselect[0].value == [pool[1]]
    app.multiselect[0].set_value([])
    next(b for b in app.button if b.label == "Save and re-estimate").click().run()
    assert not app.exception
    assert store.claim_prediction()["task"]["relevance_overrides"] == dict.fromkeys(pool, False)


def test_reviewed_pairs_keep_pdf_scope_for_future_decision_and_forecast(tmp_path):
    import retrieval

    store = PlannerStore(tmp_path)

    def assignment(text, prediction=BUDGET):
        return store.create_task(day="2026-10-28", title="Assignment", description="Finish the attachment",
                                 category="Math", estimate_minutes=30, pdf_text=text, prediction=prediction)["id"]

    target_page = "Page 2: derive the variance bound and prove the final inequality."
    prove_page = "Page 1: derive the sampling variance and verify it analytically."
    code_page = "Page 1: implement a Monte Carlo run and graph the observed variance."
    prove, code = assignment(prove_page), assignment(code_page)
    for source in (prove, code):
        store.transition(source, "finish", spent_seconds=1800, completion_note="PRIVATE OUTCOME")
    target = assignment(target_page, dict(BUDGET, related_tasks=[{"task_id": code}]))
    store.revise_related_tasks(target, [prove], previous_ids=[code], reviewed_ids=[prove, code])
    changed = store.claim_prediction()
    store.resolve_prediction(target, changed["token"], BUDGET)
    add(store, "Follow-up assignment", prediction=None)
    job = store.claim_prediction()
    # Both the forecast and Kev serialize corrections with reviewed_examples().
    examples = retrieval.reviewed_examples(job["task"]["relevance_examples"])
    by_decision = {ex["related"]: ex["candidate"]["document_excerpt"] for ex in examples}
    assert by_decision == {True: prove_page, False: code_page}
    assert all(ex["target"]["document_excerpt"] == target_page for ex in examples)
    assert "PRIVATE OUTCOME" not in str(examples)
