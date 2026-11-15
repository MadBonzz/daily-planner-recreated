"""Exercise the boundary between a committed pair decision and the actual forecast inputs."""

import json

import pytest

import estimate_worker
from planner_store import PlannerStore


def card(store, title, *, fixed=False):
    return store.create_task(day="2026-11-15", title=title, description=f"Do {title}.",
                             category="Project", estimate_minutes=30, fixed_duration=fixed,
                             prediction={"point_min": 30, "explanation": "Initial estimate."})["id"]


def no_more_searches(*args, **kwargs):
    raise AssertionError("A committed active-pair decision was searched or judged again.")


@pytest.mark.parametrize("minutes", [10, 90])
def test_independent_completion_is_a_noop_then_real_duration_reaches_forecast(tmp_path, monkeypatch, minutes):
    store = PlannerStore(tmp_path)
    store.set_setting("dev_replay_enabled", True)
    target, related, other = (card(store, name) for name in ("Sign package B", "Sign package A", "Buy tape"))
    decisions, payloads = [], []

    def judge(left, right, **kwargs):
        pair = frozenset((left["id"], right["id"]))
        decisions.append(pair)
        return pair == {target, related}

    def forecast(messages, schema, **kwargs):
        history = json.loads(messages[-1]["content"])["history"]
        payloads.append(history)
        point = round(history[0]["actual_minutes"])
        return {"point_min": point, "low_min": max(1, point - 5), "high_min": point + 5,
                "confidence": "medium", "rationale": "These two tasks package the same release.",
                "confidence_reason": "One directly comparable signing run.",
                "comparisons": [{"slot": 1, "reason": "Both tasks sign the release package."}], "warnings": []}

    monkeypatch.setattr(estimate_worker.estimator, "_search", no_more_searches)
    monkeypatch.setattr(estimate_worker.estimator, "chat_json", forecast)
    monkeypatch.setattr(estimate_worker.estimator, "index_current_task", lambda *args: None)
    monkeypatch.setattr(estimate_worker.retrieval, "decide_related", judge)
    def drain():
        while estimate_worker.run_one_job(store, tmp_path / "vectors"):
            pass
    drain()
    assert len(set(decisions)) == len(decisions) == 3
    monkeypatch.setattr(estimate_worker.retrieval, "decide_related", no_more_searches)
    store.transition(other, "finish", spent_seconds=70 * 60)
    drain()
    assert not payloads and len(store.forecast_timeline(target)) == 1
    store.transition(related, "finish", spent_seconds=minutes * 60)
    drain()
    assert [[(r["title"], r["actual_minutes"]) for r in batch] for batch in payloads] == [[("Sign package A", minutes)]]
    updated = store.get_task(target)
    assert updated["creation_prediction"]["point_min"] == 30 and updated["prediction"]["point_min"] == minutes
    assert updated["prediction"]["related_tasks"] == [
        {"task_id": related, "title": "Sign package A", "source": "active-task decision model", "score": None}]
    assert updated["prediction"]["used_task_ids"] == [related]
    assert len(store.forecast_timeline(target)) == 2


def test_two_completions_coalesce_and_an_identical_rebuilt_report_keeps_lease(tmp_path):
    store = PlannerStore(tmp_path)
    target, a, b = [card(store, s) for s in ("Library run C", "Library run A", "Library run B")]
    # A and B both finish before all three open-pair decisions resolve.
    original = {"summary": "Checked all labels.", "intervals": []}
    store.transition(a, "finish", spent_seconds=1200, report=original)
    store.transition(b, "finish", spent_seconds=2400)
    first = store.claim_relation()
    assert store.resolve_relation(first["pair"], first["token"], True)
    assert store.claim_prediction() is None  # Finish the other already-due edge before the expensive forecast.
    second = store.claim_relation()
    assert store.resolve_relation(second["pair"], second["token"], True)
    job = store.claim_prediction()
    assert {h["id"]: h["matrix_related"] for h in job["history"]} == {a: True, b: True}
    assert store.save_report(a, dict(reversed(list(original.items()))))
    assert store.resolve_prediction(target, job["token"], {"point_min": 31})
    assert not store.has_estimation_work()
    store.save_report(a, {"summary": "One label was damaged.", "intervals": []})
    assert store.claim_prediction()["task"]["id"] == target


def test_offline_edge_does_not_block_a_refresh_from_a_known_source(tmp_path):
    now = [1000]
    store = PlannerStore(tmp_path, clock=lambda: now[0])
    target, a, b = [card(store, s) for s in ("Remaining", "Measured 1", "Measured 2")]
    store.transition(a, "finish", spent_seconds=1200)
    store.transition(b, "finish", spent_seconds=2400)
    first = store.claim_relation()
    assert store.resolve_relation(first["pair"], first["token"], True)
    second = store.claim_relation()
    assert store.resolve_relation(second["pair"], second["token"], None)
    job = store.claim_prediction()
    assert job["task"]["id"] == target
    assert store.resolve_prediction(target, job["token"], {"point_min": 31})
    assert store.next_estimation_delay() == 15
    now[0] += 16
    retry = store.claim_relation()
    assert retry["pair"] == second["pair"]
    assert store.resolve_relation(retry["pair"], retry["token"], False)
    assert not store.has_estimation_work()


def test_fixed_source_ignored_automatically_but_explicit_inclusion_wins(tmp_path):
    store = PlannerStore(tmp_path)
    a = card(store, "Attend onboarding", fixed=True)
    t = card(store, "Flexible onboarding appointment")
    pair = store.claim_relation()
    assert store.resolve_relation(pair["pair"], pair["token"], True)
    store.transition(a, "finish", spent_seconds=2100, queue_report=True)
    assert not store.get_task(t)["awaiting_related_update"]
    assert store.claim_prediction() is None
    first = store.report_job(a)
    assert store.save_report(a, {"summary": "Supervisor was late."}, generation=first["generation"])
    assert store.claim_prediction() is None
    store.revise_related_tasks(t, [a], previous_ids=[], reviewed_ids=[a])
    choice = store.claim_prediction()
    assert choice["task"]["relevance_overrides"] == {a: True}
    assert store.resolve_prediction(t, choice["token"], {"point_min": 36, "related_tasks": [{"task_id": a}]})
    store.transition(a, "adjust_time", spent_seconds=2500)
    updated = store.claim_prediction()
    assert updated["task"]["id"] == t and updated["history"][0]["actual_seconds"] == 2500


def test_rebuilding_identical_report_does_not_replace_an_active_forecast(tmp_path):
    store = PlannerStore(tmp_path)
    target, source = card(store, "Inspect part B"), card(store, "Inspect part A")
    pair = store.claim_relation()
    store.resolve_relation(pair["pair"], pair["token"], True)
    report = {"summary": "Inspected parts.", "intervals": []}
    store.transition(source, "finish", spent_seconds=1300, report=report)
    in_flight = store.claim_prediction()
    store.save_report(source, {"intervals": [], "summary": "Inspected parts."})
    assert store.resolve_prediction(target, in_flight["token"], {"point_min": 24})
    assert not store.has_estimation_work()


def test_forgetting_an_example_revokes_a_matrix_backed_forecast_and_rejudges(tmp_path, monkeypatch):
    store = PlannerStore(tmp_path)
    example = card(store, "Corrected historical example")
    store.transition(example, "finish", spent_seconds=600)
    target, source = card(store, "Book inventory B"), card(store, "Book inventory A")
    pair = store.claim_relation()
    store.resolve_relation(pair["pair"], pair["token"], True, examples=[{"candidate_id": example}])
    store.transition(source, "finish", spent_seconds=3600)
    monkeypatch.setattr(estimate_worker.estimator, "_search", lambda *args, **kw: [])
    monkeypatch.setattr(estimate_worker.estimator, "index_current_task", lambda *args: None)
    def offline(*args, **kwargs):
        raise estimate_worker.ModelUnavailable("Offline for this scenario.")
    monkeypatch.setattr(estimate_worker.estimator, "chat_json", offline)
    assert estimate_worker.run_one_job(store, tmp_path / "vectors")
    prior = store.get_task(target)["prediction"]
    assert [item["task_id"] for item in prior["related_tasks"]] == [source]
    assert example in prior["feedback_source_ids"]
    store.delete_task(example)
    assert not store.get_task(target)["prediction"]["related_tasks"]
    pair = store.claim_relation()
    assert set(pair["pair"]) == {target, source}
    store.resolve_relation(pair["pair"], pair["token"], False)
    assert estimate_worker.run_one_job(store, tmp_path / "vectors")
    assert not store.get_task(target)["prediction"]["related_tasks"]
    assert not store.has_estimation_work()


def test_interleaved_work_preserves_each_initial_snapshot_and_final_matrix(tmp_path, monkeypatch):
    """Twelve deterministic interleavings of seven submissions, four timed completions and worker ticks."""
    from itertools import combinations
    import random

    def key(task):
        return task["title"].split()[0]

    def predict(task, history, path):
        selected = sorted(h["id"] for h in history if key(h) == key(task))
        return {"point_min": 30 + len(selected), "explanation": "Controlled source set.",
                "related_tasks": [{"task_id": candidate} for candidate in selected]}

    monkeypatch.setattr(estimate_worker.estimator, "predict", predict)
    monkeypatch.setattr(estimate_worker.estimator, "index_current_task", lambda *args: None)
    monkeypatch.setattr(estimate_worker.retrieval, "decide_related", lambda left, right, **kw: key(left) == key(right))

    for seed in range(12):
        rng = random.Random(seed)
        store = PlannerStore(tmp_path / str(seed))
        saved, expected_initial, completed = {}, {}, set()
        active_pairs = set()
        for i in range(7):
            opened = list(saved.keys() - completed)
            opened.sort(key=lambda t: saved[t]["title"])
            if i >= 3:
                source = rng.choice(opened)
                if store.get_task(source)["status"] == "running":
                    store.transition(source, "stop")
                store.transition(source, "finish", spent_seconds=rng.choice([900, 2700, 5400]))
                completed.add(source)
                opened.remove(source)
            title = f"{('Ledger', 'Repairs', 'Ledger')[i % 3]} {i}"
            task = store.create_task(day="2026-11-15", title=title, description=title, estimate_minutes=30,
                                     category="Work")
            saved[task["id"]] = task
            active_pairs.update(frozenset((task["id"], peer)) for peer in opened)
            expected_initial[task["id"]] = {k for k in completed if key(saved[k]) == key(task)}
            if rng.random() < .6:
                store.transition(task["id"], "start")
            for _ in range(rng.randrange(4)):
                estimate_worker.run_one_job(store, store.data_dir / "vectors")
        while estimate_worker.run_one_job(store, store.data_dir / "vectors"):
            pass
        active = saved.keys() - completed
        with store._db() as db:
            edges = {frozenset((a, b)): v for a, b, v in db.execute("SELECT left_id,right_id,related FROM task_relations")}
        assert set(edges) <= active_pairs
        assert {frozenset(p) for p in combinations(active, 2)} <= set(edges)
        for pair in active_pairs:
            if pair & active:
                a, b = pair
                assert edges[pair] == int(key(saved[a]) == key(saved[b])), seed
        for task_id, original in saved.items():
            state = store.get_task(task_id)
            assert {r["task_id"] for r in state["creation_prediction"]["related_tasks"]} == expected_initial[task_id], seed
            if task_id in active:
                assert {r["task_id"] for r in state["prediction"]["related_tasks"]} == {
                    k for k in completed if key(saved[k]) == key(original)}, seed
        assert not store.has_estimation_work()


def test_same_active_pair_uses_same_judge_input_and_feedback_in_either_order(tmp_path, monkeypatch):
    import retrieval

    payloads = []
    def record(left, right, examples):
        payloads.append(retrieval.decision_payload(left, right, examples, retrieval.DECISION_MODEL))
        return left["description"].startswith("Check")

    monkeypatch.setattr(retrieval, "decide_related", record)
    for i, order in enumerate([(0, 1), (1, 0)]):
        store = PlannerStore(tmp_path / str(i))
        first = card(store, "Past electronics review")
        store.transition(first, "finish", spent_seconds=900)
        past = card(store, "Scheduling")
        store.revise_related_tasks(past, [first], previous_ids=[], reviewed_ids=[first])
        lease = store.claim_prediction()
        store.resolve_prediction(past, lease["token"], {"point_min": 30, "related_tasks": [{"task_id": first}]})
        store.transition(past, "finish", spent_seconds=1200)
        titles = [("Bench check", "Check voltage across the test terminals.", "Lab"),
                  ("Bench check", "Test the supply output with the multimeter.", "Study")]
        ids = []
        for item in order:
            title, description, category = titles[item]
            row = store.create_task(day="2026-11-15", title=title, description=description,
                                    category=category, estimate_minutes=30, prediction={"point_min": 30})
            ids.append(row["id"])
        assert estimate_worker.run_one_job(store, store.data_dir / "vectors")
        with store._db() as db:
            verdict = db.execute("SELECT related FROM task_relations WHERE left_id=? AND right_id=?",
                                 sorted(ids)).fetchone()[0]
        assert verdict
    assert payloads[0] == payloads[1]
    assert json.loads(payloads[0]["state"])["reviewed_examples"]


def test_reverse_only_packaged_vote_is_preserved_under_both_insertion_orders(monkeypatch):
    import retrieval

    a, z = ({"title": name} for name in ("A paper", "Z paper"))
    legacy = retrieval.decision_payload(z, a, (), retrieval.DECISION_MODEL)
    monkeypatch.setattr(retrieval.sample_retrieval, "decision_for", lambda p: True if p == legacy else None)
    monkeypatch.setattr(retrieval, "_local_post", no_more_searches)
    for first, second in ((a, z), (z, a)):
        l, r = retrieval.active_decision_order(first, second, ())
        assert l == z and r == a
        assert retrieval.decide_related(l, r)
