import json

import pytest

import focus_judge

from activity import activity_key
from focus_domains import normalize_domain, policy, policy_from_db, save_domains
from planner_store import PlannerStore


def open_task(store, title, sources=(), **extra):
    return store.create_task(title=title, description="Implement and benchmark the kernel", category="Study",
                             estimate_minutes=60, day="2026-11-14", prediction={
                                 "point_min": 60, "related_tasks": [{"task_id": key} for key in sources]},
                             **extra)["id"]


def visit(domain="x.com", label="off_task", seconds=120, **extra):
    return {"app": "Chrome", "title": "Thread", "domain": domain, "label": label,
            "seconds": seconds, "idle": False, "excluded": False, **extra}


def history(store, title, visits):
    task = open_task(store, title)
    store.transition(task, "finish", spent_seconds=3600, report={"intervals": visits})
    return task


@pytest.fixture
def approved_repeats(monkeypatch):
    """These policy composition tests supply a positive Kev verdict only for repeated evidence."""
    def verdict(request):
        return True if "previous_tasks" in json.loads(request["state"]) else None
    monkeypatch.setattr(focus_judge, "packaged", verdict)


def test_recurring_across_distinct_related_tasks_with_live_edits_and_deletion(tmp_path, approved_repeats):
    store = PlannerStore(tmp_path)
    a = history(store, "Tiling kernel", [visit(), visit(seconds=60), visit("tutorial.edu", "on_task")])
    b = history(store, "Shared memory", [visit("www.x.com"), visit("tutorial.edu")])
    unrelated = history(store, "Shopping", [visit("sports.example")])
    target = open_task(store, "Matrix multiply", [a])
    initial = policy(store, target)
    assert initial["domains"] == []
    assert "x.com" in initial["pending"]  # Both visits belong to one task; its Kev review is still missing.
    assert store.retry_estimate(target)
    lease = store.claim_prediction()
    assert store.resolve_prediction(target, lease["token"], {"related_tasks": [{"task_id": a}, {"task_id": b}]})
    chosen = policy(store, target)
    assert chosen["domains"] == ["x.com"]
    assert "5.0 minutes" in chosen["suggestions"]["x.com"] and "Tiling kernel" in chosen["suggestions"]["x.com"]
    with store._db() as db:
        assert policy_from_db(db, target) == chosen
    save_domains(store, target, ["https://www.scores.example/latest"], chosen["domains"])
    assert policy(store, target)["domains"] == ["scores.example"]
    reopened = PlannerStore(tmp_path)
    assert policy(reopened, target)["removed"] == ["x.com"]
    reopened.archive_task(a)
    assert "x.com" in policy(reopened, target)["suggestions"]
    reopened.delete_task(a)
    assert policy(reopened, target)["suggestions"] == {}
    assert policy(reopened, target)["domains"] == ["scores.example"]
    assert policy(reopened, unrelated)["domains"] == []
    reopened.delete_task(target)
    with reopened._db() as db:
        assert not db.execute("SELECT 1 FROM focus_domain_choices WHERE task_id=?", (target,)).fetchone()


def test_mixed_use_phone_exclusions_target_links_and_label_correction(tmp_path, approved_repeats):
    store = PlannerStore(tmp_path)
    noisy = [visit("www.youtube.com"), visit("notes.example"), visit("overlap.example", phone_app="Sports"),
             visit("hidden.example", excluded=True), visit("idle.example", idle=True), visit("private.example"),
             visit("needed.example"), visit("forum.example", label="uncertain")]
    a = history(store, "Kernel A", noisy + [visit("music.youtube.com", "on_task")])
    b = history(store, "Kernel B", noisy)
    t = open_task(store, "Tune kernels", [a, b], details="Check https://needed.example. Then record timings.")
    store.set_setting("exclusions", ["PRIvate.example"])
    assert policy(store, t)["domains"] == ["notes.example"]
    store.request_report(a, {activity_key(visit("notes.example")): "on_task"})
    assert policy(store, t)["domains"] == []  # The submitted correction applies before report rebuilding.


def test_saved_edits_merge_with_new_suggestions_and_manual_relations(tmp_path, approved_repeats):
    store = PlannerStore(tmp_path)
    a = history(store, "Left", [visit("x.com"), visit("news.example")])
    b = history(store, "Right", [visit("x.com")])
    target = open_task(store, "Next", [a, b])
    snapshot = policy(store, target)["domains"]
    assert snapshot == ["x.com"]
    store.save_report(b, {"intervals": [visit("x.com"), visit("news.example")]})
    save_domains(store, target, ["games.example"], snapshot)
    assert policy(store, target)["domains"] == ["games.example", "news.example"]
    save_domains(store, a, [], ["news.example"])
    assert policy(store, target)["domains"] == ["games.example"]
    before = policy(store, target)
    with pytest.raises(ValueError):
        save_domains(store, target, ["safe.example", "*.net"], before["domains"])
    assert policy(store, target) == before
    refresh = store.claim_prediction()
    assert refresh["task"]["id"] == target
    assert store.resolve_prediction(target, refresh["token"], store.get_task(target)["prediction"])
    store.revise_related_tasks(target, [a], previous_ids=[a, b], reviewed_ids=[a, b])
    assert policy(store, target)["suggestions"] == {}


@pytest.mark.parametrize(("input", "expected"), [
    ("HTTPS://WWW.Example.com/path?q=1", "example.com"), ("mail.example.org:443", "mail.example.org"),
    ("https://bücher.de", "xn--bcher-kva.de"), ("example.com.", "example.com"),
])
def test_normalized_host(input, expected):
    assert normalize_domain(input) == expected


@pytest.mark.parametrize("bad", ["https://user@site.com", "*.example.com", "localhost", "127.0.0.1",
                                 "site..org", "example.com:bad", "a.com\\b.org", "chrome://settings",
                                 "site.example extra", "example.com:99999", ".example.com", "foo.localhost"])
def test_invalid_block_domain(bad):
    with pytest.raises(ValueError):
        normalize_domain(bad)


def test_kev_can_reject_a_recurring_platform_needed_for_the_current_task(tmp_path, monkeypatch):
    store = PlannerStore(tmp_path)
    sources = [history(store, title, [visit("chess.com", title="Play blitz online")])
               for title in ("Morning notebook", "Afternoon notebook")]
    task = open_task(store, "Analyze my past blitz games", sources, details="Open my Chess games and compare the moves.")
    # Missing, negative and positive Kev decisions are distinct. Page renders never run inference.
    captured = []
    monkeypatch.setattr(focus_judge, "packaged", lambda request: captured.append(request))
    proposed = policy(store, task)
    assert "chess.com" in proposed["pending"] and not proposed["domains"]
    state = json.loads(captured[0]["state"])
    assert state["previous_tasks"] == ["Afternoon notebook", "Morning notebook"]
    assert "Play blitz online" in state["observed_off_task_browser_visit"]["tab_titles"]
    monkeypatch.setattr(focus_judge, "packaged", lambda request: False)
    assert not policy(store, task)["domains"] and not policy(store, task)["pending"]
    save_domains(store, task, ["chess.com"], [])
    assert policy(store, task)["domains"] == ["chess.com"]


def test_explicit_unblock_survives_subdomain_variations_in_future_reports(tmp_path, monkeypatch):
    store = PlannerStore(tmp_path)
    source = history(store, "Review prior draft", [visit("stream.example")])
    target = open_task(store, "Draft chapter two", [source])
    monkeypatch.setattr(focus_judge, "packaged", lambda request: True)
    save_domains(store, target, [], policy(store, target)["domains"])
    store.save_report(source, {"intervals": [visit("clips.stream.example")]})
    assert policy(store, target)["domains"] == []
    # Explicitly choosing to block just this subdomain still works after unblocking the parent.
    save_domains(store, target, ["clips.stream.example"], [])
    assert policy(store, target)["domains"] == ["clips.stream.example"]


def test_saving_an_unchecked_pending_site_remains_effective_after_kev_answers(tmp_path, monkeypatch):
    store = PlannerStore(tmp_path)
    source = history(store, "Write test script", [visit("video.example", title="Full match replay")])
    target = open_task(store, "Debug another test", [source])
    pending = policy(store, target)
    assert "video.example" in pending["pending"]
    save_domains(store, target, ["shop.example"], pending["domains"], pending=pending["pending"])
    monkeypatch.setattr(focus_judge, "packaged", lambda req: True)
    updated = policy(store, target)
    assert updated["domains"] == ["shop.example"] and updated["removed"] == ["video.example"]
