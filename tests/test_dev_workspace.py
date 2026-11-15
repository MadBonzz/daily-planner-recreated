import json
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from shutil import copyfile

import pytest

import activity
import dev_workspace as dev
from planner_store import PlannerStore


def example_history(history_id):
    row = {
        "id": "past", "day": "2026-06-01", "title": "Review the handout",
        "description": "Read the handout and compare its proofs.", "category": "Research",
        "details": "Read section 2.", "estimate_minutes": 10, "actual_minutes": 14,
        "sessions_minutes": [10, 4], "completion_note": "I needed more time on the proof.",
        "activity": [
            [{"minutes": 4, "app": "Reader", "title": "Introduction", "label": "on_task", "pause_after_minutes": 3},
             {"minutes": 6, "app": "Reader", "title": "Proof", "label": "on_task"}],
            [{"minutes": 4, "app": "Browser", "title": "Streaming", "label": "off_task"}],
        ],
        "pdf": "brief",
    }
    common = {k: row[k] for k in ("day", "description", "category", "details", "estimate_minutes")}
    open_tasks = [
        dict(common, id="upcoming", title="First seminar", fixed_duration=True,
             status="ready", sessions_minutes=[]),
        dict(common, id="paused", title="Midway review", status="paused", sessions_minutes=[8],
             activity=[[{"minutes": 3, "app": "Reader", "title": "Proof", "label": "on_task", "pause_after_minutes": 1},
                        {"minutes": 5, "app": "Editor", "title": "Draft", "label": "on_task"}]]),
        dict(common, id="running", title="Keep drafting", status="running", sessions_minutes=[3, 2]),
    ]
    return {"id": history_id, "name": f"History {history_id}", "tasks": [row], "open_tasks": open_tasks,
            "holdouts": [dict(common, id="holdout", title="Private scoring task", actual_minutes=999)]}


@pytest.fixture
def profiles(tmp_path, monkeypatch):
    fixture_dir = tmp_path / "fixtures"
    (fixture_dir / "histories").mkdir(parents=True)
    sample_pdf = next((dev.DATASET / "pdfs").rglob("*.pdf"))
    for name in ("sample-a", "sample-b"):
        path = fixture_dir / "histories" / (name + ".json")
        path.write_text(json.dumps(example_history(name)), encoding="utf-8")
        pdf_dir = fixture_dir / "pdfs" / name
        pdf_dir.mkdir(parents=True)
        copyfile(sample_pdf, pdf_dir / "brief.pdf")
    monkeypatch.setattr(dev, "DATASET", fixture_dir)
    monkeypatch.setattr(activity, "_get", lambda *a, **kw: pytest.fail("Dev accessed the real ActivityWatch server"))
    monkeypatch.setattr(activity, "chat_json", lambda *a, **kw: pytest.fail("Dev classified invented observations"))
    return tmp_path / "profiles"


def by_title(store):
    return {row["title"]: row for row in store.list_tasks()}


def test_dev_workspace_populates_editable_timer_and_evidence(profiles):
    assert dev.history_choices() == {"sample-a": "History sample-a", "sample-b": "History sample-b"}
    store = dev.open_dev_workspace(profiles, "sample-a")
    rows = by_title(store)
    assert len(rows) == 4 and "Private scoring task" not in rows
    past = rows["Review the handout"]
    assert past["completion_note"] == "I needed more time on the proof."
    assert past["prediction"] == {} and past["pdf_text"] and past["attachments"][0]["name"] == "brief.pdf"
    assert [end - start for start, end in store.periods(past["id"])] == [240, 360, 240]
    assert past["actual_seconds"] == 840
    assert past["report"]["totals"] == {"on_task": 600, "off_task": 240, "uncertain": 0}
    assert dev.dev_activity_report(store, past)["totals"] == past["report"]["totals"]
    assert store.get_setting("capture_enabled") is False
    assert {r["title"] for r in store.list_tasks(date.today())} == {"First seminar", "Midway review", "Keep drafting"}
    assert rows["First seminar"]["fixed_duration"] and rows["First seminar"]["actual_seconds"] == 0
    paused = rows["Midway review"]
    assert paused["status"] == "paused" and paused["actual_seconds"] == 480
    assert rows["Keep drafting"]["status"] == "running"
    assert 300 <= rows["Keep drafting"]["actual_seconds"] < 310
    # Continue with an artificial clock after loading. This part of the work
    # has no device observations and must not inherit the last authored window.
    now = store.clock() + 60
    timed = PlannerStore(store.data_dir, clock=lambda: now)
    timed.transition(paused["id"], "resume")
    now += 60
    timed.transition(paused["id"], "stop")
    report = dev.dev_activity_report(timed, paused)
    assert report["totals"] == {"on_task": 480, "off_task": 0, "uncertain": 60}
    timed.set_setting("exclusions", ["proof"])
    assert dev.dev_activity_report(timed, paused)["totals"]["uncertain"] == 240


def test_materialization_reset_and_clear_are_isolated(profiles):
    personal = PlannerStore(profiles / "live")
    personal.create_task(title="My real task", description="Private", day=date.today(), estimate_minutes=5)
    with ThreadPoolExecutor(2) as pool:
        stores = list(pool.map(lambda _: dev.open_dev_workspace(profiles, "sample-a"), range(2)))
    assert {r["id"] for r in stores[0].list_tasks()} == {r["id"] for r in stores[1].list_tasks()}
    other = dev.open_dev_workspace(profiles, "sample-b")
    other_ids = {r["id"] for r in other.list_tasks()}
    deleted = stores[0].completed_tasks()[0]
    stores[0].delete_task(deleted["id"])
    assert not dev.open_dev_workspace(profiles, "sample-a").completed_tasks()
    stores[0].clear_data()
    assert dev.open_dev_workspace(profiles, "sample-a").list_tasks() == []
    reset = dev.open_dev_workspace(profiles, "sample-a", reset=True)
    assert len(reset.list_tasks()) == 4 and reset.completed_tasks()[0]["id"] != deleted["id"]
    assert [r["title"] for r in personal.list_tasks()] == ["My real task"]
    assert {r["id"] for r in other.list_tasks()} == other_ids
    with pytest.raises(ValueError, match="Invalid history ID"):
        dev.open_dev_workspace(profiles, "../live", reset=True)


def test_failed_reset_keeps_the_previous_profile(profiles):
    store = dev.open_dev_workspace(profiles, "sample-a")
    original_ids = {r["id"] for r in store.list_tasks()}
    original_generations = set(store.data_dir.parent.glob("generation-*"))
    source = dev.DATASET / "histories" / "sample-a.json"
    history = json.loads(source.read_text())
    history["tasks"][0]["sessions_minutes"] = [99]
    source.write_text(json.dumps(history))
    with pytest.raises(ValueError):
        dev.open_dev_workspace(profiles, "sample-a", reset=True)
    assert {r["id"] for r in store.list_tasks()} == original_ids
    assert set(store.data_dir.parent.glob("generation-*")) == original_generations


def test_reset_while_worker_and_database_are_in_use(profiles, monkeypatch):
    monkeypatch.setattr("retrieval.embed_task", lambda *args: [0.6, 0.8])
    import gc
    import sqlite3
    import weakref
    from contextlib import closing

    import estimator

    old = dev.open_dev_workspace(profiles, "sample-a")
    old_dir = old.data_dir
    old.create_task(title="Discarded edit", description="Temporary", estimate_minutes=5,
                    day=date.today(), prediction={})
    job = old.claim_prediction()
    task_id = job["task"]["id"]
    # A held SQLite write transaction and real Qdrant directory lock reproduce
    # both Windows hazards. A reset must publish without touching either file.
    with closing(sqlite3.connect(old.path)) as writer, estimator._index(old_dir / "vectors") as index:
        writer.execute("BEGIN IMMEDIATE")
        fresh = dev.open_dev_workspace(profiles, "sample-a", reset=True)
        assert fresh.data_dir != old_dir and not (fresh.data_dir / "vectors").exists()
        assert "Discarded edit" not in by_title(fresh)
        assert len(fresh.list_tasks()) == 4 and job["task"]["id"] not in {r["id"] for r in fresh.list_tasks()}
        assert not fresh.resolve_prediction(task_id, job["token"], {"point_min": 999})
        assert not index.collection_exists(estimator.COLLECTION)
    # A slow reply and delayed vector write can only update their old generation.
    assert old.resolve_prediction(task_id, job["token"], {"point_min": 999})
    estimator.index_current_task(old_dir / "vectors", old, task_id)
    assert not (fresh.data_dir / "vectors").exists()
    assert fresh.claim_prediction()["task"]["id"] != task_id
    assert dev.open_dev_workspace(profiles, "sample-a").path == fresh.path
    assert old.path.is_file()
    owner = weakref.ref(old)
    del old
    gc.collect()
    assert owner() is None
    assert dev.open_dev_workspace(profiles, "sample-a").path == fresh.path
    assert not old_dir.exists()


def test_registry_upgrade_preserves_legacy_edits(profiles):
    import sqlite3
    from contextlib import closing
    dev_root = profiles / "dev"
    legacy = PlannerStore(dev_root / "sample-a")
    legacy.create_task(title="Existing edited sample", description="Saved before upgrade",
                       estimate_minutes=5, day=date.today(), prediction={})
    with closing(sqlite3.connect(dev_root / "profiles.sqlite3")) as reg:
        reg.execute("CREATE TABLE profiles (id TEXT PRIMARY KEY)")
        reg.execute("INSERT INTO profiles VALUES ('sample-a')")
        reg.commit()
    assert list(by_title(dev.open_dev_workspace(profiles, "sample-a"))) == ["Existing edited sample"]
    reset = dev.open_dev_workspace(profiles, "sample-a", reset=True)
    assert reset.path != legacy.path and "Existing edited sample" not in by_title(reset)
    assert dev.open_dev_workspace(profiles, "sample-a").path == reset.path
