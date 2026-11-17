"""Load an isolated, editable synthetic planner; DEMO's personal database lives elsewhere."""

from contextlib import closing
import json
from pathlib import Path
import sqlite3
from shutil import rmtree
from tempfile import mkdtemp
import weakref
import time

import portalocker

import activity
from examples.estimation.activity_fixture import replay_task_at
from examples.estimation.histories import DATASET, DEMO_DATASET, sample_root, task_inputs, valid_id
from planner_store import PlannerStore


def history_choices():
    """Return stable IDs and human-readable names for the packaged histories."""
    choices = {}
    for path in sorted((DEMO_DATASET / "histories").glob("*.json")):
        history = json.loads(path.read_text(encoding="utf-8"))
        if history["id"] != path.stem or not valid_id(path.stem):
            raise ValueError(f"Invalid history ID in {path.name}")
        choices[path.stem] = history["name"]
    return choices


def sample_task_choices(history_id):
    directory = sample_root(history_id, DATASET)
    history = json.loads((directory / "histories" / f"{history_id}.json").read_text(encoding="utf-8"))
    return {row["id"]: row for row in history["holdouts"]}


def add_sample_task(store, history_id, sample_id, day):
    sample = sample_task_choices(history_id)[sample_id]
    fields, attachments = task_inputs(sample, history_id, sample_root(history_id, DATASET))
    fields["day"] = day
    return store.create_task(**fields, attachments=attachments)


def _refresh_sample_activity(store, history, directory):
    """Update untouched sample reports in place when their packaged watcher DB changes."""
    fingerprint = history.get("activitywatch_sha256")
    if not fingerprint or store.get_setting("dev_sample_activity_sha256") == fingerprint:
        return
    by_key = {row["id"]: row for row in history["tasks"]}
    exclusions = store.get_setting("exclusions", [])
    for task in store.completed_tasks():
        previous = task.get("report") or {}
        if previous.get("classification_source") != "synthetic_annotations":
            continue
        ref = store.get_setting(f"dev_activity_{task['id']}")
        if not ref or ref.get("history") != history["id"] or ref["task"] not in by_key:
            continue
        updated = replay_task_at(history, by_key[ref["task"]], ref["sessions"], store.periods(task["id"]),
                                 directory=directory, exclusions=exclusions, task_context=task)
        # A concurrent user review, model report or time adjustment wins.
        with store._db(write=True) as db:
            db.execute("""UPDATE tasks SET report=? WHERE id=? AND json(report)=json(?)
                          AND NOT EXISTS (SELECT 1 FROM report_jobs WHERE task_id=tasks.id)
                          AND NOT EXISTS (SELECT 1 FROM activity_feedback WHERE task_id=tasks.id)""",
                       (json.dumps(updated, allow_nan=False), task["id"], json.dumps(previous)))
    store.set_setting("dev_sample_activity_sha256", fingerprint)


def dev_activity_report(store, task, *, examples=(), overrides=None):
    """Replay only imported device events. Manually timed work is unobserved in Dev."""
    periods = store.periods(task["id"])
    ref = store.get_setting(f"dev_activity_{task['id']}")
    if ref:
        directory = sample_root(ref["history"], DATASET)
        source = directory / "histories" / (ref["history"] + ".json")
        history = json.loads(source.read_text(encoding="utf-8"))
        for row in [*history["tasks"], *history.get("open_tasks", [])]:
            if row["id"] == ref["task"]:
                return replay_task_at(history, row, ref["sessions"], periods, directory=directory,
                                      exclusions=store.get_setting("exclusions", []), task_context=task,
                                      use_model=bool(history.get("activitywatch_sha256")),
                                      examples=examples, overrides=overrides)
    return activity.build_report(task, periods, capture=False)


def _read_generation(directory):
    # Queries use short-lived SQLite/Qdrant connections. Keep a shared lock for
    # the lifetime of the *owner*, including a background worker between calls.
    handle = (directory / "workspace.lock").open("a+b")
    try:
        portalocker.lock(handle, portalocker.LOCK_SH)
        store = PlannerStore(directory)
        weakref.finalize(store, handle.close)
        return store
    except BaseException:
        handle.close()
        raise


def _reap_retired(profile, selected):
    # The caller holds the registry write lock. New opens cannot take an old
    # generation between this exclusive-lock check and removal. Close our own
    # lockfile before deleting the tree, which Windows requires.
    for candidate in profile.glob("generation-*"):
        if candidate in selected or not candidate.is_dir():
            continue
        try:
            with (candidate / "workspace.lock").open("a+b") as handle:
                portalocker.lock(handle, portalocker.LOCK_EX | portalocker.LOCK_NB)
            rmtree(candidate)
        except (portalocker.exceptions.LockException, OSError):
            # Another tab/process/worker still owns this generation. Retry on a
            # later open. A failed cleanup never prevents switching workspaces.
            continue


def open_dev_workspace(data_root, history_id, *, reset=False, replay=False):
    """Atomically select one editable generation of this scenario for new sessions."""
    directory = sample_root(history_id, DATASET)
    source = directory / "histories" / f"{history_id}.json"
    if not source.is_file():
        raise ValueError(f"Unknown example history {history_id}")
    dev_root = Path(data_root) / "dev"
    if replay:
        dev_root = dev_root / "replays"
    profile = dev_root / history_id
    profile.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(dev_root / "profiles.sqlite3", timeout=30)) as registry:
        registry.execute("BEGIN IMMEDIATE")
        registry.execute("CREATE TABLE IF NOT EXISTS profiles (id TEXT PRIMARY KEY, generation TEXT)")
        if "generation" not in {row[1] for row in registry.execute("PRAGMA table_info(profiles)")}:
            # A NULL generation is the original stable-directory layout. Keep
            # any existing edits in place until the user explicitly resets.
            registry.execute("ALTER TABLE profiles ADD COLUMN generation TEXT")
        selected = registry.execute("SELECT generation FROM profiles WHERE id=?", (history_id,)).fetchone()
        current = profile / selected[0] if selected and selected[0] else profile
        previous = current
        history = json.loads(source.read_text(encoding="utf-8"))
        if reset or not (current / "planner.sqlite3").is_file():
            current = Path(mkdtemp(dir=profile, prefix="generation-"))
            try:
                fresh = PlannerStore(current)
                import dev_schedule
                dev_schedule.initialize(fresh, history_id, directory, replay=replay, now=time.time())
                fresh.set_setting("dev_sample_activity_sha256", history.get("activitywatch_sha256"))
            except BaseException:
                rmtree(current, ignore_errors=True)
                raise
            registry.execute("INSERT OR REPLACE INTO profiles VALUES (?, ?)", (history_id, current.name))
        elif selected is None:
            registry.execute("INSERT INTO profiles VALUES (?, NULL)", (history_id,))
        # Once published, this generation keeps its own SQLite and vector
        # paths forever. A late model reply from the previous generation can
        # finish only in its original workspace, even in another process.
        store = _read_generation(current)
        _refresh_sample_activity(store, history, directory)
        if replay:
            # Use a fresh read on every tick. Workers and other tabs share the same virtual clock.
            path = store.path
            def replay_clock():
                with closing(sqlite3.connect(path, timeout=15)) as clock_db:
                    row = clock_db.execute("SELECT value FROM settings WHERE key='dev_calendar'").fetchone()
                    calendar = json.loads(row[0])
                    return calendar.get("clock", calendar["opened_at"])
            store.clock = replay_clock
        # Protect the existing selection until the commit succeeds, as well
        # as the incoming generation. Later opens collect unowned generations.
        _reap_retired(profile, {current, previous})
        registry.commit()
    return store
