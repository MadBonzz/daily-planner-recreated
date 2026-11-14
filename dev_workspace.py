"""Load an isolated, editable synthetic planner; DEMO's personal database lives elsewhere."""

from contextlib import closing
from datetime import date
import json
from pathlib import Path
import sqlite3
from shutil import rmtree
from tempfile import mkdtemp
import weakref
import time

import portalocker

import activity
from examples.estimation.activity_fixture import replay_history, replay_task_at
from examples.estimation.run import DATASET, task_inputs, valid_id
from planner_store import PlannerStore


def history_choices():
    """Return stable IDs and human-readable names for the packaged histories."""
    choices = {}
    for path in sorted((DATASET / "histories").glob("*.json")):
        history = json.loads(path.read_text(encoding="utf-8"))
        if history["id"] != path.stem or not valid_id(path.stem):
            raise ValueError(f"Invalid history ID in {path.name}")
        choices[path.stem] = history["name"]
    return choices


def _import_task(store, history_id, directory, row, sessions, *, status, now, report=None):
    """Apply the same Start/Pause/Stop transitions used by the UI, with the fixture clock."""
    stamp = sessions[0][0][0] if sessions else now
    timed = PlannerStore(store.data_dir, clock=lambda: stamp)
    fields, files = task_inputs(row, history_id, directory)
    if status != "completed":
        fields["day"] = date.fromtimestamp(now).isoformat()
    # Old completed tasks can legitimately have no stored forecast. Open tasks
    # without an authored forecast are estimated by the normal background worker.
    prediction = row.get("prediction", {} if status == "completed" else None)
    task = timed.create_task(**fields, attachments=files, prediction=prediction)
    task_id = task["id"]
    if row.get("activity"):
        # This identifies public fixture observations, without copying raw
        # unfiltered watcher content into the planner database.
        timed.set_setting(f"dev_activity_{task_id}", {
            "history": history_id, "task": row["id"], "sessions": sessions,
        })

    for i, periods in enumerate(sessions):
        for j, (start, end) in enumerate(periods):
            last = i == len(sessions) - 1 and j == len(periods) - 1
            stamp = start
            timed.transition(task_id, "start" if j == 0 else "resume")
            if last and status == "running":
                return
            stamp = end
            if (last and status == "paused") or j + 1 < len(periods):
                timed.transition(task_id, "pause")
            else:
                timed.transition(task_id, "stop")
                if last and status == "completed":
                    timed.transition(task_id, "finish", completion_note=row.get("completion_note", ""),
                                     report=report, refresh_estimates=False)
                else:
                    timed.transition(task_id, "continue_later")


def seed_completed(store, history, directory):
    """Import prior work from a single synthetic history with zero model or network calls."""
    store.set_capture(False)
    replayed = replay_history(history, directory=directory)
    for row in sorted(history["tasks"], key=lambda task: task["day"]):
        entry = replayed[row["id"]]
        sessions = entry["sessions"]
        report = entry["report"] if row.get("activity") else None
        _import_task(store, history["id"], directory, row, sessions,
                     status="completed", now=sessions[-1][-1][1], report=report)


def dev_activity_report(store, task):
    """Replay only imported device events. Manually timed work is unobserved in Dev."""
    periods = store.periods(task["id"])
    ref = store.get_setting(f"dev_activity_{task['id']}")
    if ref:
        source = DATASET / "histories" / (ref["history"] + ".json")
        history = json.loads(source.read_text(encoding="utf-8"))
        for row in [*history["tasks"], *history.get("open_tasks", [])]:
            if row["id"] == ref["task"]:
                return replay_task_at(ref["history"], row, ref["sessions"], periods,
                                      exclusions=store.get_setting("exclusions", []))
    return activity.build_report(task, periods, capture=False)


def _seed_open(store, history, now):
    rows = history.get("open_tasks", [])
    if sum(row["status"] == "running" for row in rows) > 1:
        raise ValueError("A profile can contain only one running task.")
    # Allocate time serially; past work on two paused tasks must not overlap.
    # The current session of the running task ends at import time and stays open.
    end = now
    blocks = []
    ordered = sorted(rows, key=lambda task: task["status"] == "running")
    for row in reversed(ordered):
        status = row["status"]
        if status not in {"ready", "running", "paused"}:
            raise ValueError(f"Invalid open task state: {status}")
        sessions = []
        if row.get("sessions_minutes"):
            layout = replay_history({"id": history["id"], "tasks": [row]})[row["id"]]["sessions"]
            offset = end - layout[-1][-1][1]
            sessions = [[(start + offset, stop + offset) for start, stop in group] for group in layout]
        elif status != "ready":
            sessions = [[(end, end)]]
        blocks.append((row, sessions))
        if sessions:
            end = sessions[0][0][0] - 900
    for row, sessions in reversed(blocks):
        _import_task(store, history["id"], DATASET, row, sessions, status=row["status"], now=now)


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


def open_dev_workspace(data_root, history_id, *, reset=False):
    """Atomically select one editable generation of this scenario for new sessions."""
    if not valid_id(history_id):
        raise ValueError("Invalid history ID")
    source = DATASET / "histories" / f"{history_id}.json"
    if not source.is_file():
        raise ValueError(f"Unknown example history {history_id}")
    dev_root = Path(data_root) / "dev"
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
        if reset or not (current / "planner.sqlite3").is_file():
            history = json.loads(source.read_text(encoding="utf-8"))
            current = Path(mkdtemp(dir=profile, prefix="generation-"))
            try:
                fresh = PlannerStore(current)
                seed_completed(fresh, history, DATASET)
                _seed_open(fresh, history, time.time())
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
        # Protect the existing selection until the commit succeeds, as well
        # as the incoming generation. Later opens collect unowned generations.
        _reap_retired(profile, {current, previous})
        registry.commit()
    return store
