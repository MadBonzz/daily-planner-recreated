"""Local tasks, work sessions, and opt-in activity capture windows."""

from contextlib import contextmanager
from datetime import date, timedelta
import json
import math
from pathlib import Path
import sqlite3
import time
from uuid import uuid4


_SCHEMA = """
CREATE TABLE IF NOT EXISTS tasks (
    id TEXT PRIMARY KEY,
    day TEXT NOT NULL,
    title TEXT NOT NULL,
    description TEXT NOT NULL,
    category TEXT NOT NULL,
    details TEXT NOT NULL,
    pdf_text TEXT NOT NULL,
    estimate_minutes REAL NOT NULL CHECK (estimate_minutes > 0),
    prediction TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'ready'
        CHECK (status IN ('ready','running','paused','awaiting_finish','completed')),
    created_at REAL NOT NULL,
    completed_at REAL,
    actual_seconds REAL,
    report TEXT
);
CREATE INDEX IF NOT EXISTS tasks_by_day ON tasks(day);
CREATE TABLE IF NOT EXISTS prediction_jobs (
    task_id TEXT PRIMARY KEY REFERENCES tasks ON DELETE CASCADE,
    token TEXT,
    lease_end REAL
);
CREATE TABLE IF NOT EXISTS prediction_history (
    task_id TEXT NOT NULL REFERENCES prediction_jobs ON DELETE CASCADE,
    source_id TEXT NOT NULL REFERENCES tasks ON DELETE CASCADE,
    PRIMARY KEY(task_id, source_id)
);
CREATE INDEX IF NOT EXISTS prediction_history_source ON prediction_history(source_id);
CREATE TABLE IF NOT EXISTS sessions (
    id INTEGER PRIMARY KEY,
    task_id TEXT NOT NULL REFERENCES tasks ON DELETE CASCADE,
    started_at REAL NOT NULL,
    stopped_at REAL
);
CREATE UNIQUE INDEX IF NOT EXISTS open_session_per_task ON sessions(task_id)
    WHERE stopped_at IS NULL;
CREATE TABLE IF NOT EXISTS periods (
    id INTEGER PRIMARY KEY,
    session_id INTEGER NOT NULL REFERENCES sessions ON DELETE CASCADE,
    start REAL NOT NULL,
    end REAL CHECK (end >= start)
);
CREATE UNIQUE INDEX IF NOT EXISTS one_running_period ON periods((1)) WHERE end IS NULL;
CREATE TABLE IF NOT EXISTS capture (
    id INTEGER PRIMARY KEY,
    period_id INTEGER NOT NULL REFERENCES periods ON DELETE CASCADE,
    start REAL NOT NULL,
    end REAL CHECK (end >= start)
);
CREATE UNIQUE INDEX IF NOT EXISTS one_capture_span ON capture((1)) WHERE end IS NULL;
CREATE TABLE IF NOT EXISTS attachments (
    id TEXT PRIMARY KEY,
    task_id TEXT NOT NULL REFERENCES tasks ON DELETE CASCADE,
    name TEXT NOT NULL,
    pdf BLOB NOT NULL
);
CREATE INDEX IF NOT EXISTS task_attachments ON attachments(task_id);
CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT NOT NULL);
"""

_TASKS = """
SELECT t.*, clock.started_at,
       COALESCE(t.actual_seconds, clock.total, 0) AS measured
FROM tasks AS t
LEFT JOIN (
    SELECT s.task_id, SUM(MAX(0, COALESCE(p.end, ?) - p.start)) AS total,
           MAX(CASE WHEN p.end IS NULL THEN p.start END) AS started_at
    FROM sessions AS s JOIN periods AS p ON p.session_id = s.id
    GROUP BY s.task_id
) AS clock ON clock.task_id = t.id
"""


def _day(value):
    return date.fromisoformat(str(value)).isoformat()


def _save_setting(db, key, value):
    db.execute(
        "INSERT INTO settings VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        (key, json.dumps(value)),
    )


def _get_setting(db, key, default):
    row = db.execute("SELECT value FROM settings WHERE key=?", (key,)).fetchone()
    return json.loads(row[0]) if row else default


def _close_period(db, session_id, now):
    period = db.execute(
        "SELECT id FROM periods WHERE session_id=? AND end IS NULL", (session_id,)
    ).fetchone()
    if period is None:
        return
    db.execute("UPDATE capture SET end=MAX(start, ?) WHERE period_id=? AND end IS NULL",
               (now, period["id"]))
    db.execute("UPDATE periods SET end=MAX(start, ?) WHERE id=?", (now, period["id"]))


class PlannerStore:
    def __init__(self, data_dir, *, clock=time.time):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.data_dir / "planner.sqlite3"
        self.clock = clock
        with self._db() as db:
            db.executescript(_SCHEMA)

    @contextmanager
    def _db(self, *, write=False):
        db = sqlite3.connect(self.path, timeout=15, isolation_level=None)
        db.row_factory = sqlite3.Row
        try:
            db.execute("PRAGMA foreign_keys=ON")
            db.execute("PRAGMA secure_delete=ON")
            db.execute("BEGIN IMMEDIATE" if write else "BEGIN")
            yield db
            db.commit()
        finally:
            db.close()  # Rolls back any unfinished transaction.

    def _tasks(self, db, where="", args=()):
        rows = db.execute(_TASKS + " " + where + " ORDER BY t.created_at, t.id",
                          (self.clock(), *args))
        tasks = []
        for row in rows:
            task = dict(row)
            task["actual_seconds"] = task.pop("measured")
            task["prediction"] = json.loads(task["prediction"])
            task["report"] = json.loads(task["report"]) if task["report"] else None
            task["attachments"] = [dict(a) for a in db.execute(
                "SELECT id, name FROM attachments WHERE task_id=? ORDER BY rowid", (task["id"],)
            )]
            tasks.append(task)
        return tasks

    def _task(self, db, task_id):
        tasks = self._tasks(db, "WHERE t.id=?", (task_id,))
        if not tasks:
            raise ValueError("This task no longer exists.")
        return tasks[0]

    def list_tasks(self, day=None):
        with self._db() as db:
            if day is None:
                return self._tasks(db)
            return self._tasks(db, "WHERE t.day=?", (_day(day),))

    def get_task(self, task_id):
        with self._db() as db:
            return self._task(db, task_id)

    def completed_tasks(self):
        with self._db() as db:
            return self._tasks(db, "WHERE t.status='completed'")

    def _first_with_status(self, status):
        with self._db() as db:
            tasks = self._tasks(db, "WHERE t.status=?", (status,))
        return tasks[0] if tasks else None

    def running_task(self):
        return self._first_with_status("running")

    def pending_task(self):
        return self._first_with_status("awaiting_finish")

    def create_task(self, *, title, description, estimate_minutes, day, prediction=None,
                    category="", details="", attachments=(), pdf_text=""):
        title = title.strip()
        description = description.strip()
        if not title or not description:
            raise ValueError("Enter a task name and description.")
        estimate_minutes = float(estimate_minutes)
        if not math.isfinite(estimate_minutes) or estimate_minutes <= 0:
            raise ValueError("The estimate must be a positive number of minutes.")
        task_id = str(uuid4())
        queue_prediction = prediction is None
        if queue_prediction:
            prediction = {"status": "pending", "basis": "Pending", "evidence": []}
        with self._db(write=True) as db:
            db.execute("""
                INSERT INTO tasks(id, day, title, description, category, details, pdf_text,
                                  estimate_minutes, prediction, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (task_id, _day(day), title, description, category or "", details or "",
                  pdf_text or "", estimate_minutes, json.dumps(prediction, allow_nan=False),
                  self.clock()))
            if queue_prediction:
                db.execute("INSERT INTO prediction_jobs(task_id) VALUES (?)", (task_id,))
                # Membership is frozen in the creation transaction. A task finished
                # while this estimate is running must never affect its first forecast.
                db.execute("""INSERT INTO prediction_history(task_id, source_id)
                              SELECT ?, id FROM tasks WHERE status='completed' AND actual_seconds > 0""",
                           (task_id,))
            for name, content in attachments:
                if not isinstance(content, bytes) or b"%PDF-" not in content[:1024]:
                    raise ValueError("Only PDF attachments are supported.")
                db.execute("INSERT INTO attachments VALUES (?, ?, ?, ?)",
                           (str(uuid4()), task_id, Path(name.replace("\\", "/")).name, content))
            return self._task(db, task_id)

    def has_pending_predictions(self):
        """Pending work includes currently leased jobs and is independent of the selected day."""
        with self._db() as db:
            return db.execute("SELECT 1 FROM prediction_jobs LIMIT 1").fetchone() is not None

    def claim_prediction(self, *, lease_seconds=300):
        """Lease one eligible estimate. Its source tasks were already finished when it was created."""
        if not math.isfinite(lease_seconds) or lease_seconds <= 0:
            raise ValueError("The estimate lease must be a positive number of seconds.")
        with self._db(write=True) as db:
            now = self.clock()
            row = db.execute("""
                SELECT t.id FROM prediction_jobs j JOIN tasks t ON t.id=j.task_id
                WHERE j.token IS NULL OR j.lease_end<=?
                ORDER BY t.created_at, t.rowid LIMIT 1
            """, (now,)).fetchone()
            if row is None:
                return None
            task_id = row["id"]
            token = str(uuid4())
            db.execute("UPDATE prediction_jobs SET token=?, lease_end=? WHERE task_id=?",
                       (token, now + lease_seconds, task_id))
            task = self._task(db, task_id)
            history = self._tasks(db, """WHERE t.id IN (
                SELECT source_id FROM prediction_history WHERE task_id=?
            )""", (task_id,))
            return {"task": task, "history": history, "token": token}

    def renew_prediction(self, task_id, token, *, lease_seconds=300):
        if not math.isfinite(lease_seconds) or lease_seconds <= 0:
            raise ValueError("The estimate lease must be a positive number of seconds.")
        with self._db(write=True) as db:
            now = self.clock()
            return bool(db.execute("""
                UPDATE prediction_jobs SET lease_end=? WHERE task_id=? AND token=? AND lease_end>?
            """, (now + lease_seconds, task_id, token, now)).rowcount)

    def resolve_prediction(self, task_id, token, prediction, *, pdf_text=None):
        """Publish the original forecast once, even if its timer has since finished."""
        with self._db(write=True) as db:
            deleted = db.execute("DELETE FROM prediction_jobs WHERE task_id=? AND token=? AND lease_end>?",
                                 (task_id, token, self.clock()))
            if not deleted.rowcount:
                return False
            db.execute("UPDATE tasks SET prediction=?, pdf_text=COALESCE(?,pdf_text) WHERE id=?",
                       (json.dumps(prediction, allow_nan=False), pdf_text, task_id))
            return True

    def transition(self, task_id, action, *, ended_at=None):
        """Apply a timer action and return the committed state. Starting another task pauses the first."""
        with self._db(write=True) as db:
            task = self._task(db, task_id)
            state = task["status"]
            now = self.clock()
            session = db.execute("SELECT id FROM sessions WHERE task_id=? AND stopped_at IS NULL",
                                 (task_id,)).fetchone()
            repeated = {"start": "running", "resume": "running", "pause": "paused",
                        "stop": "awaiting_finish", "finish": "completed", "continue_later": "ready"}
            if ended_at is not None and action != "stop":
                raise ValueError("An end-time correction is only available when stopping a task.")
            if repeated.get(action) == state:
                return task
            if ended_at is not None:
                end = float(ended_at)
                if state != "running" or not task["started_at"] <= end <= now:
                    raise ValueError("Choose an end time between the current timer start and now.")
                now = end
                period = db.execute("SELECT id FROM periods WHERE session_id=? AND end IS NULL",
                                    (session["id"],)).fetchone()
                db.execute("DELETE FROM capture WHERE period_id=? AND start>=?", (period["id"], end))
                db.execute("UPDATE capture SET end=? WHERE period_id=? AND end>?", (end, period["id"], end))
            if action in ("start", "resume") and state in ("ready", "paused"):
                active = db.execute("""
                    SELECT s.task_id, s.id FROM sessions s JOIN periods p ON p.session_id=s.id
                    WHERE p.end IS NULL
                """).fetchone()
                if active:
                    _close_period(db, active["id"], now)
                    db.execute("UPDATE tasks SET status='paused' WHERE id=?", (active["task_id"],))
                sid = session["id"] if session else db.execute(
                    "INSERT INTO sessions(task_id, started_at) VALUES (?,?)", (task_id, now)
                ).lastrowid
                pid = db.execute("INSERT INTO periods(session_id,start) VALUES (?,?)",
                                 (sid, now)).lastrowid
                if _get_setting(db, "capture_enabled", True):
                    db.execute("INSERT INTO capture(period_id,start) VALUES (?,?)", (pid, now))
                state = "running"
            elif action == "pause" and state == "running":
                _close_period(db, session["id"], now)
                state = "paused"
            elif action == "stop" and state in ("paused", "running"):
                _close_period(db, session["id"], now)
                db.execute("UPDATE sessions SET stopped_at=MAX(started_at,?) WHERE id=?",
                           (now, session["id"]))
                state = "awaiting_finish"
            elif action == "continue_later" and state == "awaiting_finish":
                state = "ready"
            elif action == "finish" and state == "awaiting_finish":
                db.execute("UPDATE tasks SET completed_at=?, actual_seconds=? WHERE id=?",
                           (now, task["actual_seconds"], task_id))
                state = "completed"
            else:
                raise ValueError(f"Cannot {action.replace('_', ' ')} a {state} task.")
            db.execute("UPDATE tasks SET status=? WHERE id=?", (state, task_id))
            return self._task(db, task_id)

    def periods(self, task_id):
        with self._db() as db:
            rows = db.execute("""
                SELECT p.start, MAX(p.start, COALESCE(p.end, ?)) FROM periods p
                JOIN sessions s ON s.id=p.session_id WHERE s.task_id=? ORDER BY p.start, p.id
            """, (self.clock(), task_id))
            return [tuple(r) for r in rows]

    def capture_periods(self, task_id):
        with self._db() as db:
            rows = db.execute("""
                SELECT c.start, MAX(c.start, COALESCE(c.end, ?)) FROM capture c
                JOIN periods p ON p.id=c.period_id JOIN sessions s ON s.id=p.session_id
                WHERE s.task_id=? ORDER BY c.start, c.id
            """, (self.clock(), task_id))
            return [tuple(r) for r in rows]

    def disabled_periods(self, task_id):
        """Privacy gaps within timed work. None marks an ongoing capture pause."""
        with self._db() as db:
            rows = db.execute("""
                SELECT p.id, p.start, p.end, c.start AS capture_start, c.end AS capture_end
                FROM sessions s JOIN periods p ON p.session_id=s.id
                LEFT JOIN capture c ON c.period_id=p.id
                WHERE s.task_id=? ORDER BY p.start, p.id, c.start, c.id
            """, (task_id,))
            gaps = []
            previous_id = None
            cursor = end = None
            for row in rows:
                if previous_id != row["id"]:
                    if cursor is not None and (end is None or cursor < end):
                        gaps.append((cursor, end))
                    previous_id = row["id"]
                    cursor, end = row["start"], row["end"]
                if row["capture_start"] is not None:
                    if cursor is not None and cursor < row["capture_start"]:
                        gaps.append((cursor, row["capture_start"]))
                    cursor = row["capture_end"]
            if cursor is not None and (end is None or cursor < end):
                gaps.append((cursor, end))
            return gaps

    def set_capture(self, enabled):
        with self._db(write=True) as db:
            enabled = bool(enabled)
            if enabled == _get_setting(db, "capture_enabled", True):
                return
            now = self.clock()
            if enabled:
                db.execute("""
                    INSERT INTO capture(period_id,start)
                    SELECT id, MAX(start, ?) FROM periods WHERE end IS NULL
                """, (now,))
            else:
                db.execute("UPDATE capture SET end=MAX(start, ?) WHERE end IS NULL", (now,))
            _save_setting(db, "capture_enabled", enabled)

    def save_report(self, task_id, report):
        with self._db(write=True) as db:
            changed = db.execute("UPDATE tasks SET report=? WHERE id=? AND status='completed'",
                                 (json.dumps(report, allow_nan=False), task_id))
            if not changed.rowcount:
                raise ValueError("Finish the task before saving its activity review.")

    def attachment_bytes(self, attachment_id):
        with self._db() as db:
            row = db.execute("SELECT pdf FROM attachments WHERE id=?", (attachment_id,)).fetchone()
            if row is None:
                raise ValueError("This attachment no longer exists.")
            return row[0]

    def reschedule(self, task_id, day):
        with self._db(write=True) as db:
            result = db.execute("UPDATE tasks SET day=? WHERE id=? AND status!='completed'",
                                (_day(day), task_id))
            if not result.rowcount:
                raise ValueError("Only unfinished tasks can be rescheduled.")
            return self._task(db, task_id)

    def delete_task(self, task_id):
        with self._db(write=True) as db:
            # A running model call may already hold the deleted task in memory.
            # Revoke that worker before removing the source, so its response cannot
            # restore the source title or description in a different task's forecast.
            db.execute("""UPDATE prediction_jobs SET token=NULL, lease_end=NULL
                          WHERE task_id IN (SELECT task_id FROM prediction_history WHERE source_id=?)""",
                       (task_id,))
            db.execute("DELETE FROM tasks WHERE id=?", (task_id,))
            # A stored explanation can name a comparison anywhere in its text.
            # Retain the displayed numbers while erasing that explanation and its evidence.
            for row in db.execute("SELECT id, prediction FROM tasks").fetchall():
                prediction = json.loads(row["prediction"])
                if any(str(item.get("task_id")) == task_id for item in prediction.get("evidence", [])):
                    prediction["evidence"] = []
                    prediction["explanation"] = "The supporting history included a deleted task."
                    db.execute("UPDATE tasks SET prediction=? WHERE id=?",
                               (json.dumps(prediction), row["id"]))

    def clear_data(self):
        with self._db(write=True) as db:
            db.execute("DELETE FROM tasks")
            db.execute("DELETE FROM settings")

    def get_setting(self, key, default=None):
        with self._db() as db:
            return _get_setting(db, key, default)

    def set_setting(self, key, value):
        if key == "capture_enabled":
            self.set_capture(value)
            return
        with self._db(write=True) as db:
            _save_setting(db, key, value)

    def seed_demo(self):
        """Seed an explicitly separate example workspace once."""
        if self.data_dir.name != "demo":
            raise ValueError("Example history requires a separate demo workspace.")
        examples = [
            ("Work through the convergence proof", "Derive the equations in a math-heavy paper.", "Reading", 40, 60),
            ("Reproduce a paper's main derivation", "Work through a theorem and write out the proof steps.", "Reading", 60, 90),
            ("Check the assumptions of a proof", "Read each lemma and reproduce the mathematical derivation.", "Reading", 30, 48),
            ("Review an empirical evaluation", "Skim the plots and summarize the experimental findings.", "Reading", 40, 32),
            ("Implement a CUDA reduction", "Write a block reduction kernel and check correctness.", "Programming", 60, 84),
        ]
        with self._db(write=True) as db:
            if _get_setting(db, "example_history_seeded", False):
                return
            for index, (title, desc, cat, estimate, actual) in enumerate(examples):
                task_id = str(uuid4())
                day = date.today() - timedelta(days=len(examples) - index)
                end = self.clock() - (len(examples) - index) * 86400
                start = end - actual * 60
                prediction = dict(point_min=estimate, low_min=estimate/2, high_min=estimate*1.5,
                                  confidence="low", basis="Example", explanation="Staged example budget.", evidence=[])
                db.execute("""
                    INSERT INTO tasks(id,day,title,description,category,details,pdf_text,estimate_minutes,prediction,
                                      status,created_at,completed_at,actual_seconds)
                    VALUES (?,?,?,?,?,'','',?,?,'completed',?,?,?)
                """, (task_id, day.isoformat(), title, desc, cat, estimate,
                      json.dumps(prediction), start, end, actual * 60))
                sid = db.execute("INSERT INTO sessions(task_id,started_at,stopped_at) VALUES (?,?,?)",
                                 (task_id, start, end)).lastrowid
                db.execute("INSERT INTO periods(session_id,start,end) VALUES (?,?,?)", (sid, start, end))
            _save_setting(db, "example_history_seeded", True)
