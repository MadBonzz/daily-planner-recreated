"""Local tasks, work sessions, and opt-in activity capture windows."""

from contextlib import contextmanager
from datetime import date, datetime, time as local_time, timedelta
import json
import math
from pathlib import Path
import sqlite3
import time
from uuid import uuid4

from relevance_feedback import (
    FEEDBACK_SCHEMA, comparison_options, forecast_feedback, related_ids, relevance_examples, save_choices,
)
from task_forecasts import scheduled_prediction
from task_files import upload_info


DEFAULT_CATEGORIES = ("Work", "Personal", "Study", "Health", "Creative")


_SCHEMA = """
CREATE TABLE IF NOT EXISTS tasks (
    id TEXT PRIMARY KEY,
    day TEXT NOT NULL,
    title TEXT NOT NULL,
    edit_revision INTEGER NOT NULL DEFAULT 0,
    description TEXT NOT NULL,
    category TEXT NOT NULL,
    details TEXT NOT NULL,
    pdf_text TEXT NOT NULL,
    pdf_text_version INTEGER NOT NULL DEFAULT 0,
    estimate_minutes REAL NOT NULL CHECK (estimate_minutes > 0),
    prediction TEXT NOT NULL,
    creation_prediction TEXT,
    fixed_duration INTEGER NOT NULL DEFAULT 0,
    completion_note TEXT NOT NULL DEFAULT '',
    refresh_waiting INTEGER NOT NULL DEFAULT 0,
    archived INTEGER NOT NULL DEFAULT 0,
    status TEXT NOT NULL DEFAULT 'ready'
        CHECK (status IN ('ready','running','paused','awaiting_finish','completed')),
    created_at REAL NOT NULL,
    completed_at REAL,
    actual_seconds REAL,
    time_edited_at REAL,
    report TEXT
);
CREATE INDEX IF NOT EXISTS tasks_by_day ON tasks(day);
CREATE TABLE IF NOT EXISTS task_workspaces (
    task_id TEXT PRIMARY KEY REFERENCES tasks ON DELETE CASCADE,
    request_hash TEXT NOT NULL,
    eligible INTEGER,
    source TEXT NOT NULL,
    token TEXT NOT NULL UNIQUE,
    draft TEXT NOT NULL DEFAULT ''
);
CREATE TABLE IF NOT EXISTS workspace_messages (
    id INTEGER PRIMARY KEY,
    task_id TEXT NOT NULL REFERENCES tasks ON DELETE CASCADE,
    document_id TEXT NOT NULL,
    turn_id TEXT NOT NULL,
    role TEXT NOT NULL CHECK (role IN ('user','assistant')),
    content TEXT NOT NULL,
    model TEXT NOT NULL DEFAULT '',
    sources TEXT NOT NULL DEFAULT '[]',
    workspace_run TEXT,
    recorded_at REAL NOT NULL,
    UNIQUE(task_id, document_id, turn_id, role)
);
CREATE INDEX IF NOT EXISTS workspace_messages_by_task ON workspace_messages(task_id,id);
CREATE TABLE IF NOT EXISTS workspace_progress (
    task_id TEXT PRIMARY KEY REFERENCES tasks ON DELETE CASCADE,
    last_message INTEGER NOT NULL,
    applied_message INTEGER NOT NULL DEFAULT 0,
    snapshot TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS report_jobs (
    task_id TEXT PRIMARY KEY REFERENCES tasks ON DELETE CASCADE,
    generation TEXT NOT NULL,
    action TEXT NOT NULL CHECK (action IN ('build','review')),
    error TEXT
);
CREATE TABLE IF NOT EXISTS focus_domain_choices (
    task_id TEXT NOT NULL REFERENCES tasks ON DELETE CASCADE,
    domain TEXT NOT NULL,
    blocked INTEGER NOT NULL CHECK (blocked IN (0, 1)),
    PRIMARY KEY (task_id, domain)
);
CREATE TABLE IF NOT EXISTS activity_feedback (
    task_id TEXT NOT NULL REFERENCES tasks ON DELETE CASCADE,
    activity_key TEXT NOT NULL,
    label TEXT NOT NULL CHECK (label IN ('on_task','off_task','uncertain')),
    updated_at REAL NOT NULL,
    PRIMARY KEY (task_id, activity_key)
);
CREATE TABLE IF NOT EXISTS prediction_jobs (
    task_id TEXT PRIMARY KEY REFERENCES tasks ON DELETE CASCADE,
    token TEXT,
    lease_end REAL,
    workspace_only INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS prediction_history (
    task_id TEXT NOT NULL REFERENCES prediction_jobs ON DELETE CASCADE,
    source_id TEXT NOT NULL REFERENCES tasks ON DELETE CASCADE,
    snapshot TEXT,
    PRIMARY KEY(task_id, source_id)
);
CREATE INDEX IF NOT EXISTS prediction_history_source ON prediction_history(source_id);
CREATE TABLE IF NOT EXISTS replay_forecasts (
    id INTEGER PRIMARY KEY,
    task_id TEXT NOT NULL REFERENCES tasks ON DELETE CASCADE,
    at REAL NOT NULL,
    initial INTEGER NOT NULL,
    prediction TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS replay_forecasts_task ON replay_forecasts(task_id, at);
CREATE TABLE IF NOT EXISTS task_relations (
    left_id TEXT NOT NULL REFERENCES tasks ON DELETE CASCADE,
    right_id TEXT NOT NULL REFERENCES tasks ON DELETE CASCADE,
    related INTEGER CHECK (related IN (0, 1)),
    refresh_due INTEGER NOT NULL DEFAULT 0,
    token TEXT,
    lease_end REAL,
    retry_at REAL NOT NULL DEFAULT 0,
    failures INTEGER NOT NULL DEFAULT 0,
    CHECK (left_id < right_id),
    PRIMARY KEY(left_id, right_id)
);
CREATE INDEX IF NOT EXISTS relations_by_right ON task_relations(right_id);
CREATE INDEX IF NOT EXISTS relation_refreshes ON task_relations(refresh_due) WHERE refresh_due=1;
CREATE TABLE IF NOT EXISTS task_relation_sources (
    left_id TEXT NOT NULL,
    right_id TEXT NOT NULL,
    source_id TEXT NOT NULL REFERENCES tasks ON DELETE CASCADE,
    PRIMARY KEY (left_id,right_id,source_id),
    FOREIGN KEY(left_id,right_id) REFERENCES task_relations ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS relation_sources_by_task ON task_relation_sources(source_id);
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
    end REAL CHECK (end >= start),
    workspace_run TEXT
);
CREATE UNIQUE INDEX IF NOT EXISTS one_running_period ON periods((1)) WHERE end IS NULL;
CREATE TABLE IF NOT EXISTS capture (
    id INTEGER PRIMARY KEY,
    period_id INTEGER NOT NULL REFERENCES periods ON DELETE CASCADE,
    start REAL NOT NULL,
    end REAL CHECK (end >= start)
);
CREATE UNIQUE INDEX IF NOT EXISTS one_capture_span ON capture((1)) WHERE end IS NULL;
CREATE TABLE IF NOT EXISTS manual_work (
    id INTEGER PRIMARY KEY,
    task_id TEXT NOT NULL REFERENCES tasks ON DELETE CASCADE,
    entered_at REAL NOT NULL,
    seconds REAL NOT NULL CHECK (seconds > 0)
);
CREATE INDEX IF NOT EXISTS manual_work_by_task ON manual_work(task_id);
CREATE TABLE IF NOT EXISTS attachments (
    id TEXT PRIMARY KEY,
    task_id TEXT NOT NULL REFERENCES tasks ON DELETE CASCADE,
    name TEXT NOT NULL,
    pdf BLOB NOT NULL,
    kind TEXT NOT NULL DEFAULT 'pdf'
);
CREATE INDEX IF NOT EXISTS task_attachments ON attachments(task_id);
CREATE TABLE IF NOT EXISTS chat_task_changes (
    request_id TEXT PRIMARY KEY,
    task_id TEXT NOT NULL REFERENCES tasks ON DELETE CASCADE
);
CREATE TABLE IF NOT EXISTS vector_purges (task_id TEXT PRIMARY KEY);
CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS categories (
    label_key TEXT PRIMARY KEY,
    name TEXT NOT NULL
);
"""

_TASKS = """
SELECT t.*, clock.started_at,
       EXISTS(SELECT 1 FROM prediction_jobs j WHERE j.task_id=t.id) AS estimate_updating,
       EXISTS(SELECT 1 FROM prediction_jobs j JOIN workspace_progress wp ON wp.task_id=j.task_id
              WHERE j.task_id=t.id AND t.archived=0 AND t.status!='completed') AS workspace_progress_pending,
       (t.status!='completed' AND t.archived=0 AND t.fixed_duration=0 AND EXISTS(
          SELECT 1 FROM task_relations r WHERE (r.left_id=t.id OR r.right_id=t.id)
          AND (r.refresh_due=1 OR (r.related IS NOT 0 AND EXISTS(
              SELECT 1 FROM report_jobs j JOIN tasks source ON source.id=j.task_id
              WHERE j.error IS NULL AND source.actual_seconds>0
                AND (source.fixed_duration=0 OR EXISTS(SELECT 1 FROM relevance_choices f
                     WHERE f.target_id=t.id AND f.source_id=source.id AND f.related=1))
                AND j.task_id=CASE WHEN r.left_id=t.id THEN r.right_id ELSE r.left_id END
          )))
       )) AS awaiting_related_update,
       COALESCE(manual.total, 0) AS manual_seconds,
       COALESCE(t.actual_seconds, COALESCE(clock.total, 0) + COALESCE(manual.total, 0)) AS measured
FROM tasks AS t
LEFT JOIN (
    SELECT s.task_id, SUM(MAX(0, COALESCE(p.end, ?) - p.start)) AS total,
           MAX(CASE WHEN p.end IS NULL THEN p.start END) AS started_at
    FROM sessions AS s JOIN periods AS p ON p.session_id = s.id
    GROUP BY s.task_id
) AS clock ON clock.task_id = t.id
LEFT JOIN (SELECT task_id, SUM(seconds) AS total FROM manual_work GROUP BY task_id)
  AS manual ON manual.task_id=t.id
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


def _record_replay_forecast(db, task_id, at, initial, prediction_json):
    if _get_setting(db, "dev_replay_enabled", False):
        db.execute("""INSERT INTO replay_forecasts(task_id,at,initial,prediction) VALUES (?,?,?,?)""",
                   (task_id, at, initial, prediction_json))


def _comparison_sources(prediction):
    rows = list(prediction.get("evidence", [])) + list(prediction.get("related_tasks", []))
    for caution in prediction.get("cautions", []):
        rows.extend(caution.get("sources", []))
    return {str(row["task_id"]) for row in rows if row.get("task_id")}


def _forecast_sources(prediction):
    # Removal from memory also invalidates any examples used to teach the judge.
    return _comparison_sources(prediction) | set(prediction.get("feedback_source_ids", []))


def _enqueue_prediction(db, task_id, *, workspace_only=False):
    # Replacing a refresh invalidates any older lease. Store the comparison set in
    # the same transaction, so a newer completion cannot leak into the first estimate.
    if workspace_only:
        pending = db.execute("SELECT workspace_only FROM prediction_jobs WHERE task_id=?", (task_id,)).fetchone()
        workspace_only = pending is None or bool(pending[0])
    db.execute("DELETE FROM prediction_jobs WHERE task_id=?", (task_id,))
    db.execute("INSERT INTO prediction_jobs(task_id,workspace_only) VALUES (?,?)", (task_id, workspace_only))
    # Freeze mutable historical evidence at submission. A timer correction or
    # rebuilt report while the queue is draining belongs to a later forecast.
    db.execute("""INSERT INTO prediction_history(task_id, source_id, snapshot)
                  SELECT ?, id, json_object(
                      'actual_seconds', actual_seconds, 'completion_note', completion_note,
                      'report', json(report), 'pdf_text', pdf_text,
                      'prediction', json_object('point_min', json_extract(prediction, '$.point_min')),
                      'creation_prediction', json_object('point_min', json_extract(creation_prediction, '$.point_min')))
                  FROM tasks WHERE status='completed' AND actual_seconds>0 AND id!=?""",
               (task_id, task_id))


_INITIAL_JOBS = """SELECT 1 FROM prediction_jobs j0 JOIN tasks t0 ON t0.id=j0.task_id
                   WHERE t0.creation_prediction IS NULL AND (t0.archived=0 OR t0.status='completed')"""

# Finish an overlap decision while either card remains open. refresh_due separately
# controls whether a positive result should schedule another forecast.
_UNDECIDED = """
FROM task_relations r
JOIN tasks a ON a.id=r.left_id JOIN tasks b ON b.id=r.right_id
WHERE r.related IS NULL
  AND (a.archived=0 OR a.status='completed') AND (b.archived=0 OR b.status='completed')
  AND (a.status!='completed' OR b.status!='completed')
"""

_OBSOLETE_PAIRS = """DELETE FROM task_relations AS r WHERE related IS NULL
    AND EXISTS (SELECT 1 FROM tasks WHERE id=r.left_id AND status='completed')
    AND EXISTS (SELECT 1 FROM tasks WHERE id=r.right_id AND status='completed')"""


def _queue_related_target(db, target_id, *, workspace_only=False):
    row = db.execute("""SELECT creation_prediction FROM tasks WHERE id=? AND archived=0
                        AND fixed_duration=0 AND status!='completed'""", (target_id,)).fetchone()
    if row is None:
        return
    if row["creation_prediction"] is None:
        # Flag 2 is a deferred pause-only job. A new historical comparison takes priority.
        if workspace_only:
            db.execute("UPDATE tasks SET refresh_waiting=2 WHERE id=? AND refresh_waiting=0", (target_id,))
        else:
            db.execute("UPDATE tasks SET refresh_waiting=1 WHERE id=?", (target_id,))
    else:
        _enqueue_prediction(db, target_id, workspace_only=workspace_only)


def _refresh_related_estimates(db, source_id, *, corrected=False):
    """Propagate a timed completion or report edit along its known positive edges."""
    source = db.execute("SELECT fixed_duration,actual_seconds FROM tasks WHERE id=?", (source_id,)).fetchone()
    if not corrected and not source["actual_seconds"]:
        return
    rows = db.execute("""
        SELECT r.left_id,r.right_id,r.related,t.id AS target_id FROM task_relations r
        JOIN tasks t ON t.id=CASE WHEN r.left_id=? THEN r.right_id ELSE r.left_id END
        WHERE (r.left_id=? OR r.right_id=?) AND r.related IS NOT 0
          AND t.archived=0 AND t.status!='completed' AND t.fixed_duration=0
          AND (?=0 OR EXISTS(SELECT 1 FROM relevance_choices f WHERE f.target_id=t.id
                              AND f.source_id=? AND f.related=1))
    """, (source_id, source_id, source_id, source["fixed_duration"], source_id)).fetchall()
    for row in rows:
        if row["related"] is None:
            db.execute("UPDATE task_relations SET refresh_due=1 WHERE left_id=? AND right_id=?",
                       (row["left_id"], row["right_id"]))
        else:
            _queue_related_target(db, row["target_id"])
    if corrected:
        # Targets created after this source finished have no open-task matrix edge.
        queued = {row["target_id"] for row in rows}
        for target in db.execute("""SELECT id,prediction FROM tasks WHERE archived=0
                                    AND status!='completed' AND fixed_duration=0""").fetchall():
            if target["id"] not in queued and source_id in _comparison_sources(json.loads(target["prediction"])):
                _queue_related_target(db, target["id"])


def _enqueue_report(db, task_id, action="build"):
    db.execute("""INSERT INTO report_jobs(task_id,generation,action) VALUES (?,?,?)
                  ON CONFLICT(task_id) DO UPDATE SET generation=excluded.generation,
                  action=excluded.action, error=NULL""", (task_id, uuid4().hex, action))


def _close_period(db, session_id, now):
    period = db.execute(
        "SELECT id, workspace_run FROM periods WHERE session_id=? AND end IS NULL", (session_id,)
    ).fetchone()
    if period is None:
        return
    db.execute("UPDATE capture SET end=MAX(start, ?) WHERE period_id=? AND end IS NULL",
               (now, period["id"]))
    db.execute("UPDATE periods SET end=MAX(start, ?) WHERE id=?", (now, period["id"]))
    return period["workspace_run"]



def _stopped_workspace_period(db, task_id):
    return db.execute("""SELECT p.workspace_run,s.stopped_at FROM periods p JOIN sessions s ON s.id=p.session_id
                         WHERE s.task_id=? AND s.stopped_at IS NOT NULL ORDER BY s.id DESC,p.id DESC LIMIT 1""",
                      (task_id,)).fetchone()


def _correct_total(db, task, seconds, now):
    """Add unlocated time or remove the newest work first, clipping actual capture bounds."""
    try:
        seconds = float(seconds)
    except (TypeError, ValueError, OverflowError):
        raise ValueError("Enter a finite, nonnegative time spent.") from None
    if not math.isfinite(seconds) or seconds < 0:
        raise ValueError("Enter a finite, nonnegative time spent.")
    if task["status"] == "completed":
        # Some imported records have a saved duration and no corresponding clock periods.
        timed = db.execute("""SELECT COALESCE(SUM(p.end-p.start), 0) FROM periods p
                              JOIN sessions s ON s.id=p.session_id WHERE s.task_id=?""", (task["id"],)).fetchone()[0]
        legacy = task["actual_seconds"] - timed - task["manual_seconds"]
        if legacy > 0.000001:
            db.execute("INSERT INTO manual_work(task_id,entered_at,seconds) VALUES (?,?,?)",
                       (task["id"], task["completed_at"], legacy))
    difference = seconds - task["actual_seconds"]
    if difference > 0:
        db.execute("INSERT INTO manual_work(task_id, entered_at, seconds) VALUES (?,?,?)",
                   (task["id"], now, difference))
    elif difference < 0:
        remove = -difference
        work = db.execute("""
            SELECT 'timer' AS kind, p.id, p.end AS at, p.end-p.start AS seconds
            FROM periods p JOIN sessions s ON s.id=p.session_id WHERE s.task_id=? AND p.end IS NOT NULL
            UNION ALL SELECT 'manual', id, entered_at, seconds FROM manual_work WHERE task_id=?
            ORDER BY at DESC, kind, id DESC
        """, (task["id"], task["id"])).fetchall()
        for row in work:
            if remove <= 0:
                break
            trim = min(remove, row["seconds"])
            if row["kind"] == "manual":
                if trim == row["seconds"]:
                    db.execute("DELETE FROM manual_work WHERE id=?", (row["id"],))
                else:
                    db.execute("UPDATE manual_work SET seconds=seconds-? WHERE id=?", (trim, row["id"]))
            elif trim == row["seconds"]:
                db.execute("DELETE FROM periods WHERE id=?", (row["id"],))
            else:
                end = row["at"] - trim
                db.execute("DELETE FROM capture WHERE period_id=? AND start>=?", (row["id"], end))
                db.execute("UPDATE capture SET end=? WHERE period_id=? AND end>?", (end, row["id"], end))
                db.execute("UPDATE periods SET end=? WHERE id=?", (end, row["id"]))
            remove -= trim
    if difference != 0 or task["time_edited_at"] is None:
        db.execute("UPDATE tasks SET time_edited_at=? WHERE id=?", (now, task["id"]))
    return seconds


def _task_fields(title, description, estimate_minutes):
    title = title.strip()
    description = description.strip()
    if not title or not description:
        raise ValueError("Enter a task name and description.")
    estimate_minutes = float(estimate_minutes)
    if not math.isfinite(estimate_minutes) or estimate_minutes <= 0:
        raise ValueError("The estimate must be a positive number of minutes.")
    return title, description, estimate_minutes


def _category(db, category):
    category = (category or "").strip()
    if category:
        db.execute("INSERT OR IGNORE INTO categories VALUES (?, ?)", (category.casefold(), category))
        category = db.execute("SELECT name FROM categories WHERE label_key=?",
                              (category.casefold(),)).fetchone()[0]
    return category


def _repeat_chat_change(db, submission_id, task_id):
    if not submission_id:
        return False
    inserted = db.execute("INSERT OR IGNORE INTO chat_task_changes VALUES (?,?)", (submission_id, task_id))
    if not inserted.rowcount:
        previous = db.execute("SELECT task_id FROM chat_task_changes WHERE request_id=?", (submission_id,)).fetchone()
        if previous[0] != task_id:
            raise ValueError("This chat request already changed a different task.")
        return True
    return False


def _add_files(db, task_id, attachments):
    for name, data in attachments:
        name, kind = upload_info(name, data)
        db.execute("INSERT INTO attachments(id, task_id, name, pdf, kind) VALUES (?,?,?,?,?)",
                   (str(uuid4()), task_id, name, data, kind))


class PlannerStore:
    def __init__(self, data_dir, *, clock=time.time):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.data_dir / "planner.sqlite3"
        self.clock = clock
        with self._db() as db:
            db.executescript(_SCHEMA + FEEDBACK_SCHEMA)
        with self._db(write=True) as db:
            if "workspace_run" not in {row["name"] for row in db.execute("PRAGMA table_info(periods)")}:
                db.execute("ALTER TABLE periods ADD COLUMN workspace_run TEXT")
            if "workspace_only" not in {row["name"] for row in db.execute("PRAGMA table_info(prediction_jobs)")}:
                db.execute("ALTER TABLE prediction_jobs ADD COLUMN workspace_only INTEGER NOT NULL DEFAULT 0")
            history_columns = {row["name"] for row in db.execute("PRAGMA table_info(prediction_history)")}
            if "snapshot" not in history_columns:
                db.execute("ALTER TABLE prediction_history ADD COLUMN snapshot TEXT")
            if "kind" not in {row["name"] for row in db.execute("PRAGMA table_info(attachments)")}:
                db.execute("ALTER TABLE attachments ADD COLUMN kind TEXT NOT NULL DEFAULT 'pdf'")
            columns = {row["name"] for row in db.execute("PRAGMA table_info(tasks)")}
            for name, declaration in (("edit_revision", "INTEGER NOT NULL DEFAULT 0"),
                                      ("time_edited_at", "REAL"),
                                      ("archived", "INTEGER NOT NULL DEFAULT 0"),
                                      ("pdf_text_version", "INTEGER NOT NULL DEFAULT 0"),
                                      ("fixed_duration", "INTEGER NOT NULL DEFAULT 0"),
                                      ("completion_note", "TEXT NOT NULL DEFAULT ''"),
                                      ("refresh_waiting", "INTEGER NOT NULL DEFAULT 0")):
                if name not in columns:
                    db.execute(f"ALTER TABLE tasks ADD COLUMN {name} {declaration}")
            if "creation_prediction" not in columns:
                db.execute("ALTER TABLE tasks ADD COLUMN creation_prediction TEXT")
                db.execute("""UPDATE tasks SET creation_prediction=prediction
                              WHERE COALESCE(json_extract(prediction,'$.status'),'') != 'pending'""")
            if not _get_setting(db, "forecast_timer_snapshots", False):
                # Jobs saved by earlier versions treated a completed timer as immutable.
                db.execute("""UPDATE prediction_history AS h SET snapshot=json_set(
                                  coalesce(h.snapshot,'{}'), '$.actual_seconds', t.actual_seconds,
                                  '$.completion_note', t.completion_note)
                              FROM tasks AS t WHERE t.id=h.source_id
                              AND json_type(h.snapshot,'$.actual_seconds') IS NULL""")
                _save_setting(db, "forecast_timer_snapshots", True)
            if not _get_setting(db, "relation_matrix_initialized", False):
                db.execute("""INSERT OR IGNORE INTO task_relations(left_id,right_id)
                              SELECT a.id,b.id FROM tasks a JOIN tasks b ON a.id<b.id
                              WHERE a.status!='completed' AND b.status!='completed'
                                AND a.archived=0 AND b.archived=0""")
                _save_setting(db, "relation_matrix_initialized", True)
            if not _get_setting(db, "retired_completed_pairs", False):
                db.execute(_OBSOLETE_PAIRS)
                _save_setting(db, "retired_completed_pairs", True)
            if not _get_setting(db, "continue_later_is_pause", False):
                db.execute("""UPDATE sessions SET stopped_at=NULL WHERE id IN (
                              SELECT MAX(s.id) FROM sessions s JOIN tasks t ON s.task_id=t.id
                              WHERE t.status='ready' AND t.archived=0 GROUP BY s.task_id)""")
                db.execute("""UPDATE tasks SET status='paused' WHERE status='ready' AND archived=0
                              AND EXISTS (SELECT 1 FROM sessions s WHERE s.task_id=tasks.id
                                          AND s.stopped_at IS NULL)""")
                _save_setting(db, "continue_later_is_pause", True)
            if not _get_setting(db, "category_list_initialized", False):
                labels = list(DEFAULT_CATEGORIES)
                labels.extend(row[0].strip() for row in db.execute(
                    "SELECT DISTINCT category FROM tasks WHERE trim(category) != ''"))
                db.executemany("INSERT OR IGNORE INTO categories VALUES (?, ?)",
                               [(label.casefold(), label) for label in labels])
                _save_setting(db, "category_list_initialized", True)

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
            original = task["creation_prediction"]
            task["creation_prediction"] = json.loads(original) if original is not None else None
            task["archived"] = bool(task["archived"])
            task["fixed_duration"] = bool(task["fixed_duration"])
            task["estimate_updating"] = bool(task["estimate_updating"])
            task["workspace_progress_pending"] = bool(task["workspace_progress_pending"])
            task.pop("refresh_waiting")
            task["report"] = json.loads(task["report"]) if task["report"] else None
            task["attachments"] = [dict(a) for a in db.execute(
                "SELECT id, name, kind FROM attachments WHERE task_id=? ORDER BY rowid", (task["id"],)
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
                return self._tasks(db, "WHERE t.archived=0")
            return self._tasks(db, "WHERE t.archived=0 AND t.day=?", (_day(day),))

    def worked_on(self, day):
        """Visible cards with timed activity on this local date, across planning dates."""
        day = date.fromisoformat(_day(day))
        start = datetime.combine(day, local_time.min).timestamp()
        end = datetime.combine(day + timedelta(days=1), local_time.min).timestamp()
        with self._db() as db:
            return self._tasks(db, """WHERE t.archived=0 AND t.day!=? AND EXISTS (
                SELECT 1 FROM sessions s JOIN periods p ON p.session_id=s.id
                WHERE s.task_id=t.id AND p.start<? AND COALESCE(p.end,?)>?)""",
                               (day.isoformat(), end, self.clock(), start))

    def get_task(self, task_id):
        with self._db() as db:
            return self._task(db, task_id)

    def completed_tasks(self):
        with self._db() as db:
            return self._tasks(db, "WHERE t.status='completed'")

    def _first_with_status(self, status):
        with self._db() as db:
            tasks = self._tasks(db, "WHERE t.archived=0 AND t.status=?", (status,))
        return tasks[0] if tasks else None

    def running_task(self):
        return self._first_with_status("running")

    def pending_task(self):
        return self._first_with_status("awaiting_finish")

    def create_task(self, *, title, description, estimate_minutes, day, prediction=None,
                    category="", details="", attachments=(), pdf_text="", fixed_duration=False, submission_id=None):
        title, description, estimate_minutes = _task_fields(title, description, estimate_minutes)
        task_id = submission_id or str(uuid4())
        if fixed_duration:
            prediction = scheduled_prediction(estimate_minutes)
        queue_prediction = prediction is None
        if queue_prediction:
            prediction = {"status": "pending", "basis": "Pending", "evidence": []}
        with self._db(write=True) as db:
            if submission_id and db.execute("SELECT 1 FROM tasks WHERE id=?", (submission_id,)).fetchone():
                return self._task(db, submission_id)
            category = _category(db, category)
            db.execute("""
                INSERT INTO tasks(id, day, title, description, category, details, pdf_text,
                                  estimate_minutes, prediction, created_at, fixed_duration, creation_prediction)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (task_id, _day(day), title, description, category or "", details or "",
                  pdf_text or "", estimate_minutes, json.dumps(prediction, allow_nan=False),
                  self.clock(), int(fixed_duration), None if queue_prediction else json.dumps(prediction, allow_nan=False)))
            db.execute("""INSERT INTO task_relations(left_id,right_id)
                          SELECT min(id,?),max(id,?) FROM tasks
                          WHERE id!=? AND status!='completed' AND archived=0""",
                       (task_id, task_id, task_id))
            if queue_prediction:
                _enqueue_prediction(db, task_id)
            elif prediction.get("point_min") is not None:
                _record_replay_forecast(db, task_id, self.clock(), True, json.dumps(prediction, allow_nan=False))
            _add_files(db, task_id, attachments)
            return self._task(db, task_id)

    def edit_task(self, task_id, *, title, description, details, category, estimate_minutes,
                  fixed_duration, expected_revision, attachments=(), remove_ids=(),
                  submission_id=None, postpone_to=None, previous_day=None):
        title, description, estimate_minutes = _task_fields(title, description, estimate_minutes)
        with self._db(write=True) as db:
            task = self._task(db, task_id)
            if _repeat_chat_change(db, submission_id, task_id):
                return task
            if task["archived"] or task["edit_revision"] != expected_revision:
                raise ValueError("This task changed. Reopen the editor to see the latest version.")
            category = _category(db, category)
            remove_ids = set(remove_ids)
            if not remove_ids.issubset({doc["id"] for doc in task["attachments"]}):
                raise ValueError("An attachment changed. Reopen the editor.")
            fields = (title, description, details.strip(), category, estimate_minutes, bool(fixed_duration))
            old = tuple(task[key] for key in ("title", "description", "details", "category", "estimate_minutes", "fixed_duration"))
            documents_changed = bool(attachments or remove_ids)
            if postpone_to:
                self._postpone(db, task, postpone_to, previous_day)
            if old == fields and not documents_changed:
                return self._task(db, task_id)
            db.execute("""UPDATE tasks SET title=?,description=?,details=?,category=?,estimate_minutes=?,
                          fixed_duration=?,edit_revision=edit_revision+1 WHERE id=?""", (*fields, task_id))
            if documents_changed:
                db.executemany("DELETE FROM attachments WHERE id=? AND task_id=?", [(i, task_id) for i in remove_ids])
                _add_files(db, task_id, attachments)
                db.execute("UPDATE tasks SET pdf_text='',pdf_text_version=0 WHERE id=?", (task_id,))
            scope_changed = old[:4] != fields[:4] or documents_changed
            if scope_changed:
                # An older worker must never publish a judgment for the previous scope.
                db.execute("""UPDATE task_relations AS r SET related=NULL,token=NULL,lease_end=NULL,
                              retry_at=0,failures=0,refresh_due=EXISTS (
                                  SELECT 1 FROM tasks t WHERE t.id IN (r.left_id,r.right_id)
                                  AND t.status='completed' AND t.actual_seconds>0)
                              WHERE (left_id=? OR right_id=? OR (left_id,right_id) IN
                                  (SELECT left_id,right_id FROM task_relation_sources WHERE source_id=?))
                              AND NOT EXISTS (SELECT 1 FROM relevance_choices f
                                  WHERE (f.target_id=r.left_id AND f.source_id=r.right_id)
                                     OR (f.target_id=r.right_id AND f.source_id=r.left_id))
                              AND EXISTS (SELECT 1 FROM tasks t WHERE t.id IN (r.left_id,r.right_id)
                                          AND t.archived=0 AND t.status!='completed')""", (task_id, task_id, task_id))
                db.execute("UPDATE task_workspaces SET eligible=NULL,request_hash='' WHERE task_id=?", (task_id,))
                db.execute("DELETE FROM workspace_progress WHERE task_id=?", (task_id,))
                # Earlier chat stays readable but its milestones belonged to the former scope.
                db.execute("UPDATE workspace_messages SET workspace_run=NULL WHERE task_id=?", (task_id,))
            if task["status"] == "completed":
                # Preserve the historical forecast and timer. New scope can introduce a
                # comparison even for a target that previously had no link to this task.
                if task["creation_prediction"] is None:
                    _enqueue_prediction(db, task_id)
                if scope_changed:
                    _enqueue_report(db, task_id)
                if task["actual_seconds"]:
                    for target in db.execute("""SELECT id FROM tasks WHERE status!='completed' AND archived=0
                                                AND fixed_duration=0""").fetchall():
                        _enqueue_prediction(db, target["id"])
            elif fixed_duration:
                db.execute("DELETE FROM prediction_jobs WHERE task_id=?", (task_id,))
                forecast = json.dumps(scheduled_prediction(estimate_minutes))
                db.execute("""UPDATE tasks SET prediction=?,creation_prediction=coalesce(creation_prediction,?),
                              refresh_waiting=0 WHERE id=?""", (forecast, forecast, task_id))
            else:
                _enqueue_prediction(db, task_id)
            return self._task(db, task_id)

    def retry_estimate(self, task_id):
        """Recompute an unfinished task, preserving its timer and creation forecast."""
        with self._db(write=True) as db:
            task = self._task(db, task_id)
            if task["estimate_updating"] or task["status"] == "completed" or task["fixed_duration"] or task["archived"]:
                return False
            prediction = task["prediction"]
            retry_progress = ((prediction.get("workspace_update") or {}).get("status") == "unavailable"
                              and not prediction.get("retrieval_error") and not prediction.get("model_error"))
            _enqueue_prediction(db, task_id, workspace_only=retry_progress)
            return True

    def comparison_options(self, task_id):
        with self._db() as db:
            return comparison_options(db, task_id)

    def revise_related_tasks(self, task_id, selected_ids, *, previous_ids, reviewed_ids):
        """Save the user's exact selection and enqueue its forecast atomically."""
        with self._db(write=True) as db:
            task = self._task(db, task_id)
            if task["archived"] or task["fixed_duration"] or task["status"] == "completed" or task["estimate_updating"]:
                raise ValueError("The task must be unfinished and its previous estimate complete.")
            baseline = related_ids(task["prediction"])
            if set(previous_ids) != set(baseline):
                raise ValueError("The forecast changed. Reopen the editor to review the current comparisons.")
            save_choices(db, task_id, selected_ids, baseline, reviewed_ids, self.clock())
            _enqueue_prediction(db, task_id)

    def has_pending_predictions(self):
        """Pending work includes currently leased jobs and is independent of the selected day."""
        with self._db() as db:
            return db.execute("SELECT 1 FROM prediction_jobs LIMIT 1").fetchone() is not None

    def next_estimation_delay(self):
        """Seconds until the next forecast or pair is claimable; None when idle."""
        with self._db() as db:
            now = self.clock()
            # If another process owns an initial forecast, both refreshes and
            # pair checks must wait until that forecast is saved or reclaimed.
            initial = db.execute(_INITIAL_JOBS + " LIMIT 1").fetchone()
            jobs = db.execute("""SELECT min(coalesce(j.lease_end,0)) FROM prediction_jobs j
                                 JOIN tasks t ON t.id=j.task_id WHERE (t.archived=0 OR t.status='completed')
                                 AND (?=0 OR t.creation_prediction IS NULL)""", (int(bool(initial)),)).fetchone()[0]
            due = [jobs] if jobs is not None else []
            if not initial:
                pairs = db.execute("SELECT min(max(r.retry_at,coalesce(r.lease_end,0))) " + _UNDECIDED).fetchone()[0]
                if pairs is not None:
                    due.append(pairs)
            return max(0, min(due) - now) if due else None

    def has_estimation_work(self):
        return self.next_estimation_delay() is not None

    def claim_relation(self, *, lease_seconds=300):
        """Claim one distinct pair after initial forecasts have released the model."""
        if not math.isfinite(lease_seconds) or lease_seconds <= 0:
            raise ValueError("The relevance lease must be a positive number of seconds.")
        with self._db(write=True) as db:
            now = self.clock()
            row = db.execute("SELECT r.left_id,r.right_id " + _UNDECIDED + """
                AND r.retry_at<=? AND (r.token IS NULL OR r.lease_end<=?)
                AND NOT EXISTS (""" + _INITIAL_JOBS + """)
                ORDER BY r.refresh_due DESC,r.retry_at,r.rowid LIMIT 1
            """, (now, now)).fetchone()
            if row is None:
                return None
            token = str(uuid4())
            pair = tuple(row)
            db.execute("""UPDATE task_relations SET token=?,lease_end=?
                          WHERE left_id=? AND right_id=?""", (token, now + lease_seconds, *pair))
            # The job identifies which task created the cell, even if it already
            # finished. The worker chooses a stable model input order after loading PDFs.
            target, candidate = [dict(t) for t in db.execute("""
                SELECT id,title,description,category,details,pdf_text,pdf_text_version,edit_revision FROM tasks WHERE id IN (?,?)
                ORDER BY created_at DESC,rowid DESC""", pair)]
            for task in (target, candidate):
                # Fixed-duration and legacy tasks can have uploads but no forecasting job to parse them.
                if not task["pdf_text_version"]:
                    task["attachments"] = [dict(row) for row in db.execute(
                        "SELECT id,name,kind FROM attachments WHERE task_id=? ORDER BY rowid", (task["id"],))]
            return {"pair": pair, "target": target, "candidate": candidate, "token": token,
                    "examples": relevance_examples(db, target, candidate)}

    def renew_relation(self, pair, token, *, lease_seconds=300):
        with self._db(write=True) as db:
            now = self.clock()
            return bool(db.execute("""UPDATE task_relations SET lease_end=? WHERE left_id=?
                                      AND right_id=? AND token=? AND lease_end>?""",
                                   (now + lease_seconds, *pair, token, now)).rowcount)

    def resolve_relation(self, pair, token, related, *, examples=()):
        """Commit a binary decision, or leave an unavailable model's pair for retry."""
        if related is not None and not isinstance(related, bool):
            raise ValueError("A relevance decision must be a boolean.")
        with self._db(write=True) as db:
            now = self.clock()
            row = db.execute("""SELECT refresh_due,failures FROM task_relations WHERE left_id=?
                                AND right_id=? AND token=? AND lease_end>?""", (*pair, token, now)).fetchone()
            if row is None:
                return False
            if related is None:
                delay = min(300, 15 * 2 ** min(row["failures"], 5))
                db.execute("""UPDATE task_relations SET token=NULL,lease_end=NULL,failures=failures+1,retry_at=?
                              WHERE left_id=? AND right_id=?""", (now + delay, *pair))
            else:
                db.execute("""UPDATE task_relations SET related=?,refresh_due=0,token=NULL,lease_end=NULL
                              WHERE left_id=? AND right_id=?""", (int(related), *pair))
                db.execute("DELETE FROM task_relation_sources WHERE left_id=? AND right_id=?", pair)
                source_ids = {example[key] for example in examples
                              for key in ("target_id", "candidate_id") if example.get(key)}
                db.executemany("""INSERT INTO task_relation_sources(left_id,right_id,source_id)
                                  VALUES (?,?,?)""", [(*pair, source_id) for source_id in source_ids])
                if related and row["refresh_due"]:
                    for task_id in pair:
                        _queue_related_target(db, task_id)
            return True

    def claim_prediction(self, *, lease_seconds=300):
        """Lease one eligible estimate. Its source tasks were already finished when it was created."""
        if not math.isfinite(lease_seconds) or lease_seconds <= 0:
            raise ValueError("The estimate lease must be a positive number of seconds.")
        with self._db(write=True) as db:
            now = self.clock()
            row = db.execute("""
                SELECT t.id,j.workspace_only FROM prediction_jobs j JOIN tasks t ON t.id=j.task_id
                WHERE (t.archived=0 OR t.status='completed') AND (j.token IS NULL OR j.lease_end<=?)
                  AND (t.creation_prediction IS NULL OR NOT EXISTS (""" + _INITIAL_JOBS + """))
                  AND (t.creation_prediction IS NULL OR NOT EXISTS (
                      SELECT 1 FROM task_relations r WHERE (r.left_id=t.id OR r.right_id=t.id)
                      AND r.refresh_due=1 AND r.related IS NULL AND r.retry_at<=?
                      AND (r.token IS NULL OR r.lease_end<=?)))
                ORDER BY t.creation_prediction IS NOT NULL,t.created_at,t.rowid LIMIT 1
            """, (now, now, now)).fetchone()
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
            snapshots = dict(db.execute("SELECT source_id,snapshot FROM prediction_history WHERE task_id=?",
                                        (task_id,)))
            # These tasks were judged as peers while both cards were open. A completion
            # refresh must use the same verdict that triggered it, even if the newly
            # completed task was originally the target of the symmetric matrix check.
            peers = {row["source_id"]: row for row in db.execute("""
                SELECT h.source_id,r.related,coalesce((SELECT json_group_array(s.source_id)
                  FROM task_relation_sources s WHERE s.left_id=r.left_id AND s.right_id=r.right_id),'[]') AS examples
                FROM prediction_history h JOIN task_relations r
                  ON r.left_id=min(h.task_id,h.source_id) AND r.right_id=max(h.task_id,h.source_id)
                WHERE h.task_id=? AND r.related IS NOT NULL""", (task_id,))}
            from activity import exclusion_terms, visible_saved_event
            needles = exclusion_terms(_get_setting(db, "exclusions", []))
            for source in history:
                peer = peers.get(source["id"])
                if peer is not None:
                    source["matrix_related"] = bool(peer["related"])
                    source["matrix_feedback_ids"] = json.loads(peer["examples"])
                snapshot = snapshots[source["id"]]
                if snapshot is not None:
                    fields = json.loads(snapshot)
                    # Uploads are immutable. Prefer the complete local extraction of those original pages.
                    if source["pdf_text_version"]:
                        fields.pop("pdf_text", None)
                    source.update(fields)
                if needles and source.get("report"):
                    # A saved report may predate an exclusion. Apply today's rules to both
                    # old snapshots and user-reviewed labels before exposing forecast evidence.
                    report = source["report"]
                    source["report"] = {"intervals": [event for event in report.get("intervals", [])
                                                       if visible_saved_event(event, needles)],
                                        "feedback_source_ids": report.get("feedback_source_ids", [])}
            # A task can finish before its forecast worker runs. Only information
            # available to the user at creation is a valid forecast input.
            inputs = {key: task[key] for key in ("id", "title", "description", "category", "details",
                      "day", "estimate_minutes", "fixed_duration", "pdf_text", "pdf_text_version", "attachments", "edit_revision")}
            inputs.update(forecast_feedback(db, task_id))
            job = {"task": inputs, "history": history, "token": token,
                   "focus_verdicts": (task.get("prediction") or {}).get("focus_verdicts", {})}
            if task["creation_prediction"] is not None and task["status"] != "completed":
                progress = db.execute("SELECT snapshot FROM workspace_progress WHERE task_id=?", (task_id,)).fetchone()
                if progress:
                    job.update(workspace_progress=json.loads(progress[0]), workspace_only=bool(row["workspace_only"]),
                               previous_prediction=task["prediction"])
            return job

    def renew_prediction(self, task_id, token, *, lease_seconds=300):
        if not math.isfinite(lease_seconds) or lease_seconds <= 0:
            raise ValueError("The estimate lease must be a positive number of seconds.")
        with self._db(write=True) as db:
            now = self.clock()
            return bool(db.execute("""
                UPDATE prediction_jobs SET lease_end=? WHERE task_id=? AND token=? AND lease_end>?
            """, (now + lease_seconds, task_id, token, now)).rowcount)

    def resolve_prediction(self, task_id, token, prediction, *, pdf_text=None):
        """Save the current forecast while preserving the task's initial prediction."""
        with self._db(write=True) as db:
            # A task submitted after a source finished has no open/open matrix
            # edge to it. Until this forecast returns, the planner cannot know
            # which of the historical records it used. Check those records now,
            # before deleting the job cascades to its snapshots.
            used = _comparison_sources(prediction)
            changed_source = used and db.execute("""SELECT 1 FROM prediction_history h
                JOIN tasks s ON s.id=h.source_id
                WHERE h.task_id=? AND h.source_id IN (SELECT value FROM json_each(?))
                AND h.snapshot IS NOT NULL
                AND (s.actual_seconds IS NOT json_extract(h.snapshot,'$.actual_seconds')
                     OR s.completion_note IS NOT json_extract(h.snapshot,'$.completion_note')
                     OR json(s.report) IS NOT json_extract(h.snapshot,'$.report')) LIMIT 1""",
                (task_id, json.dumps(sorted(used)))).fetchone()
            deleted = db.execute("DELETE FROM prediction_jobs WHERE task_id=? AND token=? AND lease_end>?",
                                 (task_id, token, self.clock()))
            if not deleted.rowcount:
                return False
            data = json.dumps(prediction, allow_nan=False)
            waiting = db.execute("""SELECT refresh_waiting,status,creation_prediction IS NULL AS initial
                                    FROM tasks WHERE id=?""", (task_id,)).fetchone()
            if (prediction.get("workspace_update") or {}).get("status") in {"adjusted", "insufficient_evidence"}:
                db.execute("UPDATE workspace_progress SET applied_message=last_message WHERE task_id=?", (task_id,))
            _record_replay_forecast(db, task_id, self.clock(), waiting["initial"], data)
            db.execute("""UPDATE tasks SET prediction=?, creation_prediction=COALESCE(creation_prediction, ?),
                          pdf_text=COALESCE(?,pdf_text), refresh_waiting=0 WHERE id=?""",
                       (data, data, pdf_text, task_id))
            if (waiting["refresh_waiting"] or changed_source) and waiting["status"] != "completed":
                _enqueue_prediction(db, task_id, workspace_only=waiting["refresh_waiting"] == 2 and not changed_source)
            return True

    def forecast_timeline(self, task_id):
        with self._db() as db:
            return [{"at": row["at"], "initial": bool(row["initial"]), "prediction": json.loads(row["prediction"])}
                    for row in db.execute("""SELECT at,initial,prediction FROM replay_forecasts WHERE task_id=?
                                             ORDER BY at,id""", (task_id,))]

    def _workspace_pause(self, db, task_id, run, paused_at):
        if not run:
            return
        target = self._task(db, task_id)
        if target["fixed_duration"] or target["archived"]:
            return
        prior = db.execute("SELECT last_message,applied_message FROM workspace_progress WHERE task_id=?",
                           (task_id,)).fetchone()
        previous = prior["last_message"] if prior else 0
        applied = prior["applied_message"] if prior else 0
        fresh = db.execute("""SELECT 1 FROM workspace_messages WHERE task_id=? AND workspace_run=?
                              AND role='user' AND id>? AND recorded_at<=?""", (task_id, run, previous, paused_at)).fetchone()
        if not fresh:
            return
        rows = db.execute("""SELECT m.id,m.role,m.content,a.name AS document
                             FROM workspace_messages m JOIN workspace_messages q ON q.task_id=m.task_id
                                 AND q.document_id=m.document_id AND q.turn_id=m.turn_id AND q.role='user'
                             LEFT JOIN attachments a ON a.id=m.document_id AND a.task_id=m.task_id
                             WHERE m.task_id=? AND m.workspace_run IS NOT NULL AND m.recorded_at<=?
                             ORDER BY q.id, m.id""", (task_id, paused_at)).fetchall()
        snapshot = {"paused_at": paused_at, "elapsed_seconds": target["actual_seconds"],
                    "messages": [{"role": r["role"], "content": r["content"], "document": r["document"]} for r in rows],
                    "new_message_start": sum(r["id"] <= applied for r in rows)}
        db.execute("""INSERT INTO workspace_progress(task_id,last_message,snapshot) VALUES (?,?,?)
                      ON CONFLICT(task_id) DO UPDATE SET last_message=excluded.last_message,snapshot=excluded.snapshot""",
                   (task_id, max(r["id"] for r in rows), json.dumps(snapshot, allow_nan=False)))
        _queue_related_target(db, task_id, workspace_only=True)

    def transition(self, task_id, action, *, ended_at=None, completion_note="", report=None,
                   pdf_text=None, refresh_estimates=True, queue_report=False, spent_seconds=None, workspace_token=None):
        """Apply a timer action and return the committed state. Starting another task pauses the first."""
        with self._db(write=True) as db:
            task = self._task(db, task_id)
            if task["archived"]:
                raise ValueError("This task has been removed from the planner.")
            state = task["status"]
            now = self.clock()
            session = db.execute("SELECT id FROM sessions WHERE task_id=? AND stopped_at IS NULL",
                                 (task_id,)).fetchone()
            repeated = {"start": "running", "resume": "running", "pause": "paused",
                        "stop": "awaiting_finish", "finish": "completed", "continue_later": "paused"}
            if ended_at is not None and action != "stop":
                raise ValueError("An end-time correction is only available when stopping a task.")
            if spent_seconds is not None and action not in ("adjust_time", "finish", "continue_later"):
                raise ValueError("Stop or pause the timer before editing total time.")
            if workspace_token is not None:
                if action not in ("start", "resume") or not db.execute(
                    "SELECT 1 FROM task_workspaces WHERE task_id=? AND token=? AND eligible=1", (task_id, workspace_token)
                ).fetchone():
                    raise ValueError("This task workspace is unavailable.")
                if state == "running":
                    db.execute("""UPDATE periods SET workspace_run=coalesce(workspace_run,?)
                                  WHERE session_id=? AND end IS NULL""", (str(uuid4()), session["id"]))
            if repeated.get(action) == state:
                return task
            stop_run = None
            if action == "continue_later" and state == "awaiting_finish":
                stop_run = _stopped_workspace_period(db, task_id)
            if ended_at is not None:
                end = float(ended_at)
                if state != "running" or not task["started_at"] <= end <= now:
                    raise ValueError("Choose an end time between the current timer start and now.")
                db.execute("UPDATE tasks SET time_edited_at=? WHERE id=?", (now, task_id))
                now = end
                period = db.execute("SELECT id FROM periods WHERE session_id=? AND end IS NULL",
                                    (session["id"],)).fetchone()
                db.execute("DELETE FROM capture WHERE period_id=? AND start>=?", (period["id"], end))
                db.execute("UPDATE capture SET end=? WHERE period_id=? AND end>?", (end, period["id"], end))
            if action == "adjust_time" or spent_seconds is not None:
                if state == "running":
                    raise ValueError("Stop or pause the timer before editing total time.")
                previous = task["actual_seconds"]
                total = _correct_total(db, task, spent_seconds, now)
                task["actual_seconds"] = total
                if total != previous and state != "completed":
                    updated = db.execute("""UPDATE workspace_progress
                                            SET snapshot=json_set(snapshot,'$.elapsed_seconds',?)
                                            WHERE task_id=?""", (total, task_id)).rowcount
                    if updated:
                        _queue_related_target(db, task_id, workspace_only=True)
                if action == "adjust_time":
                    if state == "completed" and (total != previous or task["time_edited_at"] is None):
                        db.execute("UPDATE tasks SET actual_seconds=?, report=NULL WHERE id=?", (total, task_id))
                        _enqueue_report(db, task_id)
                        if previous == 0 and total > 0 and not task["fixed_duration"]:
                            # This source only now qualifies as timed history. Later tasks may have no edge to it.
                            db.execute("""INSERT OR IGNORE INTO task_relations(left_id,right_id,refresh_due)
                                          SELECT MIN(?,id),MAX(?,id),1 FROM tasks
                                          WHERE id!=? AND archived=0 AND fixed_duration=0 AND status!='completed'""",
                                       (task_id, task_id, task_id))
                        if total != previous:
                            _refresh_related_estimates(db, task_id, corrected=True)
                    elif state == "ready" and total > 0:
                        db.execute("UPDATE tasks SET status='paused' WHERE id=?", (task_id,))
                    return self._task(db, task_id)
            if action in ("start", "resume") and state in ("ready", "paused"):
                active = db.execute("""
                    SELECT s.task_id, s.id FROM sessions s JOIN periods p ON p.session_id=s.id
                    WHERE p.end IS NULL
                """).fetchone()
                if active:
                    run = _close_period(db, active["id"], now)
                    db.execute("UPDATE tasks SET status='paused' WHERE id=?", (active["task_id"],))
                    self._workspace_pause(db, active["task_id"], run, now)
                sid = session["id"] if session else db.execute(
                    "INSERT INTO sessions(task_id, started_at) VALUES (?,?)", (task_id, now)
                ).lastrowid
                run = str(uuid4()) if workspace_token is not None else None
                pid = db.execute("INSERT INTO periods(session_id,start,workspace_run) VALUES (?,?,?)",
                                 (sid, now, run)).lastrowid
                if _get_setting(db, "capture_enabled", True):
                    db.execute("INSERT INTO capture(period_id,start) VALUES (?,?)", (pid, now))
                state = "running"
            elif action == "pause" and state == "running":
                run = _close_period(db, session["id"], now)
                self._workspace_pause(db, task_id, run, now)
                state = "paused"
            elif action == "stop" and state in ("paused", "running"):
                if session:
                    run = _close_period(db, session["id"], now)
                    if run and ended_at is not None:
                        # Keep the conversation readable; the user moved these turns outside the work interval.
                        db.execute("""UPDATE workspace_messages SET workspace_run=NULL
                                      WHERE task_id=? AND workspace_run=? AND recorded_at>?""", (task_id, run, now))
                    db.execute("UPDATE sessions SET stopped_at=MAX(started_at,?) WHERE id=?", (now, session["id"]))
                state = "awaiting_finish"
            elif action == "continue_later" and state == "awaiting_finish":
                if stop_run:
                    self._workspace_pause(db, task_id, stop_run["workspace_run"], stop_run["stopped_at"])
                # Stop closed the timer immediately. Resume should use the same
                # session as Pause, including decisions pending from older versions.
                db.execute("""UPDATE sessions SET stopped_at=NULL WHERE id=(
                              SELECT MAX(id) FROM sessions WHERE task_id=?)""", (task_id,))
                state = "paused"
            elif action == "finish" and state in ("ready", "paused", "awaiting_finish"):
                if session:
                    db.execute("UPDATE sessions SET stopped_at=MAX(started_at,?) WHERE id=?", (now, session["id"]))
                db.execute("""UPDATE tasks SET completed_at=?, actual_seconds=?, completion_note=?,
                              report=?, pdf_text=COALESCE(?,pdf_text), refresh_waiting=0 WHERE id=?""",
                           (now, task["actual_seconds"], completion_note.strip(),
                            json.dumps(report, allow_nan=False) if report is not None else None, pdf_text, task_id))
                if task["creation_prediction"] is not None:
                    db.execute("DELETE FROM prediction_jobs WHERE task_id=?", (task_id,))
                if queue_report:
                    _enqueue_report(db, task_id)
                state = "completed"
            else:
                raise ValueError(f"Cannot {action.replace('_', ' ')} a {state} task.")
            db.execute("UPDATE tasks SET status=? WHERE id=?", (state, task_id))
            if action == "finish":
                # No future completion can refresh either endpoint once both are
                # finished. Keep decisions already made and retire obsolete jobs.
                db.execute(_OBSOLETE_PAIRS + " AND (left_id=? OR right_id=?)", (task_id, task_id))
                if refresh_estimates and task["actual_seconds"] > 0:
                    _refresh_related_estimates(db, task_id)
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

    def next_report_job(self):
        with self._db() as db:
            row = db.execute("SELECT * FROM report_jobs WHERE error IS NULL ORDER BY rowid LIMIT 1").fetchone()
            return dict(row) if row else None

    def report_job(self, task_id):
        with self._db() as db:
            row = db.execute("SELECT * FROM report_jobs WHERE task_id=?", (task_id,)).fetchone()
            return dict(row) if row else None

    def request_report(self, task_id, edits=None):
        """Persist corrections and schedule regeneration in one transaction."""
        import activity

        with self._db(write=True) as db:
            task = self._task(db, task_id)
            if task["status"] != "completed":
                raise ValueError("Finish the task before requesting an activity report.")
            groups = activity.review_groups(task["report"] or {})
            for key, label in (edits or {}).items():
                if key not in groups or label not in activity.LABELS:
                    raise ValueError("This activity has changed. Reopen the report and review it again.")
                db.execute("""INSERT INTO activity_feedback VALUES (?,?,?,?)
                              ON CONFLICT(task_id,activity_key) DO UPDATE SET
                              label=excluded.label, updated_at=excluded.updated_at""",
                           (task_id, key, label, self.clock()))
            _enqueue_report(db, task_id, "review" if edits else "build")

    def report_feedback(self, task):
        with self._db() as db:
            overrides = dict(db.execute("SELECT activity_key,label FROM activity_feedback WHERE task_id=?",
                                        (task["id"],)))
            blocked = _get_setting(db, "exclusions", [])
            if isinstance(blocked, str):
                blocked = blocked.splitlines()
            needles = [term.strip().casefold() for term in blocked if term.strip()]
            examples = []
            for row in db.execute("""SELECT f.*, t.title, t.category FROM activity_feedback f
                                     JOIN tasks t ON t.id=f.task_id WHERE t.id!=?
                                     ORDER BY t.category=? COLLATE NOCASE DESC, f.updated_at DESC, f.rowid DESC""",
                                  (task["id"], task["category"])):
                app, domain, title, phone, idle = json.loads(row["activity_key"])
                if any(term in raw.casefold() for raw in (app, domain, title, phone) for term in needles):
                    continue
                examples.append({"source_id": row["task_id"], "task": row["title"][:120], "category": row["category"],
                                 "app": app, "domain": domain, "title": title, "phone_app": phone, "idle": idle,
                                 "label": row["label"]})
                if len(examples) == 6:
                    break
            return overrides, examples

    def fail_report(self, task_id, generation, message):
        with self._db(write=True) as db:
            db.execute("UPDATE report_jobs SET error=? WHERE task_id=? AND generation=?",
                       (message, task_id, generation))

    def save_report(self, task_id, report, *, refresh_estimates=True, generation=None, pdf_text=None):
        with self._db(write=True) as db:
            if generation is not None:
                claimed = db.execute("DELETE FROM report_jobs WHERE task_id=? AND generation=?",
                                     (task_id, generation))
                if not claimed.rowcount:
                    return False
            source = db.execute("SELECT report,pdf_text,actual_seconds FROM tasks WHERE id=? AND status='completed'",
                                (task_id,)).fetchone()
            if source is None:
                raise ValueError("Finish the task before saving its activity review.")
            # A rebuild can produce exactly the same observations. Preserve an in-flight
            # forecast's lease (and its historical snapshot) when nothing changed.
            if (source["report"] is not None and json.loads(source["report"]) == report
                    and (pdf_text is None or pdf_text == source["pdf_text"])):
                return True
            db.execute("UPDATE tasks SET report=?, pdf_text=COALESCE(?,pdf_text) WHERE id=?",
                       (json.dumps(report, allow_nan=False), pdf_text, task_id))
            if refresh_estimates and source["actual_seconds"] > 0:
                _refresh_related_estimates(db, task_id, corrected=True)
            return True

    def cache_document_text(self, task_id, text, *, expected_revision):
        """Upgrade legacy sampled excerpts once from the immutable original upload."""
        with self._db(write=True) as db:
            db.execute("UPDATE tasks SET pdf_text=?,pdf_text_version=1 WHERE id=? AND pdf_text_version=0 AND edit_revision=?",
                       (text, task_id, expected_revision))

    def attachment_bytes(self, attachment_id):
        with self._db() as db:
            row = db.execute("SELECT pdf FROM attachments WHERE id=?", (attachment_id,)).fetchone()
            if row is None:
                raise ValueError("This attachment no longer exists.")
            return row[0]

    def reschedule(self, task_id, day):
        with self._db(write=True) as db:
            result = db.execute("UPDATE tasks SET day=? WHERE id=? AND archived=0 AND status!='completed'",
                                (_day(day), task_id))
            if not result.rowcount:
                raise ValueError("Only unfinished tasks can be rescheduled.")
            return self._task(db, task_id)

    def postpone(self, task_id, day, *, previous_day, submission_id=None, expected_revision=None):
        """Move unfinished work to a later day and leave its existing session paused."""
        with self._db(write=True) as db:
            task = self._task(db, task_id)
            if not _repeat_chat_change(db, submission_id, task_id):
                if expected_revision is not None and task["edit_revision"] != expected_revision:
                    raise ValueError("This task changed. Check its current details and try again.")
                self._postpone(db, task, day, previous_day)
            return self._task(db, task_id)

    def _postpone(self, db, task, day, previous_day):
        task_id = task["id"]
        day = _day(day)
        if task["archived"] or task["status"] == "completed":
            raise ValueError("Only unfinished tasks can be postponed.")
        if task["day"] != _day(previous_day):
            raise ValueError("This task was already moved. Open its current plan to postpone it again.")
        now = self.clock()
        if day <= task["day"] or day < date.fromtimestamp(now).isoformat():
            raise ValueError("Choose a later plan date that is today or in the future.")
        state = task["status"]
        if state == "running":
            session = db.execute("SELECT id FROM sessions WHERE task_id=? AND stopped_at IS NULL",
                                 (task_id,)).fetchone()
            run = _close_period(db, session["id"], now)
            self._workspace_pause(db, task_id, run, now)
            state = "paused"
        elif state == "awaiting_finish":
            period = _stopped_workspace_period(db, task_id)
            if period:
                self._workspace_pause(db, task_id, period["workspace_run"], period["stopped_at"])
            db.execute("UPDATE sessions SET stopped_at=NULL WHERE id=(SELECT max(id) FROM sessions WHERE task_id=?)",
                       (task_id,))
            state = "paused"
        db.execute("UPDATE tasks SET day=?, status=? WHERE id=?", (day, state, task_id))

    def archived_tasks(self):
        with self._db() as db:
            return [dict(row) for row in db.execute(
                "SELECT id,title,day FROM tasks WHERE archived=1 ORDER BY day DESC,created_at DESC")]

    def archive_task(self, task_id):
        """Hide the card and close active timers while retaining historical evidence."""
        with self._db(write=True) as db:
            task = self._task(db, task_id)
            if task["archived"]:
                return
            now = self.clock()
            for row in db.execute("SELECT id FROM sessions WHERE task_id=? AND stopped_at IS NULL",
                                  (task_id,)).fetchall():
                _close_period(db, row["id"], now)
                db.execute("UPDATE sessions SET stopped_at=MAX(started_at, ?) WHERE id=?", (now, row["id"]))
            db.execute("""UPDATE tasks SET archived=1, refresh_waiting=0,
                          status=CASE WHEN status IN ('running','awaiting_finish') THEN 'paused' ELSE status END
                          WHERE id=?""", (task_id,))
            if task["status"] != "completed":
                db.execute("DELETE FROM prediction_jobs WHERE task_id=?", (task_id,))
                db.execute("DELETE FROM task_relations WHERE left_id=? OR right_id=?", (task_id, task_id))

    def delete_task(self, task_id):
        """Forget a task, revoke in-flight reads and scrub dependent saved outputs."""
        with self._db(write=True) as db:
            forgotten = self._task(db, task_id)
            db.execute("UPDATE prediction_jobs SET token=NULL, lease_end=NULL WHERE token IS NOT NULL")
            db.execute("UPDATE task_relations SET token=NULL, lease_end=NULL WHERE related IS NULL")
            db.execute("DELETE FROM settings WHERE key GLOB 'dev_task_*' AND value=?", (json.dumps(task_id),))
            # Rejudge edges learned from a now-forgotten human correction. A
            # decided edge whose participants remain can outlive an example.
            db.execute("""UPDATE task_relations AS r SET related=NULL,retry_at=0,failures=0,
                          refresh_due=EXISTS(SELECT 1 FROM tasks t WHERE t.id IN (r.left_id,r.right_id)
                                             AND t.status='completed' AND t.actual_seconds>0)
                          WHERE (r.left_id,r.right_id) IN
                                (SELECT left_id,right_id FROM task_relation_sources WHERE source_id=?)""",
                       (task_id,))
            db.execute("DELETE FROM tasks WHERE id=?", (task_id,))
            db.execute("DELETE FROM settings WHERE key=?", (f"dev_activity_{task_id}",))
            db.execute("INSERT OR IGNORE INTO vector_purges VALUES (?)", (task_id,))
            _save_setting(db, "memory_revision", _get_setting(db, "memory_revision", 0) + 1)
            affected_forecasts = []
            for row in db.execute("SELECT id, created_at, completed_at, prediction, creation_prediction, report FROM tasks").fetchall():
                for column in ("prediction", "creation_prediction"):
                    if row[column] is None:
                        continue
                    prediction = json.loads(row[column])
                    # Old forecasts did not record every model-visible comparison.
                    # For those versions only, erase prose if this source could have been available.
                    latest = row["created_at"] if column == "creation_prediction" else row["completed_at"]
                    legacy = ("related_tasks" not in prediction and forgotten["status"] == "completed"
                              and forgotten["actual_seconds"] > 0
                              and (latest is None or latest >= forgotten["completed_at"])
                              and (prediction.get("model_rationale") or prediction.get("evidence")
                                   or prediction.get("provenance", {}).get("minutes_by") == "model"
                                   or prediction.get("basis") == "Model-selected tasks"))
                    if legacy or task_id in _forecast_sources(prediction):
                        # Free text can cite any of the model's inputs. Keep only the original
                        # numeric forecast and attribution for comparisons against actual time.
                        safe = {key: prediction[key] for key in ("point_min", "low_min", "high_min", "confidence",
                                "basis", "provenance") if key in prediction}
                        safe.update(evidence=[], related_tasks=[], explanation="Supporting history was removed from memory.")
                        db.execute(f"UPDATE tasks SET {column}=? WHERE id=?", (json.dumps(safe), row["id"]))
                        if column == "prediction":
                            affected_forecasts.append(row["id"])
                if row["report"] and task_id in json.loads(row["report"]).get("feedback_source_ids", []):
                    db.execute("UPDATE tasks SET report=NULL WHERE id=?", (row["id"],))
                    _enqueue_report(db, row["id"])
            for row in db.execute("SELECT id,prediction FROM replay_forecasts").fetchall():
                if task_id in _forecast_sources(json.loads(row["prediction"])):
                    db.execute("DELETE FROM replay_forecasts WHERE id=?", (row["id"],))
            # Snapshots in queued forecasts can also contain an earlier version of a report.
            db.execute("""UPDATE prediction_history SET snapshot=json_set(snapshot,'$.report',NULL)
                          WHERE snapshot IS NOT NULL AND EXISTS
                          (SELECT 1 FROM json_each(snapshot,'$.report.feedback_source_ids') WHERE value=?)""", (task_id,))
            db.execute("""UPDATE report_jobs SET generation=lower(hex(randomblob(16))) WHERE error IS NULL""")
            for target_id in affected_forecasts:
                _queue_related_target(db, target_id)

    def pending_vector_purges(self):
        with self._db() as db:
            return [r[0] for r in db.execute("SELECT task_id FROM vector_purges")]

    def confirm_vector_purge(self, task_id):
        with self._db(write=True) as db:
            db.execute("DELETE FROM vector_purges WHERE task_id=?", (task_id,))

    def clear_data(self):
        with self._db(write=True) as db:
            revision = _get_setting(db, "memory_revision", 0)
            db.execute("INSERT OR IGNORE INTO vector_purges SELECT id FROM tasks")
            db.execute("DELETE FROM tasks")
            db.execute("DELETE FROM settings")
            _save_setting(db, "memory_revision", revision + 1)
            db.execute("DELETE FROM categories")
            db.executemany("INSERT INTO categories VALUES (?, ?)",
                           [(label.casefold(), label) for label in DEFAULT_CATEGORIES])
            _save_setting(db, "category_list_initialized", True)

    def list_categories(self):
        with self._db() as db:
            return [row[0] for row in db.execute("SELECT name FROM categories ORDER BY rowid")]

    def add_category(self, name):
        name = name.strip()
        if not name:
            raise ValueError("Enter a category name.")
        with self._db(write=True) as db:
            inserted = db.execute("INSERT OR IGNORE INTO categories VALUES (?, ?)", (name.casefold(), name))
            if not inserted.rowcount:
                raise ValueError("That category already exists.")
        return name

    def get_setting(self, key, default=None):
        with self._db() as db:
            return _get_setting(db, key, default)

    def set_setting(self, key, value):
        if key == "capture_enabled":
            self.set_capture(value)
            return
        with self._db(write=True) as db:
            if key == "exclusions" and value != _get_setting(db, key, []):
                # A retry rereads the current exclusions before it can publish its result.
                db.execute("UPDATE prediction_jobs SET token=NULL, lease_end=NULL WHERE token IS NOT NULL")
                db.execute("UPDATE report_jobs SET generation=lower(hex(randomblob(16))) WHERE error IS NULL")
            _save_setting(db, key, value)
