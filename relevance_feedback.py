"""Workspace-local comparison choices and human-reviewed relevance examples."""


from pdf_reader import document_excerpt


FEEDBACK_SCHEMA = """
CREATE TABLE IF NOT EXISTS relevance_choices (
    target_id TEXT NOT NULL REFERENCES tasks(id) ON DELETE CASCADE,
    source_id TEXT NOT NULL REFERENCES tasks(id) ON DELETE CASCADE,
    related INTEGER NOT NULL CHECK(related IN (0,1)),
    user_changed INTEGER NOT NULL CHECK(user_changed IN (0,1)),
    changed_at REAL NOT NULL,
    PRIMARY KEY(target_id, source_id),
    CHECK(target_id != source_id)
);
CREATE INDEX IF NOT EXISTS relevance_choices_source ON relevance_choices(source_id);
CREATE INDEX IF NOT EXISTS relevance_examples_by_decision ON relevance_choices(user_changed, related, changed_at);
"""


def related_ids(prediction):
    """Older forecasts recorded only the cited comparisons."""
    entries = prediction.get("related_tasks", prediction.get("evidence", []))
    return [entry["task_id"] for entry in entries if entry.get("task_id")]


def comparison_options(db, task_id):
    rows = db.execute("""SELECT id, title, category, day FROM tasks
                         WHERE status='completed' AND actual_seconds>0 AND id!=?
                         ORDER BY day DESC, title, id""", (task_id,))
    return [dict(row) for row in rows]


def save_choices(db, task_id, selected, previous, reviewed, now):
    """Pin the reviewed pool, learning only from explicit additions and removals."""
    if len(set(selected)) != len(selected):
        raise ValueError("A comparison can only be selected once.")
    eligible = {row["id"] for row in comparison_options(db, task_id)}
    reviewed = set(reviewed)
    if reviewed != eligible:
        raise ValueError("Completed history changed. Reopen the editor to review the available tasks.")
    selected = set(selected)
    if not selected.issubset(reviewed):
        raise ValueError("A selected task is no longer available as completed history.")
    previous = set(previous)
    decisions = [(task_id, source_id, source_id in selected,
                  (source_id in selected) != (source_id in previous), now) for source_id in reviewed]
    db.executemany("""INSERT INTO relevance_choices(target_id,source_id,related,user_changed,changed_at)
                      VALUES(?,?,?,?,?)
                      ON CONFLICT(target_id,source_id) DO UPDATE SET
                        related=excluded.related,
                        user_changed=MAX(relevance_choices.user_changed,excluded.user_changed),
                        changed_at=CASE WHEN excluded.user_changed=1 THEN excluded.changed_at
                                        ELSE relevance_choices.changed_at END""", decisions)
    # Reuse reviewed decisions when a corrected activity report changes a source later.
    existing = db.execute("""SELECT CASE WHEN left_id=? THEN right_id ELSE left_id END
                             FROM task_relations WHERE left_id=? OR right_id=?""", (task_id, task_id, task_id))
    affected = reviewed.intersection(selected | previous | {row[0] for row in existing})
    pairs = [(*sorted((task_id, source_id)), source_id in selected) for source_id in affected]
    db.executemany("""INSERT INTO task_relations(left_id,right_id,related) VALUES(?,?,?)
                      ON CONFLICT(left_id,right_id) DO UPDATE SET related=excluded.related, packaged_fallback=0, refresh_due=0,
                         token=NULL, lease_end=NULL, failures=0, retry_at=0""", pairs)
    db.executemany("DELETE FROM task_relation_sources WHERE left_id=? AND right_id=?",
                   [(left, right) for left, right, _ in pairs])


def _descriptor(row, prefix):
    record = {field: row[prefix + field][:limit] for field, limit in (
        ("title", 150), ("description", 220), ("details", 120), ("category", 60))}
    record["pdf_text"] = document_excerpt({**record, "pdf_text": row[prefix + "pdf_text"]}, 240)
    return record


def relevance_examples(db, target, peer=None):
    """Prefer examples from either participant's category for an undirected matrix cell."""
    peer = peer or target
    participants = (target.get("id") or "", peer.get("id") or "")
    categories = (target.get("category", ""), peer.get("category", ""))
    examples = []
    for verdict in (1, 0):
        rows = db.execute("""
            SELECT f.target_id, f.source_id, f.related, f.changed_at,
                   a.title, a.description, a.details, a.category, a.pdf_text,
                   b.title AS other_title,b.description AS other_description,
                   b.details AS other_details,b.category AS other_category,b.pdf_text AS other_pdf_text
            FROM relevance_choices f JOIN tasks a ON a.id=f.target_id JOIN tasks b ON b.id=f.source_id
            WHERE f.user_changed=1 AND f.related=? AND f.target_id NOT IN (?,?) AND f.source_id NOT IN (?,?)
            ORDER BY (a.category COLLATE NOCASE IN (?,?)) DESC, f.changed_at DESC, a.id, b.id LIMIT 2
        """, (verdict, *participants, *participants, *categories))
        for row in rows:
            examples.append({"target_id": row["target_id"], "candidate_id": row["source_id"],
                             "target": _descriptor(row, ""), "candidate": _descriptor(row, "other_"),
                             "related": bool(row["related"])})
    return examples


def forecast_feedback(db, task_id):
    target = dict(db.execute("SELECT id, category FROM tasks WHERE id=?", (task_id,)).fetchone())
    choices = db.execute("SELECT source_id, related FROM relevance_choices WHERE target_id=?", (task_id,))
    return {"relevance_overrides": {row["source_id"]: bool(row["related"]) for row in choices},
            "relevance_examples": relevance_examples(db, target)}
